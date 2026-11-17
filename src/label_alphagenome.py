"""Model 3 teacher labels from AlphaGenome-FT (MPRA-fine-tuned AlphaGenome encoder), Gosai heads, stage 2.

Runs in the SEPARATE environment ~/PurvoLabs/agft-venv (JAX stack), or on a Kaggle GPU (see the Kaggle folder README).
Recipe = src/practice_checks/alphagenome_predict.py (the teacher check that qualified, commit cadb30a): the 200-letter piece is
flanked with the Gosai vector context (last 200 letters of MPRA_UPSTREAM and first 200 of MPRA_DOWNSTREAM, read from
src/third_party/modeling_mpac.py) to 600 letters; the prediction is the mean of the forward pass and the reverse complement of
the whole 600-letter construct. The model returns bfloat16: every output is cast to float32 BEFORE the two strands are averaged.
Heads: Gosai-HepG2-optimal, Gosai-K562-optimal, Gosai-SKNSH-optimal.

    --part train,mutants,practice13   which tables (data/model3/<part>.parquet; mutants use seq1_mut / seq2_mut)
    --cells HepG2,K562,SKNSH
    --batch 32                        sequences per forward pass (the last batch of a shard is padded, so the jitted function sees one shape)
    --shard 1024                      rows per saved shard; shards are resumable
    --data_dir, --out_dir             default data/model3 (local); on Kaggle: the input dataset folder and /kaggle/working
    --validate                        local check: first 300 rows of practice13, HepG2, version 1, against results/practice/alphagenome_chr13_flanks.npz
    --limit N                         only the first N rows of each part (for tests)
Output: <out_dir>/teacher_ag_<part>.npz with float32 arrays <cell>_v1 and <cell>_v2, row-aligned with the source table
(written when every requested cell is complete for the part). Shards: <out_dir>/ag_shards/<part>_<cell>_v<1|2>_<start>.npy.

Hugging Face login: the base weights google/alphagenome-all-folds are gated. If env HF_TOKEN is missing, the script tries
kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN"), then calls huggingface_hub.login(token=..., add_to_git_credential=False).
The token is never printed. Locally an existing login (cached token) is enough.

Local run (needs the environment variable below because the venv does not find the system certificates):
    SSL_CERT_FILE=$(~/PurvoLabs/agft-venv/bin/python -I -c "import certifi;print(certifi.where())") \
      ~/PurvoLabs/agft-venv/bin/python -I src/label_alphagenome.py --validate
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELL_HEADS = {"HepG2": "Gosai-HepG2-optimal", "K562": "Gosai-K562-optimal", "SKNSH": "Gosai-SKNSH-optimal"}
SOURCES = {"train": ("train.parquet", "seq1", "seq2"), "mutants": ("mutants.parquet", "seq1_mut", "seq2_mut"),
           "practice13": ("practice13.parquet", "seq1", "seq2")}
_MPAC_SRC = (HERE / "third_party" / "modeling_mpac.py").read_text()
UP = _MPAC_SRC.split("MPRA_UPSTREAM = '")[1].split("'")[0][-200:]
DOWN = _MPAC_SRC.split("MPRA_DOWNSTREAM = '")[1].split("'")[0][:200]
assert len(UP) == 200 and len(DOWN) == 200
COMP = str.maketrans("ACGTN", "TGCAN")


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def hf_login() -> None:
    """Make the gated base weights reachable on Kaggle (no-op when HF_TOKEN is set or no secret store exists). Never prints the token."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as e:  # noqa: BLE001  (not on Kaggle, or the secret is not attached: a cached local login may still work)
            print(f"Kaggle secret HF_TOKEN not available: {type(e).__name__}: {str(e)[:200]}", flush=True)
            token = None
    if token:
        import huggingface_hub
        huggingface_hub.login(token=token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = token
        print("Hugging Face login done (token not shown)", flush=True)
        try:
            print("Hugging Face account:", huggingface_hub.whoami().get("name"), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"whoami failed: {type(e).__name__}", flush=True)
    else:
        print("No Hugging Face token found (env HF_TOKEN or Kaggle secret HF_TOKEN)", flush=True)


def load_cell(cell: str):
    """The fine-tuned model and its head name (the loader of src/practice_checks/alphagenome_predict.py)."""
    from alphagenome_ft_mpra.hub import download, load_pretrained
    from huggingface_hub import snapshot_download
    name = CELL_HEADS[cell]
    root = download(name)
    cfg = json.loads((root / "stage2" / "config.json").read_text())
    base = snapshot_download("google/alphagenome-all-folds")
    return load_pretrained(name, stage="stage2", base_checkpoint_path=base), cfg["custom_heads"][0]


def make_predictor(model, head_name: str, batch: int):
    """run(seqs) -> float32 (n,) for 600-letter strings; batched like the authors' _ag_predict_batch, last batch padded."""
    import jax
    import jax.numpy as jnp
    from alphagenome.models import dna_output
    from alphagenome_research.model import dna_model

    organism = dna_model.Organism.HOMO_SAPIENS
    requested = tuple(dna_output.OutputType)
    meta = getattr(model, "_head_configs", {}).get(head_name)
    meta = getattr(meta, "metadata", {}) if meta else {}
    pooling, center_bp = meta.get("pooling_type", "sum"), meta.get("center_bp", 256)

    def run(seqs: list[str]) -> np.ndarray:
        out_all = []
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            n = len(chunk)
            chunk = chunk + [chunk[0]] * (batch - n)  # pad: one compiled shape
            x = jnp.stack([model._one_hot_encoder.encode(s) for s in chunk])
            with model._device_context:
                out = model._predict(model._params, model._state, x, jnp.zeros((batch, 1), dtype=jnp.int32),
                                     requested_outputs=requested, negative_strand_mask=jnp.zeros(batch, dtype=bool),
                                     strand_reindexing=jax.device_put(model._metadata[organism].strand_reindexing,
                                                                      model._device_context._device))
            h = np.asarray(out[head_name])
            h = h.astype(np.float32)  # bfloat16 -> float32 before any arithmetic
            if h.ndim <= 2:  # the Gosai (boda-flatten) head returns one value per sequence
                out_all.append(h.reshape(batch, -1)[:, 0][:n])
                continue
            L = h.shape[1]
            if pooling == "flatten":
                p = h.squeeze(1)
            elif pooling == "center":
                p = h[:, L // 2, :]
            else:
                w = min(max(1, center_bp // 128), L)
                cs = max((L - w) // 2, 0)
                sl = h[:, cs:cs + w, :]
                p = sl.mean(1) if pooling == "mean" else sl.max(1) if pooling == "max" else sl.sum(1)
            out_all.append(p.reshape(batch, -1)[:, 0][:n])
        return np.concatenate(out_all).astype(np.float32)

    return run


def both_strands(run, pieces: list[str]) -> np.ndarray:
    """Mean (in float32) of the forward pass and the reverse complement of the whole flanked construct."""
    fwd = [UP + s + DOWN for s in pieces]
    return ((run(fwd) + run([rc(s) for s in fwd])) / np.float32(2)).astype(np.float32)


def shard_path(out_dir: Path, part: str, cell: str, v: int, start: int) -> Path:
    return out_dir / "ag_shards" / f"{part}_{cell}_v{v}_{start:07d}.npy"


def label(parts: list[str], cells: list[str], data_dir: Path, out_dir: Path, batch: int, shard: int, limit: int | None) -> None:
    import pandas as pd
    (out_dir / "ag_shards").mkdir(parents=True, exist_ok=True)
    tables = {}
    for part in parts:
        f, c1, c2 = SOURCES[part]
        t = pd.read_parquet(data_dir / f, columns=[c1, c2])
        tables[part] = t.iloc[:limit] if limit else t
    total = sum(2 * len(t) for t in tables.values()) * len(cells)
    done, t0 = 0, time.time()
    for cell in cells:
        pending = any(not shard_path(out_dir, p, cell, v, s).exists()
                      for p, t in tables.items() for v in (1, 2) for s in range(0, len(t), shard))
        if not pending:
            done += sum(2 * len(t) for t in tables.values())
            continue
        model, head = load_cell(cell)
        run = make_predictor(model, head, batch)
        print(f"[{cell}] model loaded, head {head}", flush=True)
        for part, t in tables.items():
            _, c1, c2 = SOURCES[part]
            for v, col in ((1, c1), (2, c2)):
                seqs = t[col].tolist()
                for start in range(0, len(seqs), shard):
                    p = shard_path(out_dir, part, cell, v, start)
                    n = min(shard, len(seqs) - start)
                    if p.exists():
                        done += n
                        continue
                    tt = time.time()
                    pred = both_strands(run, seqs[start:start + shard])
                    assert pred.shape == (n,) and np.isfinite(pred).all() and pred.dtype == np.float32
                    np.save(str(p) + ".tmp.npy", pred)
                    Path(str(p) + ".tmp.npy").rename(p)
                    done += n
                    print(f"[{cell}] {part} v{v} rows {start}-{start + n}: {n / (time.time() - tt):.1f} seq/s; "
                          f"{done:,}/{total:,}; elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        del model, run
    for part, t in tables.items():
        assemble(part, cells, len(t), out_dir, shard)


def assemble(part: str, cells: list[str], n: int, out_dir: Path, shard: int) -> None:
    out = {}
    for cell in cells:
        for v in (1, 2):
            files = [shard_path(out_dir, part, cell, v, s) for s in range(0, n, shard)]
            if not all(p.exists() for p in files):
                print(f"{part}: {cell} v{v} incomplete, not assembling", flush=True)
                return
            arr = np.concatenate([np.load(p) for p in files])
            assert arr.shape == (n,) and arr.dtype == np.float32
            out[f"{cell}_v{v}"] = arr
    if set(cells) == set(CELL_HEADS):
        np.savez(out_dir / f"teacher_ag_{part}.npz", **out)
        print("wrote", out_dir / f"teacher_ag_{part}.npz", flush=True)


def validate(data_dir: Path, batch: int) -> None:
    """First 300 rows of practice13, HepG2, version 1, against the stored bfloat16 outputs of the teacher check."""
    import ml_dtypes
    import pandas as pd
    ref = np.load(HERE.parent / "results/practice/alphagenome_chr13_flanks.npz")
    old = np.ascontiguousarray(ref["HepG2_v1"]).view(ml_dtypes.bfloat16).astype(np.float32)
    t = pd.read_parquet(data_dir / "practice13.parquet", columns=["seq1", "variant"])
    old_t = pd.read_parquet(HERE.parent / "data/more_data/practice.parquet", columns=["chr", "variant"])
    old_t = old_t[old_t.chr.astype(str) == "13"].reset_index(drop=True)
    assert (old_t.variant.to_numpy()[:300] == t.variant.to_numpy()[:300]).all(), "row order differs from the teacher check"
    t = t.iloc[:300]
    hf_login()
    model, head = load_cell("HepG2")
    run = make_predictor(model, head, batch)
    t0 = time.time()
    new = both_strands(run, t.seq1.tolist())
    ref300 = old[:300]
    d = np.abs(new - ref300)
    rel = d / np.maximum(np.maximum(np.abs(new), np.abs(ref300)), 1e-6)
    eps = 2.0 ** -8  # relative rounding step of bfloat16
    print(f"validation: 300 rows, {time.time() - t0:.0f} s; max |new - stored| = {d.max():.5f}, mean {d.mean():.5f}; "
          f"max relative {rel.max():.4f} (one bfloat16 rounding = {eps:.4f}); "
          f"rows within 2 roundings (rel <= {2 * eps:.4f}): {(rel <= 2 * eps).mean():.3f}; within 4 roundings: {(rel <= 4 * eps).mean():.3f}; "
          f"Pearson {np.corrcoef(new, ref300)[0, 1]:.6f}; value range {new.min():.2f} to {new.max():.2f}")
    res = {"rows": 300, "max_abs": float(d.max()), "mean_abs": float(d.mean()), "max_rel": float(rel.max()),
           "frac_within_2_bf16_roundings": float((rel <= 2 * eps).mean()), "frac_within_4_bf16_roundings": float((rel <= 4 * eps).mean()),
           "pearson": float(np.corrcoef(new, ref300)[0, 1])}
    out = HERE.parent / "results/model3_ag_validation.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="train,mutants,practice13")
    ap.add_argument("--cells", default="HepG2,K562,SKNSH")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard", type=int, default=1024)
    ap.add_argument("--data_dir", default=str(HERE.parent / "data/model3"))
    ap.add_argument("--out_dir", default=str(HERE.parent / "data/model3"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate(Path(a.data_dir), a.batch)
    else:
        hf_login()
        label(a.part.split(","), a.cells.split(","), Path(a.data_dir), Path(a.out_dir), a.batch, a.shard, a.limit)

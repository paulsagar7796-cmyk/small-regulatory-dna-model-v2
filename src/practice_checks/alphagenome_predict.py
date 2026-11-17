"""Step 1 of the AlphaGenome teacher check (predeclarations/TEACHER_CHECK_ALPHAGENOME.md). Inference only.
Runs in the SEPARATE environment ~/PurvoLabs/agft-venv (JAX stack), from the repo root:

    ~/PurvoLabs/agft-venv/bin/python -I src/practice_checks/alphagenome_predict.py --format-check
    ~/PurvoLabs/agft-venv/bin/python -I src/practice_checks/alphagenome_predict.py --format flanks|alone

Reads practice chromosome 13 only. Writes results/practice/alphagenome_chr13_{format}.npz with, per cell, predictions
for version 1 and version 2 of every chromosome-13 variant (both strands averaged)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
TABLE = {"HepG2": "HEPG2", "K562": "K562"}
# Gosai vector flanks, copied from src/third_party/modeling_mpac.py (MPAC's add_flanks uses the last 200 / first 200)
UP = open(ROOT / "src/third_party/modeling_mpac.py").read().split("MPRA_UPSTREAM = '")[1].split("'")[0][-200:]
DOWN = open(ROOT / "src/third_party/modeling_mpac.py").read().split("MPRA_DOWNSTREAM = '")[1].split("'")[0][:200]
COMP = str.maketrans("ACGTN", "TGCAN")


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def load(cell: str):
    from alphagenome_ft_mpra.hub import download, load_pretrained
    root = download(f"Gosai-{cell}-optimal")
    cfg = json.loads((root / "stage2" / "config.json").read_text())
    print(f"[{cell}] config:", json.dumps(cfg)[:1500], flush=True)
    return load_pretrained(f"Gosai-{cell}-optimal", stage="stage2"), cfg


def predictor(model, head_name: str, batch: int = 32):
    """Batched version of _ag_predict_batch in the authors' scripts/test_episomal_mpra.py (same calls, same pooling)."""
    import jax
    import jax.numpy as jnp
    from alphagenome.models import dna_output
    from alphagenome_research.model import dna_model

    organism = dna_model.Organism.HOMO_SAPIENS
    requested = tuple(dna_output.OutputType)
    meta = getattr(model, "_head_configs", {}).get(head_name)
    meta = getattr(meta, "metadata", {}) if meta else {}
    pooling, center_bp = meta.get("pooling_type", "sum"), meta.get("center_bp", 256)
    print("head metadata:", meta, flush=True)

    def run(seqs: list[str]) -> np.ndarray:
        out_all = []
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            x = jnp.stack([model._one_hot_encoder.encode(s) for s in chunk])
            n = len(chunk)
            with model._device_context:
                out = model._predict(model._params, model._state, x, jnp.zeros((n, 1), dtype=jnp.int32),
                                     requested_outputs=requested, negative_strand_mask=jnp.zeros(n, dtype=bool),
                                     strand_reindexing=jax.device_put(model._metadata[organism].strand_reindexing,
                                                                      model._device_context._device))
            h = np.array(out[head_name])
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
            out_all.append(p.reshape(n, -1)[:, 0])
        return np.concatenate(out_all)
    return run


def build(seqs, fmt):
    return [UP + s + DOWN for s in seqs] if fmt == "flanks" else list(seqs)


def both_strands(run, seqs):
    return (run(seqs) + run([rc(s) for s in seqs])) / 2


def main():
    pr = pd.read_parquet(ROOT / "data/more_data/practice.parquet")
    pr = pr[pr.chr.astype(str) == "13"].reset_index(drop=True)
    print("chromosome 13 variants:", len(pr), flush=True)
    fmt_check = "--format-check" in sys.argv
    fmt = None if fmt_check else sys.argv[sys.argv.index("--format") + 1]
    out = {}
    for cell, tab in TABLE.items():
        model, cfg = load(cell)
        head_name = cfg["custom_heads"][0]
        run = predictor(model, head_name)
        if fmt_check:
            t = pr.iloc[:300]
            y = t[f"allele1.log2FC_{tab}"].to_numpy()
            ok = ~np.isnan(y)
            for f in ("flanks", "alone"):
                p = both_strands(run, build(t.seq1.tolist(), f))
                print(f"[{cell}] format {f}: activity Pearson on first 300 pieces {np.corrcoef(y[ok], p[ok])[0, 1]:.4f}", flush=True)
            continue
        p1 = both_strands(run, build(pr.seq1.tolist(), fmt))
        p2 = both_strands(run, build(pr.seq2.tolist(), fmt))
        a1, a2 = pr[f"allele1.log2FC_{tab}"].to_numpy(), pr[f"allele2.log2FC_{tab}"].to_numpy()
        ok = ~np.isnan(a1) & ~np.isnan(a2)
        act = np.corrcoef(np.concatenate([a1[ok], a2[ok]]), np.concatenate([p1[ok], p2[ok]]))[0, 1]
        print(f"[{cell}] activity Pearson, all chromosome-13 pieces: {act:.4f} (sanity line 0.80)", flush=True)
        out[f"{cell}_v1"], out[f"{cell}_v2"], out[f"{cell}_activity"] = p1, p2, act
    if not fmt_check:
        np.savez(ROOT / f"results/practice/alphagenome_chr13_{fmt}.npz", ids=pr.index.to_numpy(), **out)


if __name__ == "__main__":
    main()

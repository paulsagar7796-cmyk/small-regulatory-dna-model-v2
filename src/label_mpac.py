"""Model 3 teacher labels from MPAC: the 10 networks of MPAC's (7, 16) fold ONLY (their test pair holds chromosome 7, so they
never trained on chromosome 7; they trained on chromosome 13, so chromosome-13 labels are NOT clean and are for checks only).

For every row of data/model3/train.parquet, data/model3/mutants.parquet and data/model3/practice13.parquet, version 1 and
version 2: the 10-network mean through MPAC's own predict (vendored src/third_party/modeling_mpac.py): the 200-letter piece
is flanked with MPAC's vector context, run forward and as the reverse complement of the insert only, and the two passes are
averaged. One centred 200-letter window per version.

Outputs (resumable): shards in data/model3/mpac_shards/<part>_v<1|2>_<start>.npy (float32, rows x 3, columns HepG2, K562, SKNSH),
then data/model3/teacher_mpac.npz with float32 arrays, row-aligned with the source table (mutants.parquet is row-aligned with
train.parquet):
    <part>_<cell>_v<1|2>     part in {train, mutants, practice13}, cell in {HepG2, K562, SKNSH}   shape (rows,)
    cells, parts, fold, members  (small metadata arrays)

    --check      MPACEnsemble output against the plain mean of the 10 single networks, on 64 practice-13 pieces, then exit
    --bench      time one shard of 2,048 pieces, then exit
    (no flag)    label everything (skips shards that exist), then assemble the npz
    --parts a,b  restrict to some parts;  --shard N  rows per shard (default 2048, keep it the same when resuming);  --batch N  (default 256)
Run from the repository root with the main venv:  nohup ~/PurvoLabs/small-dna-model/.venv/bin/python -I src/label_mpac.py > runs/label_mpac.log 2>&1 &
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
import modeling_mpac as mm  # noqa: E402
from mpac_reference import MPAC_DIR, load_ensemble, provenance, verify_fold  # noqa: E402

DATA = Path("data/model3")
SHARDS = DATA / "mpac_shards"
FOLD_CHROM = "7"
CELLS = ["HepG2", "K562", "SKNSH"]
SOURCES = {"train": ("train.parquet", "seq1", "seq2"), "mutants": ("mutants.parquet", "seq1_mut", "seq2_mut"),
           "practice13": ("practice13.parquet", "seq1", "seq2")}


def pick_device() -> torch.device:
    """MPS if it gives the same numbers as CPU on a small batch (checked on practice-13 pieces), else CPU."""
    if not torch.backends.mps.is_available():
        return torch.device("cpu")
    try:
        seqs = pd.read_parquet(DATA / "practice13.parquet", columns=["seq1"]).seq1.tolist()[:16]
        a = load_ensemble(FOLD_CHROM, torch.device("cpu")).predict(seqs, device=torch.device("cpu")).numpy()
        b = load_ensemble(FOLD_CHROM, torch.device("mps")).predict(seqs, device=torch.device("mps")).numpy()
        print("MPS vs CPU max abs difference:", float(np.abs(a - b).max()), flush=True)
        return torch.device("mps") if np.abs(a - b).max() < 1e-3 else torch.device("cpu")
    except Exception as e:  # noqa: BLE001
        print("MPS not usable, using CPU:", type(e).__name__, str(e)[:100], flush=True)
        return torch.device("cpu")


def column_order() -> list[int]:
    names = json.loads((MPAC_DIR / "config.json").read_text())["output_names"]
    return [names.index(c) for c in CELLS]  # read from the config, not assumed


def check() -> None:
    prov, config = provenance(), json.loads((MPAC_DIR / "config.json").read_text())
    fold, members = verify_fold(FOLD_CHROM, prov)
    assert all(FOLD_CHROM in r["test_chrs"] for r in members) and len(members) == 10
    print("fold", fold, "test pair", members[0]["test_chrs"], "validation pairs", sorted(tuple(r["val_chrs"]) for r in members))
    seqs = pd.read_parquet(DATA / "practice13.parquet", columns=["seq1"]).seq1.tolist()[:64]
    dev = torch.device("cpu")
    singles = []
    for r in members:
        net = mm.MPACModel(**config)
        net.load_state_dict(load_file(str(MPAC_DIR / r["file"])))
        singles.append(net.eval().predict(seqs, device=dev).numpy())
    mean_single = np.mean(singles, axis=0)
    ens = load_ensemble(FOLD_CHROM, dev).predict(seqs, device=dev).numpy()
    d = float(np.abs(ens - mean_single).max())
    print(f"max |MPACEnsemble - mean of 10 single networks| over 64 pieces x 3 outputs: {d:.2e}; output scale {float(np.abs(ens).max()):.2f}")
    assert d < 1e-4, "ensemble does not equal the per-network average"
    print("CHECK PASSED")


def shard_path(part: str, v: int, start: int) -> Path:
    return SHARDS / f"{part}_v{v}_{start:07d}.npy"


def run(parts: list[str], shard: int, batch: int, bench: bool) -> None:
    SHARDS.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print("device", device, flush=True)
    ens = load_ensemble(FOLD_CHROM, device)
    order = column_order()
    jobs = []
    for part in parts:
        f, c1, c2 = SOURCES[part]
        t = pd.read_parquet(DATA / f, columns=[c1, c2])
        for v, col in ((1, c1), (2, c2)):
            jobs.append((part, v, t[col].tolist()))
    total = sum(len(s) for _, _, s in jobs)
    done, t0 = 0, time.time()
    for part, v, seqs in jobs:
        for start in range(0, len(seqs), shard):
            p = shard_path(part, v, start)
            n = min(shard, len(seqs) - start)
            if p.exists():
                done += n
                continue
            tt = time.time()
            pred = ens.predict(seqs[start:start + shard], batch_size=batch, device=device).numpy()[:, order].astype(np.float32)
            assert pred.shape == (n, 3) and np.isfinite(pred).all()
            np.save(str(p) + ".tmp.npy", pred)
            Path(str(p) + ".tmp.npy").rename(p)
            done += n
            el = time.time() - t0
            print(f"{part} v{v} rows {start}-{start + n}: {n / (time.time() - tt):.0f} seq/s; {done:,}/{total:,} done; "
                  f"elapsed {el / 60:.1f} min", flush=True)
            if bench:
                return
    assemble(parts)


def assemble(parts: list[str]) -> None:
    prov = provenance()
    fold, members = verify_fold(FOLD_CHROM, prov)
    out = {"cells": np.array(CELLS), "parts": np.array(parts), "fold": np.array(fold), "members": np.array([r["file"] for r in members])}
    for part in parts:
        f, c1, _ = SOURCES[part]
        n = len(pd.read_parquet(DATA / f, columns=[c1]))
        for v in (1, 2):
            files = sorted(SHARDS.glob(f"{part}_v{v}_[0-9]*.npy"))
            files = [p for p in files if not p.name.endswith(".tmp.npy")]
            arr = np.concatenate([np.load(p) for p in files])
            assert arr.shape == (n, 3), f"{part} v{v}: {arr.shape} rows, expected {n}"
            for k, c in enumerate(CELLS):
                out[f"{part}_{c}_v{v}"] = np.ascontiguousarray(arr[:, k], dtype=np.float32)
    np.savez(DATA / "teacher_mpac.npz", **out)
    print("wrote", DATA / "teacher_mpac.npz", {k: v.shape for k, v in out.items() if v.ndim == 1 and len(v) > 10}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--assemble_only", action="store_true")
    ap.add_argument("--parts", default="practice13,train,mutants")
    ap.add_argument("--shard", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    if a.check:
        check()
    elif a.assemble_only:
        assemble(a.parts.split(","))
    else:
        run(a.parts.split(","), a.shard, a.batch, a.bench)

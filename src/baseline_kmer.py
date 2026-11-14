"""Simple baseline for Path B: count every 6-letter word in the DNA piece (both strands), then ridge regression.
No deep learning. A model of ours has to beat this clearly to be worth anything.

Run from the repository root:  .venv/bin/python src/baseline_kmer.py            (validation fold only)
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
from mpra_data import CELLS, TRAIN_FOLDS, VALIDATION_FOLD, element_pearson, encode, load  # noqa: E402

K = 6


def kmer_counts(codes: np.ndarray) -> np.ndarray:
    n, length = codes.shape
    idx = np.zeros((n, length - K + 1), dtype=np.int64)
    for j in range(K):
        idx = idx * 4 + codes[:, j : length - K + 1 + j]
    out = np.zeros((n, 4**K), dtype=np.float32)
    np.add.at(out, (np.repeat(np.arange(n), idx.shape[1]), idx.ravel()), 1.0)
    return out


def features(codes: np.ndarray) -> np.ndarray:
    rc = (3 - codes)[:, ::-1]
    return kmer_counts(codes) + kmer_counts(rc)


def main() -> None:
    results = {}
    for cell in CELLS:
        tr, va = load(cell, TRAIN_FOLDS), load(cell, [VALIDATION_FOLD])
        model = Ridge(alpha=100.0).fit(features(encode(tr.seq)), tr.mean_value.to_numpy())
        results[cell] = element_pearson(va, model.predict(features(encode(va.seq))))
        print(f"{cell}: 6-mer ridge, validation fold {VALIDATION_FOLD}: element Pearson {results[cell]:.3f}", flush=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/baseline_kmer_validation.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

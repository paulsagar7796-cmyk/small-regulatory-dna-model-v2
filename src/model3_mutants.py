"""Model 3: one fixed mutated copy of BOTH versions of every training variant (pre-declaration: 'one fixed copy of each piece
with 5% of letters changed'). For variant j both versions get the SAME mask: 10 of the 200 positions (5%), never the variant
position (the one letter where version 1 and version 2 differ), each changed to a different random letter. So the two mutated
copies still differ only at the variant position, and the paired effect of the variant is kept.

Seed 20261118. Reads data/model3/train.parquet (sequences only), writes data/model3/mutants.parquet with columns
    row        position of the variant in train.parquet (0-based)
    seq1_mut, seq2_mut
Run from the repository root:
    ~/PurvoLabs/small-dna-model/.venv/bin/python -I src/model3_mutants.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED, N_MUT, LENGTH = 20261118, 10, 200
LETTERS = np.frombuffer(b"ACGT", dtype=np.uint8)
CODE = np.full(256, 255, np.uint8)
CODE[LETTERS] = np.arange(4)


def to_codes(seqs) -> np.ndarray:
    a = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), -1)
    c = CODE[a]
    assert (c < 4).all(), "only A, C, G, T allowed"
    return c


def to_strings(codes: np.ndarray) -> list[str]:
    return [b.decode() for b in LETTERS[codes].view(f"S{codes.shape[1]}").ravel()]


def mutate(seq1, seq2, seed: int = SEED):
    """Returns (codes1_mut, codes2_mut, mask) for lists of 200-letter strings; mask is (n, 200) bool, True = changed."""
    a, b = to_codes(seq1), to_codes(seq2)
    n = len(a)
    diff = a != b
    assert (diff.sum(1) == 1).all(), "every variant must be a single-letter change"
    var_pos = diff.argmax(1)
    rng = np.random.default_rng(seed)
    keys = rng.random((n, LENGTH))
    keys[np.arange(n), var_pos] = np.inf  # the variant position is never picked
    chosen = np.argsort(keys, axis=1)[:, :N_MUT]
    mask = np.zeros((n, LENGTH), bool)
    mask[np.arange(n)[:, None], chosen] = True
    shift = rng.integers(1, 4, size=(n, LENGTH), dtype=np.uint8)  # 1, 2 or 3: a different letter
    a_mut = np.where(mask, (a + shift) % 4, a).astype(np.uint8)
    b_mut = np.where(mask, (b + shift) % 4, b).astype(np.uint8)  # SAME new letters; at unmasked positions b keeps its own letter
    return a_mut, b_mut, mask


def main() -> None:
    src, out = Path("data/model3/train.parquet"), Path("data/model3/mutants.parquet")
    t = pd.read_parquet(src, columns=["seq1", "seq2"])
    a_mut, b_mut, mask = mutate(t.seq1.tolist(), t.seq2.tolist())
    pd.DataFrame({"row": np.arange(len(t)), "seq1_mut": to_strings(a_mut), "seq2_mut": to_strings(b_mut)}).to_parquet(out, index=False)
    print(f"wrote {out}: {len(t):,} rows; mutated positions per piece: {int(mask.sum(1).min())} to {int(mask.sum(1).max())}")


if __name__ == "__main__":
    main()

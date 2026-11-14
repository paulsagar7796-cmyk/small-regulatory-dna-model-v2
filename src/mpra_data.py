"""Path B data: lentiMPRA lab measurements (Agarwal and others, Nature 2025), as packaged with MPRALegNet
(github.com/autosome-ru/human_legnet, MIT licence), files datasets/original/{HepG2,K562,WTC11}.tsv.

Each row is one 230-letter DNA piece (a 200-letter element between two fixed 15-letter adapters) and its
measured activity in one cell type (log2 of RNA over DNA: higher = the piece switches a gene on more).
Every element was measured in both orientations. The files also hold a reverse-complement copy of each row
(rev = 1) with the same value; we drop those and do reverse-complement augmentation ourselves.

Split (fixed): the dataset's own 10 folds. Folds 1 to 8 train, fold 9 validates, fold 10 is the LOCKED TEST.
Both orientations of an element are always in the same fold (checked).
"""

from pathlib import Path

import numpy as np
import pandas as pd

MPRA_DIR = Path("data/mpra")
CELLS = ["HepG2", "K562", "WTC11"]
TRAIN_FOLDS, VALIDATION_FOLD, TEST_FOLD = list(range(1, 9)), 9, 10
_CODE = np.full(256, 4, dtype=np.uint8)
for i, b in enumerate("ACGT"):
    _CODE[ord(b)] = i


def load(cell: str, folds: list[int]) -> pd.DataFrame:
    """Rows of the given folds, one per measured orientation, with an `element` id shared by both orientations."""
    d = pd.read_csv(MPRA_DIR / f"{cell}.tsv", sep="\t")
    d = d[(d.rev == 0) & d.fold.isin(folds)].reset_index(drop=True)
    d["element"] = d.seq_id.str.replace(r"(_Reversed:|_F|_R)$", "", regex=True)
    return d


def encode(seqs: pd.Series) -> np.ndarray:
    """(n, 230) uint8 codes: A0 C1 G2 T3 (anything else 4)."""
    return np.stack([_CODE[np.frombuffer(s.encode(), dtype=np.uint8)] for s in seqs])


def element_pearson(d: pd.DataFrame, pred: np.ndarray) -> float:
    """The paper's measure: average prediction and measurement over both orientations of each element, then Pearson r."""
    g = pd.DataFrame({"element": d.element.to_numpy(), "y": d.mean_value.to_numpy(), "p": pred}).groupby("element").mean()
    return float(np.corrcoef(g.y, g.p)[0, 1])

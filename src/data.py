"""Small sequence helpers shared by the training and evaluation scripts.

DNA letters are stored as codes: A 0, C 1, G 2, T 3, anything else 4.
"""

from pathlib import Path

import numpy as np
import torch

GENOME_DIR = Path("data/genome")  # hg38, one uint8 array per chromosome (see DATA.md); needed only for the Kircher test
_GENOME_CACHE: dict[str, np.ndarray] = {}


def genome(chrom: str) -> np.ndarray:
    if chrom not in _GENOME_CACHE:
        _GENOME_CACHE[chrom] = np.load(GENOME_DIR / f"chr{chrom}.npy", mmap_mode="r")
    return _GENOME_CACHE[chrom]


def onehot(codes: torch.Tensor) -> torch.Tensor:
    """(batch, length) uint8 codes -> (batch, 4, length) float; N (code 4) becomes all zeros."""
    eye = torch.cat([torch.eye(4), torch.zeros(1, 4)])
    return eye.to(codes.device)[codes.long()].transpose(1, 2)


def reverse_complement_codes(codes: torch.Tensor) -> torch.Tensor:
    comp = torch.tensor([3, 2, 1, 0, 4], dtype=codes.dtype, device=codes.device)  # A<->T, C<->G, N->N
    return comp[codes.long()].flip(-1)

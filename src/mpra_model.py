"""Path B student: the same small convolutional body as the D3 student, with a new head.

Body (unchanged from src/model.py; this file has its own name so the two do not clash): 128 channels,
12 dilated residual blocks, reads one-hot DNA letters.
Head (new): average and maximum over all positions, then one linear layer giving ONE number per DNA piece,
the predicted lab activity. About 1.19 million parameters.

Options added for the v1 improvement runs (defaults reproduce v0 exactly, with the same weight names):
  dropout: randomly silences part of each block's update during training, to reduce memorising
  n_out:   several outputs, one per cell type, for a body shared across cell types
"""

import sys

import torch
from torch import nn

from model import ResidualBlock, count_parameters  # noqa: E402,F401


class DropBlock(ResidualBlock):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__(channels, kernel, dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(self.act(self.norm(x)))
        return x + self.drop(self.mix(self.act(h)))


class ActivityStudent(nn.Module):
    def __init__(self, channels: int = 128, kernel: int = 5, dilations=(1, 2, 4, 8, 16, 32) * 2,
                 dropout: float = 0.0, n_out: int = 1):
        super().__init__()
        self.embed = nn.Conv1d(4, channels, kernel, padding=kernel // 2)
        self.blocks = nn.Sequential(*[DropBlock(channels, kernel, d, dropout) for d in dilations])
        self.norm = nn.GroupNorm(1, channels)
        self.head = nn.Linear(2 * channels, n_out)

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        """onehot: (batch, 4, length) -> predicted activity: (batch,) or (batch, n_out)"""
        h = torch.nn.functional.gelu(self.norm(self.blocks(self.embed(onehot))))
        out = self.head(torch.cat([h.mean(2), h.amax(2)], dim=1))
        return out.squeeze(1) if out.shape[1] == 1 else out

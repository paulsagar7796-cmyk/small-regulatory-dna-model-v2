"""Building block of the lab's first (v0 and v1) students: a dilated residual convolution block.
The released v2 student does not use it; it uses the published MPRA-LegNet design (src/third_party/legnet_model.py).
"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(self.act(self.norm(x)))
        h = self.mix(self.act(h))
        return x + h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

"""The Malinois network design (the body of the competitor MPAC) as a drop-in body for train_more_data.py.

Design and settings come from the vendored code src/third_party/modeling_mpac.py (class MPACModel = Malinois' BassetBranched) and the
published configuration data/models/MPAC/config.json: three convolution blocks (300 channels, kernel 19; 200, kernel 11; 200, kernel 7)
with batch norm and max-pooling (3, 4, 4), one 1000-wide linear layer (dropout 0.1163), a branched tower of 3 layers x 140 channels per
output (dropout 0.5757, ReLU), batch norm on, weight norm off, input 600 letters. Only n_outputs differs from the published file
(3 there, ours here); the weights are randomly initialised (PyTorch defaults), never the published ones.

Input adapter (MalinoisBody.forward): our pipeline hands the model rows of 230 letters, a 15-letter adapter + 200-letter insert +
15-letter adapter, already slid and strand-flipped. The wrapper cuts out letters 15..215 and pads them to 600 with the SAME fixed vector
flanks that MPAC's own predict() uses (200 letters of MPRA_UPSTREAM before, 200 of MPRA_DOWNSTREAM after).
"""

import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent / "third_party"))
import modeling_mpac as mm  # noqa: E402  (vendored, read in full before use)

CONFIG = Path("data/models/MPAC/config.json")  # published configuration (run from the repository root)
ADAPTER = 15  # letters of adapter on each side of the 200-letter insert in our 230-letter rows

# the published configuration, written out so that the code also runs without data/ (checked equal to CONFIG in the tests)
PUBLISHED = {"branched_activation": "ReLU", "branched_channels": 140, "branched_dropout_p": 0.5757068086404574,
             "conv1_channels": 300, "conv1_kernel_size": 19, "conv2_channels": 200, "conv2_kernel_size": 11,
             "conv3_channels": 200, "conv3_kernel_size": 7, "input_len": 600, "linear_activation": "ReLU",
             "linear_channels": 1000, "linear_dropout_p": 0.11625456877954289, "n_branched_layers": 3, "n_linear_layers": 1,
             "use_batch_norm": True, "use_weight_norm": False, "variable_region_len": 200}


def published_config() -> dict:
    """The published settings (from data/models/MPAC/config.json if it is there, else the copy above), without n_outputs / output_names."""
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else dict(PUBLISHED)
    cfg.pop("n_outputs", None)
    cfg.pop("output_names", None)
    return cfg


class MalinoisBody(nn.Module):
    """A randomly initialised Malinois network with n_out outputs, reading our 230-letter rows (batch, 4, 230) in A C G T order."""

    def __init__(self, n_out: int):
        super().__init__()
        self.net = mm.MPACModel(**published_config(), n_outputs=n_out)

    def window(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 4, 230) -> (batch, 4, 600): the middle 200 letters with the fixed vector flanks added."""
        assert x.shape[-1] == 2 * ADAPTER + self.net.variable_region_len, x.shape
        return self.net.add_flanks(x[..., ADAPTER:-ADAPTER])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.window(x))


def build_arch(arch: str, n_out: int, width: float = 1.0) -> nn.Module:
    """The body for --arch: legnet (the default design of models 1 and 2, via mpra_train_v2.build) or malinois."""
    if arch == "malinois":
        assert width == 1.0, "--width only applies to the legnet body"
        return MalinoisBody(n_out)
    from mpra_train_v2 import build  # imported here: mpra_train_v2 needs data helpers this module does not
    return build(arch, n_out, width=width)

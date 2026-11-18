"""PyramidLegNet: the MPRA-LegNet body with three optional changes, each switchable so they can be screened one at a time.

    1. multi-scale stem   three parallel convolutions (kernels 5, 9, 15) -> concatenate -> BatchNorm -> SiLU -> squeeze-excitation
                          (LegNet: one convolution, kernel 11 -> BatchNorm -> SiLU)
    2. feature pyramid    the readout sees blocks 2, 3 and 4 (each taken BEFORE its max-pool, i.e. at 115, 57 and 28 positions),
                          each projected to a common width by BatchNorm + 1x1 convolution and averaged over positions, then
                          concatenated (LegNet: only the final 14-position map, through one 1x1 "mapper", averaged)
    3. assay adapters     after the shared readout layer, one small residual bottleneck per assay family (lentiMPRA outputs,
                          plasmid outputs), each followed by its own final linear layer
                          (LegNet: one final linear layer for all outputs)

With all three switched off the module IS LegNet: the same layers in the same order, so LegNet's weights load into it and give
identical outputs (checked in src/test_pyramid_legnet.py).

Input  (batch, 4, length) one-hot DNA, any length (our rows are 230 letters: 15-letter adapter + 200-letter piece + adapter).
Output (batch, n_out). The first n_lenti columns are the lentiMPRA outputs, the rest the plasmid outputs: the order our trainers
use (CELLS + new_*, i.e. HepG2, K562, WTC11, new_HEPG2, new_K562, new_SKNSH, new_HCT116, new_A549).

The LegNet building blocks below are copied from github.com/autosome-ru/human_legnet (MIT licence), as in
src/third_party/legnet_model.py. Only torch is needed.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# the published MPRA-LegNet settings (src/mpra_train_v2.py build("legnet"))
LEGNET_STEM_CH, LEGNET_STEM_KS = 64, 11
LEGNET_KS = 9
LEGNET_BLOCKS = (80, 96, 112, 128)
LEGNET_POOLS = (2, 2, 2, 2)
LEGNET_RESIZE = 4


def init_like_legnet(m: nn.Module) -> None:
    """LegNet's initialisation: convolutions N(0, sqrt(2 / (kernel * out))), BatchNorm (1, 0), linear layers N(0, 0.001)."""
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------- LegNet building blocks (unchanged)
class SELayer(nn.Module):
    """Squeeze-and-excitation: a per-channel gate computed from the channel means."""

    def __init__(self, inp: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(inp, int(inp // reduction)), nn.SiLU(), nn.Linear(int(inp // reduction), inp), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        return x * self.fc(x.view(b, c, -1).mean(dim=2)).view(b, c, 1)


class EffBlock(nn.Module):
    """EfficientNet-style block: expand (1x1) -> depthwise conv -> squeeze-excitation -> project (1x1)."""

    def __init__(self, in_ch, ks, resize_factor, activation, out_ch=None, se_reduction=None):
        super().__init__()
        inner = in_ch * resize_factor
        se_reduction = resize_factor if se_reduction is None else se_reduction
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, inner, kernel_size=1, padding="same", bias=False), nn.BatchNorm1d(inner), activation(),
            nn.Conv1d(inner, inner, kernel_size=ks, groups=inner, padding="same", bias=False), nn.BatchNorm1d(inner), activation(),
            SELayer(inner, reduction=se_reduction),
            nn.Conv1d(inner, in_ch, kernel_size=1, padding="same", bias=False), nn.BatchNorm1d(in_ch), activation(),
        )

    def forward(self, x):
        return self.block(x)


class LocalBlock(nn.Module):
    """Convolution -> BatchNorm -> activation."""

    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.block = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding="same", bias=False), nn.BatchNorm1d(out_ch), activation())

    def forward(self, x):
        return self.block(x)


class ResidualConcat(nn.Module):
    """Concatenate a block's output with its input along the channels."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return torch.concat([self.fn(x), x], dim=1)


class MapperBlock(nn.Module):
    """BatchNorm -> 1x1 convolution (LegNet's 'mapper'; also used here to project each pyramid tap)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Sequential(nn.BatchNorm1d(in_features), nn.Conv1d(in_features, out_features, kernel_size=1))

    def forward(self, x):
        return self.block(x)


class Stage(nn.Module):
    """One LegNet block, split into its body and its max-pool so the map before pooling can be tapped."""

    def __init__(self, in_ch, out_ch, ks, resize_factor, pool):
        super().__init__()
        self.body = nn.Sequential(ResidualConcat(EffBlock(in_ch, ks, resize_factor, nn.SiLU, out_ch=in_ch)),
                                  LocalBlock(in_ch * 2, ks, nn.SiLU, out_ch=out_ch))
        self.pool = nn.MaxPool1d(pool) if pool != 1 else nn.Identity()


# ---------------------------------------------------------------- the three changes
class MultiScaleStem(nn.Module):
    """Change 1: parallel convolutions of several kernel sizes, concatenated, then BatchNorm, SiLU and squeeze-excitation."""

    def __init__(self, in_ch=4, branch_ch=(32, 32, 32), kernels=(5, 9, 15), se_reduction=4):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(in_ch, c, kernel_size=k, padding="same", bias=False) for c, k in zip(branch_ch, kernels)])
        out = sum(branch_ch)
        self.post = nn.Sequential(nn.BatchNorm1d(out), nn.SiLU(), SELayer(out, reduction=se_reduction))

    def forward(self, x):
        return self.post(torch.cat([b(x) for b in self.branches], dim=1))


class ResidualAdapter(nn.Module):
    """Change 3: h + up(SiLU(down(h))). `up` starts at zero, so every adapter starts as the identity."""

    def __init__(self, dim, bottleneck):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)

    def reset(self):
        self.down.reset_parameters()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.silu(self.down(h)))


class PyramidLegNet(nn.Module):
    """LegNet with optional multi-scale stem, feature-pyramid readout and per-assay-family adapters (see the module docstring).

    n_out        number of outputs; n_lenti of them (the first ones) are lentiMPRA, the rest plasmid
    width        channel multiplier for every LegNet layer (1.0 = published sizes), as --width in our trainers
    taps         which blocks feed the pyramid (0-based; (1, 2, 3) = blocks 2, 3, 4), each taken before its max-pool
    tap_ch       width of each tap after projection; pool_stats 'mean' (as LegNet) or 'meanstd' (mean and standard deviation)
    adapter_dim  bottleneck width of each adapter
    """

    def __init__(self, n_out: int, n_lenti: int = 3, width: float = 1.0, multiscale_stem: bool = True, pyramid: bool = True,
                 adapters: bool = True, stem_kernels=(5, 9, 15), stem_branch_ch: int = 32, taps=(1, 2, 3), tap_ch: int = 128,
                 pool_stats: str = "mean", adapter_dim: int = 64):
        super().__init__()
        assert 0 < n_lenti <= n_out and pool_stats in ("mean", "meanstd")
        w = (lambda c: c) if width == 1.0 else (lambda c: int(round(c * width / 8)) * 8)
        blocks = [w(c) for c in LEGNET_BLOCKS]
        self.pool_stats = pool_stats
        self.groups = [g for g in (n_lenti, n_out - n_lenti) if g > 0]

        # stem
        if multiscale_stem:
            self.stem = MultiScaleStem(4, [w(stem_branch_ch)] * len(stem_kernels), stem_kernels)
            in_ch = w(stem_branch_ch) * len(stem_kernels)
        else:
            self.stem = LocalBlock(4, LEGNET_STEM_KS, nn.SiLU, out_ch=w(LEGNET_STEM_CH))
            in_ch = w(LEGNET_STEM_CH)

        # trunk: the four LegNet blocks
        stages = []
        for out_ch, pool in zip(blocks, LEGNET_POOLS):
            stages.append(Stage(in_ch, out_ch, LEGNET_KS, LEGNET_RESIZE, pool))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

        # readout features
        head_dim = 2 * blocks[-1]  # 256 at width 1, as LegNet's head
        k = 2 if pool_stats == "meanstd" else 1
        if pyramid:
            self.taps = tuple(taps)
            self.proj = nn.ModuleList([MapperBlock(blocks[i], w(tap_ch)) for i in self.taps])
            readout_in = len(self.taps) * w(tap_ch) * k
        else:
            self.taps = None
            self.mapper = MapperBlock(blocks[-1], head_dim)
            readout_in = head_dim * k

        # shared readout layer, then one final layer for all outputs (LegNet) or one adapter + final layer per assay family
        self.head_shared = nn.Sequential(nn.Linear(readout_in, head_dim), nn.BatchNorm1d(head_dim), nn.SiLU())
        if adapters:
            self.adapters = nn.ModuleList([ResidualAdapter(head_dim, adapter_dim) for _ in self.groups])
            self.outs = nn.ModuleList([nn.Linear(head_dim, g) for g in self.groups])
        else:
            self.adapters = None
            self.out = nn.Linear(head_dim, n_out)

        self.apply(init_like_legnet)
        if self.adapters is not None:
            for a in self.adapters:
                a.reset()

    def _pool(self, x):
        if self.pool_stats == "mean":
            return x.mean(dim=-1)
        mean = x.mean(dim=-1)
        return torch.cat([mean, torch.sqrt(x.var(dim=-1, unbiased=False) + 1e-5)], dim=1)

    def forward(self, x):
        x = self.stem(x)
        tapped = []
        for i, st in enumerate(self.stages):
            x = st.body(x)
            if self.taps is not None and i in self.taps:
                tapped.append(x)  # before the max-pool: 115, 57, 28 positions for blocks 2, 3, 4 at 230 letters
            x = st.pool(x)
        feats = [self.mapper(x)] if self.taps is None else [p(t) for p, t in zip(self.proj, tapped)]
        h = self.head_shared(torch.cat([self._pool(f) for f in feats], dim=1))
        if self.adapters is None:
            return self.out(h)
        return torch.cat([out(ad(h)) for ad, out in zip(self.adapters, self.outs)], dim=1)


# presets for --arch (all start from the published LegNet sizes; --width scales them)
PRESETS = {
    "pyramid": dict(multiscale_stem=True, pyramid=True, adapters=True),       # all three changes
    "pyr_stem": dict(multiscale_stem=True, pyramid=False, adapters=False),    # change 1 only
    "pyr_taps": dict(multiscale_stem=False, pyramid=True, adapters=False),    # change 2 only
    "pyr_taps_ms": dict(multiscale_stem=False, pyramid=True, adapters=False, pool_stats="meanstd"),  # change 2, mean+std pooling
    "pyr_adapt": dict(multiscale_stem=False, pyramid=False, adapters=True),   # change 3 only
    "pyr_off": dict(multiscale_stem=False, pyramid=False, adapters=False),    # none: identical to LegNet (for checks)
}


def build_pyramid(arch: str, n_out: int, width: float = 1.0, n_lenti: int = 3) -> PyramidLegNet:
    """Our trainers put the 3 lentiMPRA outputs first; with --new_cells none there are only those 3."""
    return PyramidLegNet(n_out, n_lenti=min(n_lenti, n_out), width=width, **PRESETS[arch])


def parameter_breakdown(model: PyramidLegNet) -> dict:
    """Trainable parameters per top-level part."""
    out = {}
    for name, child in model.named_children():
        out[name] = sum(p.numel() for p in child.parameters() if p.requires_grad)
    out["total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return out

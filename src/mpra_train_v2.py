"""Path B v2 screen: a shared-body student (all three cell types at once) with switchable design and recipe,
to close the gap to the published MPRA-LegNet model. Practice fold (9) only; fold 10 is never loaded for scoring.

  --arch ours     our dilated-convolution body (src/mpra_model.py)
  --arch legnet   the published MPRA-LegNet body (src/third_party/legnet_model.py, MIT), with three outputs
  --arch legnet_attn  the same, plus one self-attention layer after the convolutions
  --arch legnet_heads the same, but each cell type has its own small output network (v3 screen)
  --arch legnet_hires the same, but the first block keeps full letter-by-letter resolution (v3 screen)
  --arch legnet_rc    the same, but the model itself reads both strands and averages, also while training (v3 screen)
  --ema D         score and save a running average of the weights (decay D, e.g. 0.999) (v3 screen)
                  KNOWN FAULT: the averaged weights are scored with averaged BatchNorm statistics, which do not
                  fit them (0.52 on fold 9; 0.829 once the statistics are recomputed from training pieces).
                  Recompute the statistics before scoring if this option is ever used again.
  --shift K       the published augmentation: slide the piece 0..K letters, filling in with the real letters
                  of the lab construct next to it (not unknown letters, as our earlier --shift did)

The same leak guard as mpra_train_multi.py: pieces in fold 9 or 10 of ANY cell type are removed from training.

Run from the repository root, e.g. (the published recipe):
    .venv/bin/python src/mpra_train_v2.py --arch legnet --epochs 25 --batch 1024 --lr 0.01 --wd 0.1 --shift 21
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from data import onehot, reverse_complement_codes  # noqa: E402  (src/data.py)
from legnet_model import LegNet, initialize_weights  # noqa: E402
from mpra_data import CELLS, TRAIN_FOLDS, VALIDATION_FOLD, element_pearson, encode, load  # noqa: E402
from mpra_model import ActivityStudent, count_parameters  # noqa: E402
from mpra_train import RUNS  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402

RIGHT_FLANK = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"  # the construct after the piece (from the MPRA-LegNet code)


def build(arch: str, n_out: int, dropout: float = 0.0, channels: int = 128, width: float = 1.0) -> nn.Module:
    if arch == "legnet" and width != 1.0:  # a wider copy of the published body: every layer has `width` times the channels
        w = lambda c: int(round(c * width / 8)) * 8
        m = LegNet(in_ch=4, stem_ch=w(64), stem_ks=11, ef_ks=9, ef_block_sizes=[w(80), w(96), w(112), w(128)],
                   pool_sizes=[2, 2, 2, 2], resize_factor=4)
        m.head[-1] = nn.Linear(2 * w(128), n_out)
        m.apply(initialize_weights)
        return m
    if arch == "legnet_attn":
        return LegNetAttention(n_out)
    if arch == "legnet_heads":
        m = build("legnet", n_out)
        m.head = PerCellHeads(256, n_out)
        m.head.apply(initialize_weights)
        return m
    if arch == "legnet_hires":  # no pooling after the first block: the later blocks see twice as many positions
        m = LegNet(in_ch=4, stem_ch=64, stem_ks=11, ef_ks=9, ef_block_sizes=[80, 96, 112, 128], pool_sizes=[1, 2, 2, 2], resize_factor=4)
        m.head[-1] = nn.Linear(256, n_out)
        m.apply(initialize_weights)
        return m
    if arch == "legnet_rc":
        return RCSymmetric(build("legnet", n_out))
    if arch == "ours":
        return ActivityStudent(channels=channels, dropout=dropout, n_out=n_out)
    m = LegNet(in_ch=4, stem_ch=64, stem_ks=11, ef_ks=9, ef_block_sizes=[80, 96, 112, 128], pool_sizes=[2, 2, 2, 2], resize_factor=4)
    m.head[-1] = nn.Linear(256, n_out)
    m.apply(initialize_weights)
    return m


class PerCellHeads(nn.Module):
    """One small output network per cell type (the published head, repeated), on top of the shared body."""

    def __init__(self, dim: int, n_out: int):
        super().__init__()
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.SiLU(), nn.Linear(dim, 1))
                                   for _ in range(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([h(x) for h in self.heads], dim=1)


class RCSymmetric(nn.Module):
    """Reads the piece and its reverse complement with the same weights and averages the two answers, so the
    model gives exactly the same answer for both strands by construction (also during training)."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, 4 letters in A C G T order, length)
        return (self.base(x) + self.base(x.flip(1).flip(2))) / 2


class LegNetAttention(nn.Module):
    """The MPRA-LegNet body with ONE self-attention layer added after the convolutions. By then the piece is
    14 positions long (230 letters pooled 16 times), so each position can look at every other one."""

    def __init__(self, n_out: int):
        super().__init__()
        self.base = build("legnet", n_out)
        self.pos = nn.Parameter(torch.zeros(1, 16, 128))
        self.attn = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1,
                                               batch_first=True, norm_first=True)
        self.attn.apply(initialize_weights)  # small starting weights: the layer starts close to "change nothing"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.base
        h = b.main(b.stem(x)).transpose(1, 2)
        h = self.attn(h + self.pos[:, : h.shape[1]]).transpose(1, 2)
        return b.head(torch.nn.functional.adaptive_avg_pool1d(b.mapper(h), 1).squeeze(-1))


def flank_shift(c: torch.Tensor, max_shift: int, flank: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Each piece slides 0..max_shift letters; the letters that come into view are the construct's own."""
    s = torch.randint(0, max_shift + 1, (len(c), 1), generator=gen)
    ext = torch.cat([c, flank[:max_shift].expand(len(c), -1)], dim=1)
    return ext.gather(1, torch.arange(c.shape[1])[None, :] + s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["ours", "legnet", "legnet_attn", "legnet_heads", "legnet_hires", "legnet_rc"], required=True)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--shift", type=int, default=21)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ema", type=float, default=0.0)
    args = ap.parse_args()
    name = f"v2_{args.arch}_e{args.epochs}_b{args.batch}_lr{args.lr}_wd{args.wd}_shift{args.shift}" \
           + (f"_drop{args.dropout}" if args.dropout else "") + (f"_c{args.channels}" if args.channels != 128 else "") \
           + (f"_w{args.width}" if args.width != 1.0 else "") + (f"_ema{args.ema}" if args.ema else "") + (f"_{args.tag}" if args.tag else "") + f"_seed{args.seed}"
    run_dir = Path(RUNS) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    everything = {c: load(c, list(range(1, 11))) for c in CELLS}
    held_out = set()
    for d in everything.values():
        held_out |= set(d.seq[d.fold >= VALIDATION_FOLD])
    parts, removed = [], 0
    for k, c in enumerate(CELLS):
        d = everything[c]
        d = d[d.fold.isin(TRAIN_FOLDS)]
        keep = ~d.seq.isin(held_out)
        removed += int((~keep).sum())
        parts.append(d[keep].assign(cell=k))
    tr = pd.concat(parts, ignore_index=True)
    val = {c: everything[c][everything[c].fold == VALIDATION_FOLD].reset_index(drop=True) for c in CELLS}
    x_tr = torch.from_numpy(encode(tr.seq))
    k_tr = torch.tensor(tr.cell.to_numpy())
    x_val = {c: torch.from_numpy(encode(val[c].seq)) for c in CELLS}
    mean = torch.tensor([everything[c].mean_value[everything[c].fold.isin(TRAIN_FOLDS)].mean() for c in CELLS], dtype=torch.float32)
    std = torch.tensor([everything[c].mean_value[everything[c].fold.isin(TRAIN_FOLDS)].std() for c in CELLS], dtype=torch.float32)
    y_tr = (torch.tensor(tr.mean_value.to_numpy(), dtype=torch.float32) - mean[k_tr]) / std[k_tr]
    flank = torch.from_numpy(encode(pd.Series([RIGHT_FLANK]))[0])

    model = build(args.arch, len(CELLS), args.dropout, args.channels, args.width).to(device)
    ema = (torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema), use_buffers=True)
           if args.ema else None)
    scored = ema.module if ema else model  # the weights that are scored and saved
    steps_per_epoch = (len(x_tr) + args.batch - 1) // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr / 25, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.3, three_phase=False, cycle_momentum=False)
    config = {**vars(args), "parameters": count_parameters(model), "train_rows": len(x_tr), "removed_by_leak_guard": removed}
    print(f"{name}: {config['parameters']:,} parameters; train {len(x_tr):,} rows; device {device}", flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    log, best, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        order, losses = torch.randperm(len(x_tr), generator=gen), []
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            c = x_tr[idx]
            if args.shift:
                c = flank_shift(c, args.shift, flank, gen)
            c, y, k = c.to(device), y_tr[idx].to(device), k_tr[idx].to(device)
            if torch.rand(1, generator=gen).item() < 0.5:
                c = reverse_complement_codes(c)
            pred = model(onehot(c)).gather(1, k[:, None]).squeeze(1)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema:
                ema.update_parameters(model)
            losses.append(loss.item())
        r = {c: element_pearson(val[c], predict_multi(scored, x_val[c], device)[:, k]) for k, c in enumerate(CELLS)}
        mean_r = float(np.mean(list(r.values())))
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_element_pearson": r, "mean": mean_r,
                    "minutes": (time.time() - t0) / 60})
        print(f"  epoch {epoch:>2}: train loss {log[-1]['train_loss']:.4f}, validation " +
              ", ".join(f"{c} {v:.3f}" for c, v in r.items()) + f", mean {mean_r:.3f}, {log[-1]['minutes']:.1f} min", flush=True)
        if mean_r > best:
            best = mean_r
            torch.save(scored.state_dict(), run_dir / "best.pt")
        (run_dir / "log.json").write_text(json.dumps({"config": config, "log": log}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.0f} min: best mean validation Pearson {best:.3f}", flush=True)


if __name__ == "__main__":
    main()

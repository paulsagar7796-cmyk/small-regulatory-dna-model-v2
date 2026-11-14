"""Path B v2, run 5: ONE student that learns from the lab labels and from several already-trained students.

For every training batch the student gets two kinds of target:
  1. the lab label, for the cell type the piece was measured in (as before);
  2. the averaged prediction of the teachers for ALL three cell types, on the piece itself and on a randomly
     changed copy of it (a few letters swapped), for which no lab label exists.
Teachers and student only ever see training pieces (the same leak guard as mpra_train_v2.py), and the
result is a single model of the same size as run 1. Practice fold (9) only.

Run from the repository root, e.g.:
    .venv/bin/python src/mpra_distill.py --teachers legnet:runs/A/best.pt,legnet_attn:runs/B/best.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from data import onehot, reverse_complement_codes  # noqa: E402  (src/data.py)
from mpra_data import CELLS, TRAIN_FOLDS, VALIDATION_FOLD, element_pearson, encode, load  # noqa: E402
from mpra_model import count_parameters  # noqa: E402
from mpra_train import RUNS  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402
from mpra_train_v2 import RIGHT_FLANK, build, flank_shift  # noqa: E402


def mutate(c: torch.Tensor, rate: float, gen: torch.Generator) -> torch.Tensor:
    """Swap each letter for a different one with probability `rate`."""
    hit = torch.rand(c.shape, generator=gen) < rate
    return torch.where(hit, (c + torch.randint(1, 4, c.shape, generator=gen).to(c.dtype)) % 4, c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", required=True, help="comma-separated arch:path pairs")
    ap.add_argument("--arch", default="legnet")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--shift", type=int, default=21)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the teacher targets against the lab labels")
    ap.add_argument("--mut", type=float, default=0.05, help="share of letters swapped in the changed copies (0 = no copies)")
    ap.add_argument("--single", type=float, default=0.0,
                    help="v4: also add one single-letter-changed copy of each piece, with an extra loss (this weight) on the "
                         "EFFECT of the change (changed minus original), which is what a judge of single-letter changes needs")
    ap.add_argument("--max_steps", type=int, default=0, help="for a quick test only")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    teacher_specs = [t.split(":", 1) for t in args.teachers.split(",")]  # arch:path, or arch@width:path for a wider teacher
    name = f"distill_{args.arch}_t{len(teacher_specs)}_lam{args.lam}_mut{args.mut}_e{args.epochs}" \
           + (f"_single{args.single}" if args.single else "") \
           + (f"_{args.tag}" if args.tag else "") + f"_seed{args.seed}"
    run_dir = Path(RUNS) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    everything = {c: load(c, list(range(1, 11))) for c in CELLS}
    held_out = set()
    for d in everything.values():
        held_out |= set(d.seq[d.fold >= VALIDATION_FOLD])
    parts = []
    for k, c in enumerate(CELLS):
        d = everything[c]
        d = d[d.fold.isin(TRAIN_FOLDS)]
        parts.append(d[~d.seq.isin(held_out)].assign(cell=k))
    tr = pd.concat(parts, ignore_index=True)
    val = {c: everything[c][everything[c].fold == VALIDATION_FOLD].reset_index(drop=True) for c in CELLS}
    x_tr = torch.from_numpy(encode(tr.seq))
    k_tr = torch.tensor(tr.cell.to_numpy())
    x_val = {c: torch.from_numpy(encode(val[c].seq)) for c in CELLS}
    mean = torch.tensor([everything[c].mean_value[everything[c].fold.isin(TRAIN_FOLDS)].mean() for c in CELLS], dtype=torch.float32)
    std = torch.tensor([everything[c].mean_value[everything[c].fold.isin(TRAIN_FOLDS)].std() for c in CELLS], dtype=torch.float32)
    y_tr = (torch.tensor(tr.mean_value.to_numpy(), dtype=torch.float32) - mean[k_tr]) / std[k_tr]
    flank = torch.from_numpy(encode(pd.Series([RIGHT_FLANK]))[0])

    teachers = []
    for arch, path in teacher_specs:
        arch, _, width = arch.partition("@")
        t = build(arch, len(CELLS), width=float(width or 1.0)).to(device)
        t.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        teachers.append(t.eval())
    val_teacher = {c: np.mean([predict_multi(t, x_val[c], device)[:, k] for t in teachers], axis=0) for k, c in enumerate(CELLS)}
    teacher_r = {c: element_pearson(val[c], val_teacher[c]) for c in CELLS}
    for t in teachers:
        t.eval()  # predict_multi leaves a model in training mode; teachers must stay frozen
    print("teachers averaged, validation: " + ", ".join(f"{c} {v:.3f}" for c, v in teacher_r.items())
          + f", mean {np.mean(list(teacher_r.values())):.3f}", flush=True)

    @torch.no_grad()
    def teach(c: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = sum(t(onehot(c)) + t(onehot(reverse_complement_codes(c))) for t in teachers) / (2 * len(teachers))
        return out.float()

    model = build(args.arch, len(CELLS)).to(device)
    steps_per_epoch = (len(x_tr) + args.batch - 1) // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr / 25, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.3, three_phase=False, cycle_momentum=False)
    config = {**vars(args), "parameters": count_parameters(model), "train_rows": len(x_tr), "teachers_validation": teacher_r}
    print(f"{name}: {config['parameters']:,} parameters; train {len(x_tr):,} rows; {len(teachers)} teachers; device {device}", flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    log, best, t0, step = [], -1.0, time.time(), 0
    for epoch in range(1, args.epochs + 1):
        order, losses = torch.randperm(len(x_tr), generator=gen), []
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            c = x_tr[idx]
            if args.shift:
                c = flank_shift(c, args.shift, flank, gen)
            real = c
            if args.mut:
                c = torch.cat([c, mutate(real, args.mut, gen)])
            if args.single:  # one copy of each piece with exactly one letter changed
                one = real.clone()
                at = torch.randint(0, real.shape[1], (len(real),), generator=gen)
                rows = torch.arange(len(real))
                one[rows, at] = (one[rows, at] + torch.randint(1, 4, (len(real),), generator=gen).to(one.dtype)) % 4
                c = torch.cat([c, one])
            c, y, k = c.to(device), y_tr[idx].to(device), k_tr[idx].to(device)
            if torch.rand(1, generator=gen).item() < 0.5:
                c = reverse_complement_codes(c)
            target = teach(c)
            pred = model(onehot(c))
            label_loss = torch.nn.functional.mse_loss(pred[: len(idx)].gather(1, k[:, None]).squeeze(1), y)
            teacher_loss = torch.nn.functional.mse_loss(pred, target)
            loss = label_loss + args.lam * teacher_loss
            if args.single:  # the effect of the single-letter change, student against teachers, on a scale near 1
                d_t = target[-len(idx):] - target[: len(idx)]
                d_s = pred[-len(idx):] - pred[: len(idx)]
                loss = loss + args.single * torch.nn.functional.mse_loss(d_s, d_t) / (d_t.var() + 1e-6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append((label_loss.item(), teacher_loss.item()))
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        r = {c: element_pearson(val[c], predict_multi(model, x_val[c], device)[:, k]) for k, c in enumerate(CELLS)}
        mean_r = float(np.mean(list(r.values())))
        ll, tl = np.mean(losses, axis=0)
        log.append({"epoch": epoch, "label_loss": float(ll), "teacher_loss": float(tl), "validation_element_pearson": r,
                    "mean": mean_r, "minutes": (time.time() - t0) / 60})
        print(f"  epoch {epoch:>2}: label loss {ll:.4f}, teacher loss {tl:.4f}, validation " +
              ", ".join(f"{c} {v:.3f}" for c, v in r.items()) + f", mean {mean_r:.3f}, {log[-1]['minutes']:.1f} min", flush=True)
        if mean_r > best:
            best = mean_r
            torch.save(model.state_dict(), run_dir / "best.pt")
        (run_dir / "log.json").write_text(json.dumps({"config": config, "log": log}, indent=1))
        if args.max_steps and step >= args.max_steps:
            break
    print(f"done in {(time.time() - t0) / 60:.0f} min: best mean validation Pearson {best:.3f}", flush=True)


if __name__ == "__main__":
    main()

"""Path B v1 candidate: ONE shared body trained on all three cell types at once, with one output per cell type.

Why: the body sees about three times more DNA, which should help most where data is scarce (WTC11).
Leak guard: many pieces were measured in more than one cell type, and the dataset puts them in different
folds per cell type. Any piece that sits in fold 9 or 10 of ANY cell type is removed from training, so the
shared body never sees a validation or test piece, not even with another cell type's label.

Run from the repository root:
    .venv/bin/python src/mpra_train_multi.py [--seed 2] [--dropout 0.2] [--shift 10]
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
from data import onehot, reverse_complement_codes  # noqa: E402  (src/data.py)
from mpra_data import CELLS, TRAIN_FOLDS, VALIDATION_FOLD, element_pearson, encode, load  # noqa: E402
from mpra_model import ActivityStudent, count_parameters  # noqa: E402
from mpra_train import RUNS, SETTINGS, shift_codes  # noqa: E402


@torch.no_grad()
def predict_multi(model, codes: torch.Tensor, device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(codes), batch):
        c = codes[i : i + batch].to(device)
        out.append(((model(onehot(c)) + model(onehot(reverse_complement_codes(c)))) / 2).float().cpu())
    model.train()
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SETTINGS["seed"])
    ap.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--shift", type=int, default=0)
    args = ap.parse_args()
    name = (f"multi_c128_seed{args.seed}" + (f"_drop{args.dropout}" if args.dropout else "")
            + (f"_shift{args.shift}" if args.shift else ""))
    run_dir = Path(RUNS) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
    y_tr = torch.tensor(tr.mean_value.to_numpy(), dtype=torch.float32)
    k_tr = torch.tensor(tr.cell.to_numpy())
    x_val = {c: torch.from_numpy(encode(val[c].seq)) for c in CELLS}
    # each cell type's values are on a different scale: train on standardised values so no cell type dominates the loss
    mean = torch.tensor([everything[c].mean_value[everything[c].fold.isin(TRAIN_FOLDS)].mean() for c in CELLS], dtype=torch.float32)
    std = torch.tensor([everything[c].mean_value[everything[c].fold.isin(TRAIN_FOLDS)].std() for c in CELLS], dtype=torch.float32)
    y_tr = (y_tr - mean[k_tr]) / std[k_tr]

    model = ActivityStudent(dropout=args.dropout, n_out=len(CELLS)).to(device)
    steps_per_epoch = (len(x_tr) + SETTINGS["batch_size"] - 1) // SETTINGS["batch_size"]
    opt = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SETTINGS["learning_rate"], total_steps=args.epochs * steps_per_epoch)
    print(f"{name}: {count_parameters(model):,} parameters; train {len(x_tr):,} rows from {CELLS} "
          f"({removed:,} removed by the leak guard); device {device}", flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    log, best, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        order, losses = torch.randperm(len(x_tr), generator=gen), []
        for i in range(0, len(order), SETTINGS["batch_size"]):
            idx = order[i : i + SETTINGS["batch_size"]]
            c, y, k = x_tr[idx].to(device), y_tr[idx].to(device), k_tr[idx].to(device)
            if torch.rand(1, generator=gen).item() < 0.5:
                c = reverse_complement_codes(c)
            if args.shift:
                c = shift_codes(c, int(torch.randint(-args.shift, args.shift + 1, (1,), generator=gen)))
            pred = model(onehot(c)).gather(1, k[:, None]).squeeze(1)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        r = {c: element_pearson(val[c], predict_multi(model, x_val[c], device)[:, k]) for k, c in enumerate(CELLS)}
        mean_r = float(np.mean(list(r.values())))
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_element_pearson": r, "mean": mean_r,
                    "minutes": (time.time() - t0) / 60})
        print(f"  epoch {epoch:>2}: train loss {log[-1]['train_loss']:.4f}, validation " +
              ", ".join(f"{c} {v:.3f}" for c, v in r.items()) + f", mean {mean_r:.3f}, {log[-1]['minutes']:.1f} min", flush=True)
        if mean_r > best:
            best = mean_r
            torch.save(model.state_dict(), run_dir / "best.pt")
        (run_dir / "log.json").write_text(json.dumps({"settings": SETTINGS, "seed": args.seed, "dropout": args.dropout,
                                                      "shift": args.shift, "removed_by_leak_guard": removed, "log": log}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.0f} min: best mean validation Pearson {best:.3f}", flush=True)


if __name__ == "__main__":
    main()

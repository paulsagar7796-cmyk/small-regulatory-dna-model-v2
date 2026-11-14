"""Path B practice runs: train the small student to predict lab activity, validate on fold 9.

The locked test fold (10) is never loaded here.

Run from the repository root:
    .venv/bin/python src/mpra_train.py K562              # one cell type, default seed
    .venv/bin/python src/mpra_train.py K562 --seed 2
"""

import argparse
import json
import sys
import time

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from data import onehot, reverse_complement_codes  # noqa: E402  (src/data.py)
from mpra_data import TRAIN_FOLDS, VALIDATION_FOLD, element_pearson, encode, load  # noqa: E402
from mpra_model import ActivityStudent, count_parameters  # noqa: E402

SETTINGS = {"epochs": 12, "batch_size": 256, "learning_rate": 2e-3, "weight_decay": 0.01, "seed": 20261113}
RUNS = "runs"  # where training writes
RELEASED = "released_weights"  # model 1, the released student (a link to the first repository's weights/ folder)


def checkpoint(run: str) -> Path:
    """best.pt of a run: a locally trained one (runs/) if present, else the released one (weights/)."""
    for root in (RUNS, RELEASED):
        if (Path(root) / run / "best.pt").exists():
            return Path(root) / run / "best.pt"
    raise FileNotFoundError(f"No checkpoint for run '{run}' in {RUNS}/ or {RELEASED}/. Only the two released students are "
                            "shipped; earlier students and teachers must be retrained with the scripts in src/.")


def shift_codes(c: torch.Tensor, k: int) -> torch.Tensor:
    """Slide the piece k letters right (k > 0) or left, filling the gap with unknown letters (code 4)."""
    if k == 0:
        return c
    out = torch.full_like(c, 4)
    if k > 0:
        out[:, k:] = c[:, :-k]
    else:
        out[:, :k] = c[:, -k:]
    return out


@torch.no_grad()
def predict(model, codes: torch.Tensor, device, batch: int = 1024) -> np.ndarray:
    """Average of the prediction for the piece as given and for its reverse complement (the same molecule)."""
    model.eval()
    out = []
    for i in range(0, len(codes), batch):
        c = codes[i : i + batch].to(device)
        out.append(((model(onehot(c)) + model(onehot(reverse_complement_codes(c)))) / 2).float().cpu())
    model.train()
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--seed", type=int, default=SETTINGS["seed"])
    ap.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--shift", type=int, default=0, help="shift each training batch by up to this many letters")
    args = ap.parse_args()
    from pathlib import Path

    name = (f"{args.cell}_c{args.channels}_seed{args.seed}" + (f"_drop{args.dropout}" if args.dropout else "")
            + (f"_shift{args.shift}" if args.shift else ""))
    run_dir = Path(RUNS) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tr, va = load(args.cell, TRAIN_FOLDS), load(args.cell, [VALIDATION_FOLD])
    x_tr, y_tr = torch.from_numpy(encode(tr.seq)), torch.tensor(tr.mean_value.to_numpy(), dtype=torch.float32)
    x_va = torch.from_numpy(encode(va.seq))
    model = ActivityStudent(channels=args.channels, dropout=args.dropout).to(device)
    steps_per_epoch = (len(x_tr) + SETTINGS["batch_size"] - 1) // SETTINGS["batch_size"]
    opt = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SETTINGS["learning_rate"], total_steps=args.epochs * steps_per_epoch)
    print(f"{name}: {count_parameters(model):,} parameters; train {len(x_tr):,} rows (folds {TRAIN_FOLDS}), "
          f"validation {len(x_va):,} rows (fold {VALIDATION_FOLD}); device {device}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    log, best, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        order, losses = torch.randperm(len(x_tr), generator=gen), []
        for i in range(0, len(order), SETTINGS["batch_size"]):
            idx = order[i : i + SETTINGS["batch_size"]]
            c, y = x_tr[idx].to(device), y_tr[idx].to(device)
            if torch.rand(1, generator=gen).item() < 0.5:
                c = reverse_complement_codes(c)  # the same molecule read from the other strand: same label
            if args.shift:
                c = shift_codes(c, int(torch.randint(-args.shift, args.shift + 1, (1,), generator=gen)))
            loss = torch.nn.functional.mse_loss(model(onehot(c)), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        r = element_pearson(va, predict(model, x_va, device))
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_element_pearson": r,
                    "minutes": (time.time() - t0) / 60})
        print(f"  epoch {epoch:>2}: train loss {log[-1]['train_loss']:.4f}, validation Pearson {r:.3f}, "
              f"{log[-1]['minutes']:.1f} min", flush=True)
        if r > best:
            best = r
            torch.save(model.state_dict(), run_dir / "best.pt")
        (run_dir / "log.json").write_text(json.dumps({"settings": SETTINGS, "cell": args.cell, "seed": args.seed,
                                                      "channels": args.channels, "dropout": args.dropout, "shift": args.shift,
                                                      "epochs": args.epochs, "log": log}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.0f} min: best validation Pearson {best:.3f}", flush=True)


if __name__ == "__main__":
    main()

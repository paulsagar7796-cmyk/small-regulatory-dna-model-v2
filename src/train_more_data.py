"""Model 2 (more data): the shared-body student of model 1, trained on the lentiMPRA tables AND on the
natural-variant MPRA table of Siraj and others (data/more_data/train.parquet, made by src/prepare_more_data.py).

One network, eight outputs:
    0-2  lentiMPRA activity in HepG2, K562, WTC11           (the outputs of model 1)
    3-7  plasmid-MPRA activity in HepG2, K562, SK-N-SH, HCT116, A549  (new)
Every new piece is 200 letters and is placed between the same two adapters as the lentiMPRA pieces, so all
rows have the same 230-letter format. A row adds a loss only at the outputs it has a measurement for; each
output's loss is the mean over its measured rows, and the outputs' losses are averaged.

Everything else is the recipe of model 1's teachers (src/mpra_train_v2.py --arch legnet): 25 epochs, batch 1024,
AdamW, one-cycle learning rate with peak 0.01, weight decay 0.1, slide of up to 21 letters, random strand.
The checkpoint with the best mean Pearson on lentiMPRA fold 9 is kept (the rule of model 1), so the new
practice chromosomes are not used to pick the checkpoint.

  --new_cells LIST   which new outputs to train (comma-separated table names; "none" = the control run:
                     lentiMPRA only, with the same pieces removed by the leak guards)
  --teachers PATHS   model 1's second step, applied to model 2: comma-separated best.pt files of trained
                     copies of this same model. The student then also learns their averaged prediction for
                     ALL outputs, on every piece and on a copy of it with --mut of its letters swapped
                     (for which no lab label exists). Teachers saw the same training rows as the student.
Practice data only. The locked chromosomes (5, 14, 22) and lentiMPRA fold 10 are never loaded.

Run from the repository root:
    .venv/bin/python src/train_more_data.py
    .venv/bin/python src/train_more_data.py --new_cells none
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
from data import onehot, reverse_complement_codes  # noqa: E402
from mpra_data import CELLS, TRAIN_FOLDS, VALIDATION_FOLD, element_pearson, encode, load  # noqa: E402
from mpra_model import count_parameters  # noqa: E402
from mpra_train import RUNS  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402
from mpra_distill import mutate  # noqa: E402
from mpra_train_v2 import RIGHT_FLANK, build, flank_shift  # noqa: E402

MORE = Path("data/more_data")
NEW_CELLS = ["HEPG2", "K562", "SKNSH", "HCT116", "A549"]
LEFT, RIGHT = "AGGACCGGATCAACT", "CATTGCGTGAACCGA"
MIN_ACTIVITY = 1.0  # practice read-out: variants where at least one version is active (log2FC >= 1)


def between_adapters(seqs: pd.Series) -> np.ndarray:
    c = encode(seqs)
    l, r = (np.tile(encode(pd.Series([a]))[0], (len(c), 1)) for a in (LEFT, RIGHT))
    return np.concatenate([l, c, r], axis=1)


def both_orientations(model, seqs: pd.Series, device) -> np.ndarray:
    """(n, outputs): the piece between the adapters, and inserted the other way round, averaged (model 1's recipe)."""
    c = encode(seqs)
    flipped = pd.Series(["".join("ACGT"[3 - x] for x in row[::-1]) for row in c])
    return (predict_multi(model, torch.from_numpy(between_adapters(seqs)), device)
            + predict_multi(model, torch.from_numpy(between_adapters(flipped)), device)) / 2


def practice_readout(pred1: np.ndarray, pred2: np.ndarray, pr: pd.DataFrame, columns: dict) -> dict:
    """columns: name -> output index. Activity Pearson over both versions, and effect Pearson over active variants."""
    out = {}
    for name, (cell, k) in columns.items():
        a1, a2, skew = (pr[f"{x}_{cell}"].to_numpy() for x in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
        ok = ~np.isnan(a1) & ~np.isnan(a2)
        act = np.corrcoef(np.concatenate([a1[ok], a2[ok]]), np.concatenate([pred1[ok, k], pred2[ok, k]]))[0, 1]
        e = ok & ~np.isnan(skew) & (np.maximum(a1, a2) >= MIN_ACTIVITY)
        out[name] = {"activity": float(act), "effect": float(np.corrcoef(skew[e], (pred2 - pred1)[e, k])[0, 1]), "effect_variants": int(e.sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_cells", default=",".join(NEW_CELLS))
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--shift", type=int, default=21)
    ap.add_argument("--teachers", default="", help="comma-separated best.pt files of trained copies (self-distillation)")
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the teacher targets against the lab labels")
    ap.add_argument("--mut", type=float, default=0.05, help="share of letters swapped in the changed copies (0 = no copies)")
    ap.add_argument("--max_steps", type=int, default=0, help="for a quick test only")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    new_cells = [] if args.new_cells == "none" else args.new_cells.split(",")
    n_out = len(CELLS) + len(new_cells)
    teacher_paths = [t for t in args.teachers.split(",") if t]
    name = f"more_{'-'.join(new_cells) if new_cells else 'control'}" \
           + (f"_distill_t{len(teacher_paths)}_lam{args.lam}_mut{args.mut}" if teacher_paths else "") + f"_e{args.epochs}" + (f"_{args.tag}" if args.tag else "") + f"_seed{args.seed}"
    run_dir = Path(RUNS) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # ---- lentiMPRA rows (model 1's training data, minus the pieces listed by the leak guards) ----
    everything = {c: load(c, list(range(1, VALIDATION_FOLD + 1))) for c in CELLS}  # folds 1 to 9 only: fold 10 is not loaded
    held_out = set()
    for c in CELLS:
        d = pd.read_csv(Path("data/mpra") / f"{c}.tsv", sep="\t", usecols=["seq", "fold", "rev"])
        held_out |= set(d.seq[(d.rev == 0) & (d.fold >= VALIDATION_FOLD)])  # sequences only
    excluded = set(pd.read_parquet(MORE / "lentimpra_excluded.parquet").seq)
    parts = []
    for k, c in enumerate(CELLS):
        d = everything[c]
        d = d[d.fold.isin(TRAIN_FOLDS)]
        parts.append(d[~d.seq.isin(held_out) & ~d.seq.isin(excluded)].assign(cell=k))
    tr = pd.concat(parts, ignore_index=True)
    x = [encode(tr.seq)]
    y = np.full((len(tr), n_out), np.nan, dtype=np.float32)
    y[np.arange(len(tr)), tr.cell.to_numpy()] = tr.mean_value.to_numpy()
    labels = [y]
    lenti_rows = len(tr)

    # ---- new rows: both versions of every training variant ----
    new_rows = 0
    if new_cells:
        nt = pd.read_parquet(MORE / "train.parquet")
        for a, col in (("allele1", "seq1"), ("allele2", "seq2")):
            x.append(between_adapters(nt[col]))
            y = np.full((len(nt), n_out), np.nan, dtype=np.float32)
            for j, cell in enumerate(new_cells):
                y[:, len(CELLS) + j] = nt[f"{a}.log2FC_{cell}"].to_numpy()
            labels.append(y)
        new_rows = 2 * len(nt)
    x_tr = torch.from_numpy(np.concatenate(x))
    y_raw = np.concatenate(labels)
    mean, std = np.nanmean(y_raw, axis=0), np.nanstd(y_raw, axis=0)
    y_tr = torch.from_numpy((y_raw - mean) / std)

    val = {c: everything[c][everything[c].fold == VALIDATION_FOLD].reset_index(drop=True) for c in CELLS}
    x_val = {c: torch.from_numpy(encode(val[c].seq)) for c in CELLS}
    pr = pd.read_parquet(MORE / "practice.parquet")
    readout = {"lenti_HepG2": ("HEPG2", 0), "lenti_K562": ("K562", 1)}
    readout.update({f"new_{cell}": (cell, len(CELLS) + j) for j, cell in enumerate(new_cells) if cell in ("HEPG2", "K562", "SKNSH")})
    flank = torch.from_numpy(encode(pd.Series([RIGHT_FLANK]))[0])

    teachers = []
    for path in teacher_paths:
        t_cfg = json.loads((Path(path).parent / "log.json").read_text())["config"]
        assert t_cfg["outputs"] == CELLS + [f"new_{c}" for c in new_cells], f"{path}: different outputs"
        assert np.allclose(t_cfg["label_mean"], mean, atol=1e-4) and np.allclose(t_cfg["label_std"], std, atol=1e-4), \
            f"{path}: the teacher's label scaling differs, so it was not trained on the same rows"
        t = build("legnet", n_out).to(device)
        t.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        teachers.append(t.eval())
    if teachers:
        tv = {c: element_pearson(val[c], np.mean([predict_multi(t, x_val[c], device)[:, k] for t in teachers], axis=0)) for k, c in enumerate(CELLS)}
        for t in teachers:
            t.eval()  # predict_multi leaves a model in training mode; teachers must stay frozen
        print("teachers averaged, fold 9: " + ", ".join(f"{c} {v:.3f}" for c, v in tv.items()) + f", mean {np.mean(list(tv.values())):.3f}", flush=True)

    @torch.no_grad()
    def teach(c: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = sum(t(onehot(c)) + t(onehot(reverse_complement_codes(c))) for t in teachers) / (2 * len(teachers))
        return out.float()

    model = build("legnet", n_out).to(device)
    steps_per_epoch = (len(x_tr) + args.batch - 1) // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr / 25, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.3, three_phase=False, cycle_momentum=False)
    config = {**vars(args), "outputs": CELLS + [f"new_{c}" for c in new_cells], "parameters": count_parameters(model),
              "lentimpra_rows": lenti_rows, "new_rows": new_rows, "lentimpra_rows_removed_by_new_guard": int(sum(
                  (everything[c].fold.isin(TRAIN_FOLDS) & everything[c].seq.isin(excluded)).sum() for c in CELLS)),
              "label_mean": mean.tolist(), "label_std": std.tolist()}
    print(f"{name}: {config['parameters']:,} parameters; {lenti_rows:,} lentiMPRA rows + {new_rows:,} new rows; "
          f"{n_out} outputs; device {device}", flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    log, best, t0, step = [], -1.0, time.time(), 0
    for epoch in range(1, args.epochs + 1):
        order, losses = torch.randperm(len(x_tr), generator=gen), []
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            c = x_tr[idx]
            if args.shift:
                c = flank_shift(c, args.shift, flank, gen)
            if teachers and args.mut:
                c = torch.cat([c, mutate(c, args.mut, gen)])
            c, target = c.to(device), y_tr[idx].to(device)
            if torch.rand(1, generator=gen).item() < 0.5:
                c = reverse_complement_codes(c)
            pred = model(onehot(c))
            seen = ~torch.isnan(target)
            err = torch.where(seen, (pred[: len(idx)] - torch.nan_to_num(target)) ** 2, torch.zeros_like(target))
            n = seen.sum(0)
            loss = (err.sum(0)[n > 0] / n[n > 0]).mean()  # mean per output over its measured rows, then over outputs
            if teachers:  # the teachers' averaged prediction, all outputs, pieces and changed copies
                loss = loss + args.lam * torch.nn.functional.mse_loss(pred, teach(c))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        r = {c: element_pearson(val[c], predict_multi(model, x_val[c], device)[:, k]) for k, c in enumerate(CELLS)}
        mean_r = float(np.mean(list(r.values())))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_element_pearson": r, "mean": mean_r}
        if epoch % 5 == 0 or epoch == args.epochs or args.max_steps:  # the new practice chromosomes: logged, not used for choices here
            entry["practice_variants"] = practice_readout(both_orientations(model, pr.seq1, device), both_orientations(model, pr.seq2, device), pr, readout)
        entry["minutes"] = (time.time() - t0) / 60
        log.append(entry)
        print(f"  epoch {epoch:>2}: loss {entry['train_loss']:.4f}, fold 9 " + ", ".join(f"{c} {v:.3f}" for c, v in r.items())
              + f", mean {mean_r:.3f}" + ("; practice variants " + ", ".join(f"{k} act {v['activity']:.3f} eff {v['effect']:.3f}"
                                          for k, v in entry["practice_variants"].items()) if "practice_variants" in entry else "")
              + f", {entry['minutes']:.1f} min", flush=True)
        if mean_r > best:
            best = mean_r
            torch.save(model.state_dict(), run_dir / "best.pt")
        (run_dir / "log.json").write_text(json.dumps({"config": config, "log": log}, indent=1))
        if args.max_steps and step >= args.max_steps:
            break
    print(f"done in {(time.time() - t0) / 60:.0f} min: best fold-9 mean Pearson {best:.3f}", flush=True)


if __name__ == "__main__":
    main()

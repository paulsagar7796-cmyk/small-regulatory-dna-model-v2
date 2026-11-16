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
  --pair_loss W      optional paired-effect loss (0 = off, the script then behaves exactly as before). Every step also
                     draws --pair_batch variants at random (own random stream), builds both 200-letter versions of
                     each with the SAME slide and the SAME strand, and adds W times the mean over new outputs of
                     mean(((pred2 - pred1) - (y2 - y1))^2) / var(y2 - y1) to the loss, over pairs with both labels.
  --extra_table PATH extra single training pieces (data/more_data/train_gosai_extra.parquet, made by src/prepare_gosai_extra.py:
                     pieces of Table S2 of Gosai and others that the natural-variant table does not have). They are added as
                     ordinary single rows (not pairs) after the variant rows, with labels only at the outputs the table has
                     (HEPG2, K562, SKNSH; HCT116 and A549 are masked, so these rows add no loss and no gradient there).
                     An epoch still covers every row once, so with the extra rows it has more steps (and the one-cycle
                     schedule is stretched to match). The label scaling (mean and standard deviation per output) is still
                     computed from the rows WITHOUT the extra table, so the output units equal those of earlier runs and
                     earlier teachers still pass the scaling check. The run name gets the suffix _gosai. Without the option
                     nothing changes (same name, same batches).
  --teachers PATHS   model 1's second step, applied to model 2: comma-separated best.pt files of trained
                     copies of this same model. The student then also learns their averaged prediction for
                     ALL outputs, on every piece and on a copy of it with --mut of its letters swapped
                     (for which no lab label exists). Teachers saw the same training rows as the student.
  --evoaug N         EvoAug-style augmentation (Lee, Tang, Toneyan, Koo, Genome Biology 2023, doi 10.1186/s13059-023-02941-w;
                     code github.com/p-koo/evoaug). 0 = off (the script is then bit-for-bit as before). Every training row gets,
                     independently, a random number of augmentations, uniform 0..N (--evoaug_hard: exactly N), drawn without
                     replacement from five: inversion (a segment replaced by its reverse complement), deletion (a segment removed,
                     the row padded back to length with random letters, half at each end), translocation (the insert rolled by up
                     to +-max letters), insertion (random letters put in at a random point, the 3' end trimmed to length) and
                     point mutation (about --evoaug_mut of the letters swapped). They act on the 200-letter insert only, never
                     on the two adapters, are applied in EvoAug's order (inversion, deletion, translocation, insertion,
                     mutation) and come BEFORE the usual slide and random strand. Labels stay unchanged. Size limits
                     (--evoaug_inv/_del/_trans/_ins, --evoaug_mut) default to the DeepSTARR values of the paper scaled from 249 to
                     200 letters. Own random stream (seed + 104729). Refused together with --pair_loss or --teachers.
  --finetune_epochs M  after the main schedule, M more epochs with the EvoAug augmentations off (slide and strand stay on) at the
                     constant learning rate --finetune_lr (EvoAug: 5 epochs, 1e-4). The same optimiser is kept; the best checkpoint
                     (best mean Pearson on lentiMPRA fold 9) is chosen across both stages; each log entry has a "stage".
                     --max_steps (test only) caps each stage separately.
  --arch NAME        network design: legnet (default; models 1 and 2, unchanged) or malinois (the design of the competitor MPAC: one
                     randomly initialised Malinois network, 4.1M parameters with 3 outputs, with our number of outputs instead; settings of
                     data/models/MPAC/config.json; src/malinois_body.py). Our 230-letter rows (after the slide and the strand flip, which act
                     as before) are cut to the middle 200 letters and padded to 600 with the vector flanks MPAC's own predict() uses. The run
                     name gets the suffix _malinois. Optimiser, schedule and --lr/--wd are unchanged (--width is refused with malinois).
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
from malinois_body import build_arch  # noqa: E402

MORE = Path("data/more_data")
NEW_CELLS = ["HEPG2", "K562", "SKNSH", "HCT116", "A549"]
LEFT, RIGHT = "AGGACCGGATCAACT", "CATTGCGTGAACCGA"
MIN_ACTIVITY = 1.0  # practice read-out: variants where at least one version is active (log2FC >= 1)


def between_adapters(seqs: pd.Series) -> np.ndarray:
    c = encode(seqs)
    l, r = (np.tile(encode(pd.Series([a]))[0], (len(c), 1)) for a in (LEFT, RIGHT))
    return np.concatenate([l, c, r], axis=1)


def shift_by(c: torch.Tensor, s: torch.Tensor, flank: torch.Tensor) -> torch.Tensor:
    """flank_shift with the slide given: row i slides s[i] letters (s has shape (n, 1)). Used so that the two versions
    of a variant slide by the same amount."""
    ext = torch.cat([c, flank[: int(s.max())].expand(len(c), -1)], dim=1)
    return ext.gather(1, torch.arange(c.shape[1])[None, :] + s)


def draw_pairs(x_tr: torch.Tensor, first: int, n_var: int, n: int, shift: int, flank: torch.Tensor, pgen: torch.Generator):
    """n random variants: (versions 1 then versions 2 as one (2n, length) tensor, row ids of version 1, row ids of version 2).
    Both versions of a variant slide by the same random amount."""
    pv = torch.randint(0, n_var, (n,), generator=pgen)
    p1, p2 = first + pv, first + n_var + pv
    pc = x_tr[torch.cat([p1, p2])]
    if shift:
        pc = shift_by(pc, torch.randint(0, shift + 1, (n, 1), generator=pgen).repeat(2, 1), flank)
    return pc, p1, p2


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


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per output over its measured rows (NaN target = not measured), then the mean over outputs that have at least one
    measured row. An output with no measured row in the batch gets no loss and no gradient. pred and target: (rows, outputs)."""
    seen = ~torch.isnan(target)
    err = torch.where(seen, (pred - torch.nan_to_num(target)) ** 2, torch.zeros_like(target))
    n = seen.sum(0)
    return (err.sum(0)[n > 0] / n[n > 0]).mean()


EVOAUG_NAMES = ["inversion", "deletion", "translocation", "insertion", "mutation"]  # also EvoAug's order of application
EVOAUG_SALT = 104729


def _rand_letters(shape, gen: torch.Generator, dtype) -> torch.Tensor:
    return torch.randint(0, 4, shape, generator=gen).to(dtype)


def evo_inversion(x: torch.Tensor, vmax: int, gen: torch.Generator) -> torch.Tensor:
    """A random segment (length 0..vmax, start uniform over the places it fits) is replaced by its reverse complement."""
    n, length = x.shape
    v = torch.randint(0, vmax + 1, (n, 1), generator=gen)
    s = (torch.rand(n, 1, generator=gen) * (length - v + 1)).long()
    j = torch.arange(length)[None, :]
    inside = (j >= s) & (j < s + v)
    src = torch.where(inside, 2 * s + v - 1 - j, j)
    y = x.gather(1, src)
    return torch.where(inside & (y < 4), 3 - y, y)  # complement; N (code 4) stays N


def evo_deletion(x: torch.Tensor, dmax: int, gen: torch.Generator) -> torch.Tensor:
    """A random segment (length 0..dmax) is cut out; the row is padded back to length with random letters, d // 2 in front
    and the rest behind (as EvoAug does)."""
    n, length = x.shape
    d = torch.randint(0, dmax + 1, (n, 1), generator=gen)
    s = (torch.rand(n, 1, generator=gen) * (length - d + 1)).long()
    pad_l = d // 2
    j = torch.arange(length)[None, :]
    k = j - pad_l  # index into the shortened sequence
    kept = (k >= 0) & (k < length - d)
    src = (torch.where(k < s, k, k + d)).clamp(0, length - 1)
    return torch.where(kept, x.gather(1, src), _rand_letters(x.shape, gen, x.dtype))


def evo_translocation(x: torch.Tensor, tmax: int, gen: torch.Generator) -> torch.Tensor:
    """The row is rolled (cut at one point, the two parts swapped) by 0..tmax letters, either direction."""
    n, length = x.shape
    sh = torch.randint(0, tmax + 1, (n, 1), generator=gen) * (2 * torch.randint(0, 2, (n, 1), generator=gen) - 1)
    return x.gather(1, (torch.arange(length)[None, :] - sh) % length)


def evo_insertion(x: torch.Tensor, imax: int, gen: torch.Generator) -> torch.Tensor:
    """0..imax random letters are put in at a random point (uniform over 0..length-1); the 3' end is cut off to keep the length."""
    n, length = x.shape
    m = torch.randint(0, imax + 1, (n, 1), generator=gen)
    p = (torch.rand(n, 1, generator=gen) * length).long()
    j = torch.arange(length)[None, :]
    new = (j >= p) & (j < p + m)
    src = torch.where(j < p, j, j - m).clamp(0, length - 1)
    return torch.where(new, _rand_letters(x.shape, gen, x.dtype), x.gather(1, src))


def evo_mutation(x: torch.Tensor, frac: float, gen: torch.Generator) -> torch.Tensor:
    """round(frac / 0.75 * length) distinct positions get a letter drawn uniformly from ACGT (a quarter of them equal the old
    letter, so about frac of the letters change; EvoAug's rule)."""
    n, length = x.shape
    k = int(round(frac / 0.75 * length))
    if k == 0:
        return x
    pos = torch.rand(n, length, generator=gen).topk(k, dim=1).indices
    return x.scatter(1, pos, _rand_letters((n, k), gen, x.dtype))


def evoaug_insert(ins: torch.Tensor, n_max: int, hard: bool, sizes: dict, gen: torch.Generator) -> torch.Tensor:
    """EvoAug on a batch of inserts (n, 200 or so, uint8 codes). Row i gets a count uniform in 0..n_max (hard: n_max), and that many of the five
    augmentations, drawn without replacement; each augmentation then runs only on the rows that drew it."""
    n = len(ins)
    count = torch.full((n, 1), n_max) if hard else torch.randint(0, n_max + 1, (n, 1), generator=gen)
    rank = torch.rand(n, len(EVOAUG_NAMES), generator=gen).argsort(1).argsort(1)  # a random permutation of 0..4 per row
    chosen = rank < count
    fns = {"inversion": lambda x: evo_inversion(x, sizes["inversion"], gen), "deletion": lambda x: evo_deletion(x, sizes["deletion"], gen),
           "translocation": lambda x: evo_translocation(x, sizes["translocation"], gen),
           "insertion": lambda x: evo_insertion(x, sizes["insertion"], gen), "mutation": lambda x: evo_mutation(x, sizes["mutation"], gen)}
    ins = ins.clone()
    for a, name in enumerate(EVOAUG_NAMES):
        rows = chosen[:, a].nonzero().squeeze(1)
        if len(rows):
            ins[rows] = fns[name](ins[rows])
    return ins


def evoaug_batch(c: torch.Tensor, n_max: int, hard: bool, sizes: dict, gen: torch.Generator, adapter: int = 15) -> torch.Tensor:
    """c: (n, 230) rows, adapter + 200-letter insert + adapter. Only the insert is augmented."""
    out = c.clone()
    out[:, adapter:-adapter] = evoaug_insert(c[:, adapter:-adapter], n_max, hard, sizes, gen)
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
    ap.add_argument("--pair_loss", type=float, default=0.0, help="weight of the paired-effect loss (0 = off)")
    ap.add_argument("--pair_batch", type=int, default=256, help="variants drawn per step for the paired-effect loss")
    ap.add_argument("--extra_table", default="", help="parquet of extra single pieces (see the docstring); empty = off")
    ap.add_argument("--width", type=float, default=1.0, help="channel multiplier of the body (1.0 = model 2's size)")
    ap.add_argument("--arch", choices=["legnet", "malinois"], default="legnet", help="network design (see the docstring)")
    ap.add_argument("--evoaug", type=int, default=0, help="max augmentations per row (0 = off; see the docstring)")
    ap.add_argument("--evoaug_hard", action="store_true", help="exactly --evoaug augmentations per row instead of 0..N")
    ap.add_argument("--evoaug_inv", type=int, default=24, help="largest inversion (letters); 30 of 249 in the paper, scaled to 200")
    ap.add_argument("--evoaug_del", type=int, default=24, help="largest deletion; 30 of 249 in the paper, scaled to 200")
    ap.add_argument("--evoaug_trans", type=int, default=16, help="largest roll; 20 of 249 in the paper, scaled to 200")
    ap.add_argument("--evoaug_ins", type=int, default=16, help="largest insertion; 20 of 249 in the paper, scaled to 200")
    ap.add_argument("--evoaug_mut", type=float, default=0.05, help="share of letters changed by the point mutations (paper, DeepSTARR: 0.05)")
    ap.add_argument("--finetune_epochs", type=int, default=0, help="extra epochs without EvoAug after the main schedule")
    ap.add_argument("--finetune_lr", type=float, default=1e-4, help="constant learning rate of the fine-tuning epochs (EvoAug: 1e-4)")
    ap.add_argument("--max_steps", type=int, default=0, help="for a quick test only (caps each stage)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    new_cells = [] if args.new_cells == "none" else args.new_cells.split(",")
    n_out = len(CELLS) + len(new_cells)
    teacher_paths = [t for t in args.teachers.split(",") if t]
    if args.pair_loss > 0:
        assert new_cells, "--pair_loss needs new outputs (the variant table)"
    if args.extra_table:
        assert new_cells, "--extra_table needs new outputs (the variant table)"
    if args.evoaug and (args.pair_loss > 0 or teacher_paths):
        ap.error("--evoaug cannot be combined with --pair_loss or --teachers (not implemented)")
    if args.arch == "malinois" and args.width != 1.0:
        ap.error("--width only applies to --arch legnet")
    if args.evoaug < 0 or args.evoaug > len(EVOAUG_NAMES) or args.finetune_epochs < 0:
        ap.error(f"--evoaug must be 0..{len(EVOAUG_NAMES)} and --finetune_epochs >= 0")
    name = f"more_{'-'.join(new_cells) if new_cells else 'control'}" \
           + (f"_distill_t{len(teacher_paths)}_lam{args.lam}_mut{args.mut}" if teacher_paths else "") \
           + (f"_pair{args.pair_loss}" if args.pair_loss > 0 else "") + f"_e{args.epochs}" + (f"_{args.tag}" if args.tag else "") + f"_seed{args.seed}" + ("_gosai" if args.extra_table else "") + (f"_w{args.width}" if args.width != 1.0 else "") \
           + (f"_evoaug{args.evoaug}" + ("hard" if args.evoaug_hard else "") if args.evoaug else "") + (f"_ft{args.finetune_epochs}" if args.finetune_epochs else "") + ("_malinois" if args.arch == "malinois" else "")
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
    n_var = new_rows // 2  # version 1 of variant j is row lenti_rows + j, version 2 is row lenti_rows + n_var + j
    y_raw = np.concatenate(labels)
    mean, std = np.nanmean(y_raw, axis=0), np.nanstd(y_raw, axis=0)  # from the lentiMPRA and variant rows only, also when --extra_table is given
    extra_rows = 0
    if args.extra_table:  # single pieces after the variant rows (the paired-loss row ids above stay valid); HCT116 and A549 stay NaN
        ex = pd.read_parquet(args.extra_table)
        x.append(between_adapters(ex.seq))
        y_ex = np.full((len(ex), n_out), np.nan, dtype=np.float32)
        for j, cell in enumerate(new_cells):
            if f"log2FC_{cell}" in ex.columns:
                y_ex[:, len(CELLS) + j] = ex[f"log2FC_{cell}"].to_numpy()
        y_raw = np.concatenate([y_raw, y_ex])
        extra_rows = len(ex)
    x_tr = torch.from_numpy(np.concatenate(x))
    y_tr = torch.from_numpy((y_raw - mean) / std)
    pair_var = None
    if args.pair_loss > 0:  # variance of (version 2 minus version 1) per new output, over the whole training table, normalised units
        d = (y_tr[lenti_rows + n_var : lenti_rows + 2 * n_var, len(CELLS):] - y_tr[lenti_rows : lenti_rows + n_var, len(CELLS):]).numpy()
        pair_var = torch.from_numpy(np.nanvar(d, axis=0)).float()
        print("variance of the paired difference per new output (normalised units): " + ", ".join(f"{v:.3f}" for v in pair_var), flush=True)
        pgen = torch.Generator().manual_seed(args.seed + 7919)  # its own stream: the main batches are unchanged

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

    model = build_arch(args.arch, n_out, width=args.width).to(device)
    steps_per_epoch = (len(x_tr) + args.batch - 1) // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr / 25, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.3, three_phase=False, cycle_momentum=False)
    config = {**vars(args), "outputs": CELLS + [f"new_{c}" for c in new_cells], "parameters": count_parameters(model),
              "lentimpra_rows": lenti_rows, "new_rows": new_rows, "extra_rows": extra_rows, "lentimpra_rows_removed_by_new_guard": int(sum(
                  (everything[c].fold.isin(TRAIN_FOLDS) & everything[c].seq.isin(excluded)).sum() for c in CELLS)),
              "label_mean": mean.tolist(), "label_std": std.tolist()}
    print(f"{name}: {config['parameters']:,} parameters; {lenti_rows:,} lentiMPRA rows + {new_rows:,} new rows"
          + (f" + {extra_rows:,} extra rows" if args.extra_table else "") + f"; {steps_per_epoch:,} steps per epoch; "
          f"{n_out} outputs; device {device}", flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    egen = torch.Generator().manual_seed(args.seed + EVOAUG_SALT)  # its own stream: the main batches are unchanged
    sizes = {"inversion": args.evoaug_inv, "deletion": args.evoaug_del, "translocation": args.evoaug_trans,
             "insertion": args.evoaug_ins, "mutation": args.evoaug_mut}
    total_epochs = args.epochs + args.finetune_epochs
    log, best, t0, step, epoch = [], -1.0, time.time(), 0, 0
    for stage, stage_epochs in (("main", args.epochs), ("finetune", args.finetune_epochs)):
        step = 0  # --max_steps counts per stage
        if stage == "finetune" and stage_epochs:
            for g in opt.param_groups:
                g["lr"] = args.finetune_lr  # constant; the one-cycle schedule has ended and is not stepped any more
        for _ in range(stage_epochs):
            epoch += 1
            order, losses, pair_losses = torch.randperm(len(x_tr), generator=gen), [], []
            for i in range(0, len(order), args.batch):
                idx = order[i : i + args.batch]
                c = x_tr[idx]
                if args.evoaug and stage == "main":
                    c = evoaug_batch(c, args.evoaug, args.evoaug_hard, sizes, egen)
                if args.shift:
                    c = flank_shift(c, args.shift, flank, gen)
                if teachers and args.mut:
                    c = torch.cat([c, mutate(c, args.mut, gen)])
                target = y_tr[idx]
                if pair_var is not None:  # both versions of --pair_batch random variants, same slide, appended to the batch
                    pc, p1, p2 = draw_pairs(x_tr, lenti_rows, n_var, args.pair_batch, args.shift, flank, pgen)
                    n_main = len(c)
                    c = torch.cat([c, pc])
                    target = torch.cat([target, y_tr[torch.cat([p1, p2])]])
                c, target = c.to(device), target.to(device)
                if torch.rand(1, generator=gen).item() < 0.5:  # one strand for the whole batch, so also for both versions of a pair
                    c = reverse_complement_codes(c)
                pred = model(onehot(c))
                # label loss rows: the usual batch, then (if on) the paired rows; mutated copies (if any) sit in between and have no labels
                lab = torch.cat([torch.arange(len(idx)), n_main + torch.arange(2 * args.pair_batch)]).to(device) if pair_var is not None \
                    else torch.arange(len(idx), device=device)
                loss = masked_mse(pred[lab], target)  # mean per output over its measured rows, then over outputs
                if pair_var is not None:
                    ppred = pred[n_main:, len(CELLS):]
                    dp = ppred[args.pair_batch:] - ppred[: args.pair_batch]
                    yp = target[len(idx):, len(CELLS):]
                    dy = yp[args.pair_batch:] - yp[: args.pair_batch]
                    ok = ~torch.isnan(dy)
                    per = (torch.where(ok, (dp - torch.nan_to_num(dy)) ** 2, torch.zeros_like(dy)).sum(0)) / ok.sum(0).clamp(min=1) / pair_var.to(device)
                    pair_term = per[ok.sum(0) > 0].mean() if ok.any() else loss.new_zeros(())
                    pair_losses.append(pair_term.item())
                    loss = loss + args.pair_loss * pair_term
                if teachers:  # the teachers' averaged prediction, all outputs, pieces and changed copies
                    loss = loss + args.lam * torch.nn.functional.mse_loss(pred, teach(c))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if stage == "main":
                    sched.step()
                losses.append(loss.item())
                step += 1
                if args.max_steps and step >= args.max_steps:
                    break
            r = {c: element_pearson(val[c], predict_multi(model, x_val[c], device)[:, k]) for k, c in enumerate(CELLS)}
            mean_r = float(np.mean(list(r.values())))
            entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_element_pearson": r, "mean": mean_r}
            if args.finetune_epochs:
                entry["stage"] = stage
            if epoch % 5 == 0 or epoch == total_epochs or args.max_steps:  # the new practice chromosomes: logged, not used for choices here
                entry["practice_variants"] = practice_readout(both_orientations(model, pr.seq1, device), both_orientations(model, pr.seq2, device), pr, readout)
            if pair_var is not None:
                entry["pair_loss"] = float(np.mean(pair_losses))
            entry["minutes"] = (time.time() - t0) / 60
            log.append(entry)
            print(f"  epoch {epoch:>2}" + (f" [{stage}]" if args.finetune_epochs else "") + f": loss {entry['train_loss']:.4f}" + (f" (pair term {entry['pair_loss']:.4f})" if "pair_loss" in entry else "") + ", fold 9 " + ", ".join(f"{c} {v:.3f}" for c, v in r.items())
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

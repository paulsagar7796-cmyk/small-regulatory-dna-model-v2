"""Model 3: model 2's student (same network, 8 outputs, same recipe) taught by PRECOMPUTED teacher labels
(AlphaGenome-FT and MPAC) on the three plasmid-MPRA outputs HepG2, K562 and SK-N-SH.
Contract: predeclarations/PREDECLARATION_MODEL3.md.  Practice chromosome 13 only is logged; chromosome 7 is never loaded here.

Data (all in --data_dir, default data/model3; file and key names are the dicts FILES and NPZ_KEYS below):
    train.parquet        same columns as data/more_data/train.parquet (variants on training chromosomes, two 200-letter versions)
    practice13.parquet   same columns, chromosome 13 (logged each epoch, never used to pick the checkpoint)
    mutants.parquet      row-linked to train.parquet: seq1_mut, seq2_mut (the same ~5% of positions changed in both versions)
    lentimpra_excluded.parquet   lentiMPRA pieces to leave out of training (leak guard), as for model 2 (--lenti_excluded none = nobody)
    teacher_ag_train.npz, teacher_mpac.npz     float32 (one value per variant row) per cell and version
    teacher_ag_mutants.npz (and the "_mut" keys of teacher_mpac.npz)   the same for the mutated copies

Rows of the training tensor (as in model 2): lentiMPRA rows first, then version 1 of every variant, then version 2.
A teacher exists only for the variant rows, only at outputs 3, 4, 5 (new_HEPG2, new_K562, new_SKNSH).

Teacher target (units of the standardised label of that output; everything below is computed once, before training)
    z       each teacher's predictions are z-scored per cell over all training rows (both versions)
    mix     (w_ag * z_ag + w_mpac * z_mpac) / (w_ag + w_mpac)             (--w_ag 3 --w_mpac 1)
    t       a + b * mix, with (a, b) = least-squares fit of the standardised label on mix over rows with a measured label
    sy2     squared standard error of the label / std(output)^2           (columns allele{1,2}.log2FC.SE_<CELL>)
    st2     var(label - t) - mean(sy2) over measured rows of the --st2_chrs chromosomes (default 19, 21, X: AlphaGenome-FT's
            validation chromosomes, which it never fitted; on the chromosomes it trained on, its residual is about 2.5x smaller,
            so the teacher would look nearly error-free), floored at 0.01 (the teacher's own error variance)

Modes (loss on a batch; label loss = masked_mse as in model 2: mean per output over its measured rows, then over outputs)
    P0   masked_mse(pred, label) + lam * masked_mse(pred, t restricted to rows whose label is measured)
    P1   target b = (y/sy2 + t/st2) / (1/sy2 + 1/st2) wherever label and teacher both exist, the plain label where there is no
         teacher (lentiMPRA rows), nothing where there is no label (option --teacher_unlabelled: the teacher alone there);
         loss masked_mse(pred, b).  Plus a second stream of MUTATED copies (--mut_batch random rows of mutants.parquet per step, both
         versions, same slide and strand as the main batch, one forward pass) with the teacher t as the only target on the three
         outputs: loss + mut_weight * masked_mse(pred_mut, t_mut).
    P2   P1 plus the gate (where |y - t| > k * sqrt(sy2 + st2) the target is the label alone; --gate_k 3) and the disagreement
         d = |z_ag - z_mpac| per row and output: st2 -> st2 * (1 + d^2) (used in the blend and in the gate) and the mutated-copy loss
         of that row and output is multiplied by 1 / (1 + d^2).
The checkpoint rule is unchanged: best mean Pearson on lentiMPRA fold 9.

    .venv/bin/python src/train_model3.py --mode P0
    .venv/bin/python src/train_model3.py --mode P2 --seed 2 --width 1.75
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
from mpra_train_v2 import RIGHT_FLANK, flank_shift  # noqa: E402
from malinois_body import build_arch  # noqa: E402
from train_more_data import NEW_CELLS, between_adapters, both_orientations, masked_mse, practice_readout, shift_by  # noqa: E402

# ---- names of the files and arrays of the other engineer: change here only ----
FILES = {"train": "train.parquet", "practice": "practice13.parquet", "mutants": "mutants.parquet",
         "lenti_excluded": "lentimpra_excluded.parquet",
         "ag": "teacher_ag_train.npz", "mpac": "teacher_mpac.npz", "ag_mut": "teacher_ag_mutants.npz"}
# (file key, array name) per teacher; {cell} and {v} are filled in with TEACHER_CELL_NAMES and 1 / 2
NPZ_KEYS = {"ag": ("ag", "{cell}_v{v}"), "mpac": ("mpac", "train_{cell}_v{v}"),
            "ag_mut": ("ag_mut", "{cell}_v{v}"), "mpac_mut": ("mpac", "mutants_{cell}_v{v}")}  # label_mpac.py writes <part>_<cell>_v<n>
TEACHER_CELL_NAMES = {"HEPG2": "HepG2", "K562": "K562", "SKNSH": "SKNSH"}  # our cell name -> the cell name inside the array names
# ---------------------------------------------------------------------------------

T_CELLS = ["HEPG2", "K562", "SKNSH"]  # the outputs that have teachers: indices 3, 4, 5 of the 8
T_OUT = [len(CELLS) + NEW_CELLS.index(c) for c in T_CELLS]
ST2_FLOOR, SY2_FLOOR = 0.01, 1e-6


# ======================= the maths (pure numpy, unit-tested) =======================

def zscore_params(a: np.ndarray):
    """Per-column mean and standard deviation, ignoring NaN. a: (rows, cells)."""
    return np.nanmean(a, axis=0), np.nanstd(a, axis=0)


def mix_teachers(z_ag: np.ndarray, z_mpac: np.ndarray, w_ag: float, w_mpac: float) -> np.ndarray:
    """Weighted mean of the two z-scored teachers (NaN in either gives NaN)."""
    return (w_ag * z_ag + w_mpac * z_mpac) / (w_ag + w_mpac)


def calibrate(y: np.ndarray, m: np.ndarray):
    """Per-column least squares y ~ a + b * m over rows where both are present. Returns (a, b)."""
    a, b = np.zeros(y.shape[1]), np.zeros(y.shape[1])
    for k in range(y.shape[1]):
        ok = ~np.isnan(y[:, k]) & ~np.isnan(m[:, k])
        mk, yk = m[ok, k].astype(np.float64), y[ok, k].astype(np.float64)
        b[k] = np.mean((mk - mk.mean()) * (yk - yk.mean())) / np.var(mk)
        a[k] = yk.mean() - b[k] * mk.mean()
    return a, b


def teacher_error_variance(y: np.ndarray, t: np.ndarray, sy2: np.ndarray, rows: np.ndarray | None = None):
    """Per-column variance of (label - teacher) over rows where both exist, minus the mean label error variance sy2 over the same rows
    (a missing sy2 is skipped in that mean), floored at ST2_FLOOR. rows: optional boolean mask of the rows to use.
    Returns (floored, unfloored)."""
    raw = np.zeros(y.shape[1])
    for k in range(y.shape[1]):
        ok = ~np.isnan(y[:, k]) & ~np.isnan(t[:, k])
        if rows is not None:
            ok &= rows
        raw[k] = np.var((y[ok, k] - t[ok, k]).astype(np.float64)) - np.nanmean(sy2[ok, k].astype(np.float64))
    return np.maximum(raw, ST2_FLOOR), raw


def disagreement(z_ag: np.ndarray, z_mpac: np.ndarray) -> np.ndarray:
    return np.abs(z_ag - z_mpac)


def blend(y, t, sy2, st2):
    """Inverse-variance blend of the label y (error variance sy2) and the teacher t (error variance st2)."""
    return (y / sy2 + t / st2) / (1.0 / sy2 + 1.0 / st2)


def gated(y, t, sy2, st2, k: float):
    """True where the teacher disagrees with the label by more than k combined standard deviations."""
    return np.abs(y - t) > k * np.sqrt(sy2 + st2)


def make_target(y, t, sy2, st2, gate_k: float = 0.0, teacher_unlabelled: bool = False):
    """The training target of P1 (gate_k = 0) or P2 (gate_k > 0). y, t, sy2, st2 broadcast to (rows, outputs); st2 may be per row.
    Label and teacher both present: the blend (the label alone where gated); no teacher: the label; no label: NaN (no loss),
    or the teacher alone if teacher_unlabelled."""
    with np.errstate(invalid="ignore", divide="ignore"):
        b = blend(y, t, sy2, st2)
        if gate_k > 0:
            b = np.where(gated(y, t, sy2, st2, gate_k), y, b)
    out = np.where(np.isnan(t), y, b)
    return np.where(np.isnan(y), t if teacher_unlabelled else np.nan, out).astype(np.float32)


def weighted_masked_mse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """masked_mse with a weight per entry: per output sum(w * err) / (number of measured entries), then the mean over outputs with any.
    With w = 1 it is exactly masked_mse."""
    seen = ~torch.isnan(target)
    err = torch.where(seen, w * (pred - torch.nan_to_num(target)) ** 2, torch.zeros_like(target))
    n = seen.sum(0)
    return (err.sum(0)[n > 0] / n[n > 0]).mean()


# ======================= teacher tables =======================

def load_teacher(files: dict, which: str, n: int, tag: str = "") -> np.ndarray:
    """(2n, 3) float32: version 1 of every row, then version 2, for the three cells (NaN stays NaN). files: dir of npz files."""
    fkey, pattern = NPZ_KEYS[which]
    npz = np.load(Path(files["dir"]) / FILES[fkey])
    out = np.empty((2 * n, len(T_CELLS)), dtype=np.float32)
    for j, cell in enumerate(T_CELLS):
        for v in (1, 2):
            key = pattern.format(cell=TEACHER_CELL_NAMES[cell], v=v)
            if key not in npz.files:
                raise KeyError(f"{FILES[fkey]}: no array {key!r}; it has {sorted(npz.files)} (names are set in NPZ_KEYS at the top of train_model3.py)")
            a = np.asarray(npz[key], dtype=np.float32).reshape(-1)
            assert len(a) == n, f"{FILES[fkey]}:{key} has {len(a)} values, expected {n} (one per row of the table)"
            out[(v - 1) * n : v * n, j] = a
    return out


def has_teacher(files: dict, which: str) -> bool:
    fkey, pattern = NPZ_KEYS[which]
    p = Path(files["dir"]) / FILES[fkey]
    if not p.exists():
        return False
    return pattern.format(cell=TEACHER_CELL_NAMES[T_CELLS[0]], v=1) in np.load(p).files


def build_teacher_targets(mode: str, y_n: np.ndarray, sy_n: np.ndarray, tr: dict, mu: dict, args, st2_rows=None) -> dict:
    """y_n, sy_n: (2n, 3) standardised labels and standard errors (NaN = missing) of the variant rows. tr: training teachers
    {'ag': (2n,3), 'mpac': (2n,3)}; mu: the mutated copies' teachers (same keys) or None.
    Returns the arrays the training loop uses and a dict of numbers for the log."""
    zp = {k: zscore_params(tr[k]) for k in ("ag", "mpac")}
    z = lambda k, a: (a - zp[k][0]) / zp[k][1]
    zA, zM = z("ag", tr["ag"]), z("mpac", tr["mpac"])
    m = mix_teachers(zA, zM, args.w_ag, args.w_mpac)
    a, b = calibrate(y_n, m)
    t = (a + b * m).astype(np.float32)
    sy2 = np.maximum(sy_n.astype(np.float64) ** 2, SY2_FLOOR)
    sy2 = np.where(np.isnan(sy_n) & ~np.isnan(y_n), np.nanmean(np.where(np.isnan(y_n), np.nan, sy2), axis=0), sy2)  # a label without SE: mean SE
    st2, st2_raw = teacher_error_variance(y_n, t, sy2, st2_rows)
    d = disagreement(zA, zM)
    info = {"zscore_mean": {k: v[0].tolist() for k, v in zp.items()}, "zscore_std": {k: v[1].tolist() for k, v in zp.items()},
            "w_ag": args.w_ag, "w_mpac": args.w_mpac, "calibration_intercept": a.tolist(), "calibration_slope": b.tolist(),
            "st2_rows": None if st2_rows is None else int(st2_rows.sum()), "st2": st2.tolist(), "st2_before_floor": st2_raw.tolist(), "mean_sy2": np.nanmean(np.where(np.isnan(y_n), np.nan, sy2), axis=0).tolist(),
            "labelled_rows": (~np.isnan(y_n)).sum(0).tolist(), "teacher_missing_rows": np.isnan(t).sum(0).tolist(),
            "pearson_mix_label": [float(np.corrcoef(m[ok, k], y_n[ok, k])[0, 1]) for k in range(len(T_CELLS))
                                  for ok in [~np.isnan(m[:, k]) & ~np.isnan(y_n[:, k])]],
            "mean_disagreement": np.nanmean(d, axis=0).tolist()}
    out = {"t": t}
    if mode == "P0":
        out["t_labelled"] = np.where(np.isnan(y_n), np.nan, t).astype(np.float32)  # the teacher term only on labelled rows
    else:
        st2_row = st2[None, :] * (1 + d ** 2) if mode == "P2" else np.broadcast_to(st2[None, :], y_n.shape)
        st2_row = st2_row * getattr(args, "st2_scale", 1.0)  # < 1 trusts the teacher more (0.01: the target is almost the teacher alone)
        gk = args.gate_k if mode == "P2" else 0.0
        out["target"] = make_target(y_n, t, sy2, st2_row, gk, args.teacher_unlabelled)
        if gk > 0:
            with np.errstate(invalid="ignore"):
                g = gated(y_n, t, sy2, st2_row, gk) & ~np.isnan(y_n) & ~np.isnan(t)
            info["gated_share"] = (g.sum(0) / np.maximum((~np.isnan(y_n) & ~np.isnan(t)).sum(0), 1)).tolist()
        both = ~np.isnan(y_n) & ~np.isnan(t)
        wt_label = (1 / sy2) / (1 / sy2 + 1 / st2_row)  # the label's weight in the blend (before the gate)
        info["mean_label_weight_in_blend"] = [float(np.mean(wt_label[both[:, k], k])) for k in range(len(T_CELLS))]
    if mu is not None:  # mutated copies: the same z-score parameters, the same mix and calibration
        mA, mM = z("ag", mu["ag"]), z("mpac", mu["mpac"])
        mm_ = mix_teachers(mA, mM, args.w_ag, args.w_mpac)
        out["t_mut"] = (a + b * mm_).astype(np.float32)
        dm = disagreement(mA, mM)
        out["w_mut"] = (1.0 / (1 + dm ** 2) if mode == "P2" else np.ones_like(dm)).astype(np.float32)
        info["mut_mean_weight"] = np.nanmean(out["w_mut"], axis=0).tolist()
        info["mut_teacher_std_over_training_teacher_std"] = [float(np.nanstd(out["t_mut"][:, k]) / np.nanstd(t[:, k])) for k in range(len(T_CELLS))]
    return out | {"info": info}


class MutantStream:
    """Endless random order over the mutated rows, own generator (so the main stream of batches is unchanged)."""

    def __init__(self, n: int, gen: torch.Generator):
        self.n, self.gen, self.order, self.pos = n, gen, torch.randperm(n, generator=gen), 0

    def take(self, k: int) -> torch.Tensor:
        parts = []
        while k > 0:
            if self.pos >= self.n:
                self.order, self.pos = torch.randperm(self.n, generator=self.gen), 0
            j = min(k, self.n - self.pos)
            parts.append(self.order[self.pos : self.pos + j])
            self.pos += j
            k -= j
        return torch.cat(parts)


# ======================= training =======================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["P0", "P1", "P2"], required=True)
    ap.add_argument("--data_dir", default="data/model3")
    ap.add_argument("--lenti_excluded", default="", help="parquet of lentiMPRA pieces to drop (default <data_dir>/lentimpra_excluded.parquet; 'none' = no guard)")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--shift", type=int, default=21)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--w_ag", type=float, default=3.0, help="weight of AlphaGenome in the teacher mix")
    ap.add_argument("--w_mpac", type=float, default=1.0, help="weight of MPAC in the teacher mix")
    ap.add_argument("--lam", type=float, default=1.0, help="P0: weight of the teacher term against the label term")
    ap.add_argument("--mut_weight", type=float, default=None, help="weight of the mutated-copy loss (default 1.0 for P1 and P2, 0 for P0; 0 = no stream)")
    ap.add_argument("--mut_batch", type=int, default=0, help="mutated rows per step (default = --batch)")
    ap.add_argument("--gate_k", type=float, default=3.0, help="P2: gate threshold in combined standard deviations")
    ap.add_argument("--st2_chrs", default="19,21,X", help="P1/P2: chromosomes on which the teacher's error variance is measured "
                    "('all' = every training row, the first version's behaviour)")
    ap.add_argument("--st2_scale", type=float, default=1.0, help="P1/P2: multiply the teacher's error variance (0.01 = nearly pure teacher target)")
    ap.add_argument("--teacher_effect", type=float, default=0.0, help="weight of the loss on the TEACHER's effect (version 2 minus 1) over "
                    "--pair_batch random training variants per step, divided by the teacher-effect variance (0 = off)")
    ap.add_argument("--pair_batch", type=int, default=256)
    ap.add_argument("--amp", action="store_true", help="CUDA only: fp16 mixed precision for the training forward pass (memory)")
    ap.add_argument("--teacher_unlabelled", action="store_true", help="P1/P2: where the label is missing but the teacher exists, train on the teacher alone")
    ap.add_argument("--skip_align_check", action="store_true", help="do not stop when the teacher mix correlates below 0.3 with the labels")
    ap.add_argument("--max_steps", type=int, default=0, help="for a quick test only")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.mut_weight is None:
        args.mut_weight = 0.0 if args.mode == "P0" else 1.0
    use_mut = args.mut_weight > 0
    mut_batch = args.mut_batch or args.batch
    data_dir = Path(args.data_dir)
    files = {"dir": data_dir}
    name = f"m3_{args.mode}_e{args.epochs}_seed{args.seed}" + (f"_w{args.width}" if args.width != 1.0 else "") + (f"_{args.tag}" if args.tag else "")
    run_dir = Path(RUNS) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    n_out = len(CELLS) + len(NEW_CELLS)

    # ---- lentiMPRA rows (as model 2; the leak-guard list of this split) ----
    everything = {c: load(c, list(range(1, VALIDATION_FOLD + 1))) for c in CELLS}  # folds 1 to 9 only: fold 10 is not loaded
    held_out = set()
    for c in CELLS:
        d = pd.read_csv(Path("data/mpra") / f"{c}.tsv", sep="\t", usecols=["seq", "fold", "rev"])
        held_out |= set(d.seq[(d.rev == 0) & (d.fold >= VALIDATION_FOLD)])
    ex_path = args.lenti_excluded or str(data_dir / FILES["lenti_excluded"])
    if ex_path == "none":
        excluded = set()
    else:
        assert Path(ex_path).exists(), f"{ex_path} is missing: the lentiMPRA leak guard of this split (use --lenti_excluded none to run without)"
        excluded = set(pd.read_parquet(ex_path).seq)
    parts = []
    for k, c in enumerate(CELLS):
        d = everything[c]
        d = d[d.fold.isin(TRAIN_FOLDS)]
        parts.append(d[~d.seq.isin(held_out) & ~d.seq.isin(excluded)].assign(cell=k))
    trl = pd.concat(parts, ignore_index=True)
    x = [encode(trl.seq)]
    y = np.full((len(trl), n_out), np.nan, dtype=np.float32)
    y[np.arange(len(trl)), trl.cell.to_numpy()] = trl.mean_value.to_numpy()
    labels = [y]
    lenti_rows = len(trl)

    # ---- variant rows: both versions, all five new outputs ----
    nt = pd.read_parquet(data_dir / FILES["train"])
    n_var = len(nt)
    se_raw = np.full((2 * n_var, len(T_CELLS)), np.nan, dtype=np.float32)
    for v, (a, col) in enumerate((("allele1", "seq1"), ("allele2", "seq2"))):
        x.append(between_adapters(nt[col]))
        y = np.full((n_var, n_out), np.nan, dtype=np.float32)
        for j, cell in enumerate(NEW_CELLS):
            y[:, len(CELLS) + j] = nt[f"{a}.log2FC_{cell}"].to_numpy()
        labels.append(y)
        for j, cell in enumerate(T_CELLS):
            se_raw[v * n_var : (v + 1) * n_var, j] = nt[f"{a}.log2FC.SE_{cell}"].to_numpy()
    y_raw = np.concatenate(labels)
    mean, std = np.nanmean(y_raw, axis=0), np.nanstd(y_raw, axis=0)  # as model 2
    x_tr = torch.from_numpy(np.concatenate(x))
    y_tr = torch.from_numpy((y_raw - mean) / std)
    tgt_tr, t0_tr = y_tr, None  # what the label loss is computed against; P0's separate teacher term

    # ---- teachers ----
    tr = {"ag": load_teacher(files, "ag", n_var), "mpac": load_teacher(files, "mpac", n_var)}
    y_n = y_tr[lenti_rows:, T_OUT].numpy()
    sy_n = se_raw / std[T_OUT][None, :]
    mu = None
    if use_mut:
        mu = {"ag": load_teacher(files, "ag_mut", n_var)}
        if has_teacher(files, "mpac_mut"):
            mu["mpac"] = load_teacher(files, "mpac_mut", n_var)
        else:
            print("WARNING: no MPAC labels for the mutated copies; their teacher is AlphaGenome alone (disagreement 0)", flush=True)
            mu["mpac"] = mu["ag"]
    st2_rows = None
    if args.st2_chrs != "all":
        st2_rows = np.tile(nt.chr.astype(str).isin(args.st2_chrs.split(",")).to_numpy(), 2)
        assert st2_rows.sum() > 1000, f"only {st2_rows.sum()} rows on --st2_chrs {args.st2_chrs}"
    T = build_teacher_targets(args.mode, y_n, sy_n, tr, mu, args, st2_rows)
    info = T["info"]
    for k, cell in enumerate(T_CELLS):
        print(f"teacher mix vs label, {cell}: Pearson {info['pearson_mix_label'][k]:.3f}, calibration a {info['calibration_intercept'][k]:.3f} b {info['calibration_slope'][k]:.3f}, "
              f"st2 {info['st2'][k]:.3f} (before floor {info['st2_before_floor'][k]:.3f})", flush=True)
    if not args.skip_align_check:
        assert min(info["pearson_mix_label"]) > 0.3, f"teacher mix correlates {info['pearson_mix_label']} with the labels: rows misaligned? (--skip_align_check)"

    def lay(a: np.ndarray, rows: int, row0: int) -> torch.Tensor:
        full = np.full((rows, n_out), np.nan, dtype=np.float32)
        view = full[row0 : row0 + len(a)]
        view[:, T_OUT] = a
        return torch.from_numpy(full)

    total_rows = len(x_tr)
    if args.mode == "P0":
        t0_tr = lay(T["t_labelled"], total_rows, lenti_rows)
    else:
        blend_full = y_tr.clone()  # lentiMPRA rows and the outputs without a teacher keep the label
        blend_full[lenti_rows:, T_OUT] = torch.from_numpy(T["target"])
        tgt_tr = blend_full
    if use_mut:
        mx = pd.read_parquet(data_dir / FILES["mutants"])
        assert len(mx) == n_var, "mutants.parquet is not row-linked to train.parquet"
        x_mut = torch.from_numpy(np.concatenate([between_adapters(mx.seq1_mut), between_adapters(mx.seq2_mut)]))
        t_mut = lay(T["t_mut"], 2 * n_var, 0)
        w_mut = lay(T["w_mut"], 2 * n_var, 0)
        w_mut = torch.nan_to_num(w_mut, nan=1.0)
        mstream = MutantStream(2 * n_var, torch.Generator().manual_seed(args.seed + 15485863))

    t_eff = eff_var = None
    if args.teacher_effect > 0:  # the teacher mix's effect per variant (version 2 minus version 1), standardised units
        tt = torch.from_numpy(T["t"])
        t_eff = tt[n_var:] - tt[:n_var]
        eff_var = torch.from_numpy(np.nanvar(t_eff.numpy(), axis=0)).float()
        pgen = torch.Generator().manual_seed(args.seed + 104729)
        print("teacher effect variance per output (standardised units): " + ", ".join(f"{v:.4f}" for v in eff_var), flush=True)
    val = {c: everything[c][everything[c].fold == VALIDATION_FOLD].reset_index(drop=True) for c in CELLS}
    x_val = {c: torch.from_numpy(encode(val[c].seq)) for c in CELLS}
    pr = pd.read_parquet(data_dir / FILES["practice"])
    readout = {f"new_{cell}": (cell, len(CELLS) + NEW_CELLS.index(cell)) for cell in ("HEPG2", "K562")}
    flank = torch.from_numpy(encode(pd.Series([RIGHT_FLANK]))[0])

    model = build_arch("legnet", n_out, width=args.width).to(device)
    steps_per_epoch = (total_rows + args.batch - 1) // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr / 25, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.3, three_phase=False, cycle_momentum=False)
    config = {**vars(args), "outputs": CELLS + [f"new_{c}" for c in NEW_CELLS], "parameters": count_parameters(model),
              "lentimpra_rows": lenti_rows, "variant_rows_per_version": n_var, "mutant_rows": 2 * n_var if use_mut else 0,
              "lentimpra_rows_removed_by_guard": int(sum((everything[c].fold.isin(TRAIN_FOLDS) & everything[c].seq.isin(excluded)).sum() for c in CELLS)),
              "label_mean": mean.tolist(), "label_std": std.tolist(), "teacher_output_indices": T_OUT, "teacher": info,
              "mut_batch": mut_batch, "name": name}
    print(f"{name}: {config['parameters']:,} parameters; {lenti_rows:,} lentiMPRA rows + {2 * n_var:,} variant rows"
          + (f" + {2 * n_var:,} mutated rows ({mut_batch} per step, weight {args.mut_weight})" if use_mut else "")
          + f"; {steps_per_epoch:,} steps per epoch; device {device}", flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    mgen = torch.Generator().manual_seed(args.seed + 32452843)  # slides of the mutated copies
    use_amp = bool(args.amp) and device.type == "cuda"  # fp16 autocast + loss scaling, to fit the 4M body with the mutated stream on a 16 GB T4
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log, best, t0, step = [], -1.0, time.time(), 0
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(x_tr), generator=gen)
        losses, l_lab, l_tea, l_mut, l_eff = [], [], [], [], []
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            c = x_tr[idx]
            if args.shift:
                c = flank_shift(c, args.shift, flank, gen)
            n_main = len(c)
            if use_mut:
                mi = mstream.take(mut_batch)
                cm = x_mut[mi]
                if args.shift:
                    cm = flank_shift(cm, args.shift, flank, mgen)
                c = torch.cat([c, cm])
            if t_eff is not None:  # both versions of --pair_batch random variants, same slide, appended last
                pair_v = torch.randint(0, n_var, (args.pair_batch,), generator=pgen)
                pc = x_tr[torch.cat([lenti_rows + pair_v, lenti_rows + n_var + pair_v])]
                if args.shift:
                    pc = shift_by(pc, torch.randint(0, args.shift + 1, (args.pair_batch, 1), generator=pgen).repeat(2, 1), flank)
                n_before_pairs = len(c)
                c = torch.cat([c, pc])
            c = c.to(device)
            if torch.rand(1, generator=gen).item() < 0.5:  # one strand for the whole batch
                c = reverse_complement_codes(c)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(onehot(c))
            pred = pred.float()
            pm = pred[:n_main]
            loss = masked_mse(pm, tgt_tr[idx].to(device))
            l_lab.append(loss.item())
            if t0_tr is not None and args.lam > 0:  # P0's teacher term
                tl = args.lam * masked_mse(pm, t0_tr[idx].to(device))
                l_tea.append(tl.item())
                loss = loss + tl
            if use_mut:
                ml = args.mut_weight * weighted_masked_mse(pred[n_main : n_main + len(mi)], t_mut[mi].to(device), w_mut[mi].to(device))
                l_mut.append(ml.item())
                loss = loss + ml
            if t_eff is not None:
                pp = pred[n_before_pairs:][:, T_OUT]
                dp = pp[args.pair_batch:] - pp[: args.pair_batch]
                dt = t_eff[pair_v].to(device)
                ok = ~torch.isnan(dt)
                per = torch.where(ok, (dp - torch.nan_to_num(dt)) ** 2, torch.zeros_like(dt)).sum(0) / ok.sum(0).clamp(min=1) / eff_var.to(device)
                et = args.teacher_effect * per.mean()
                l_eff.append(et.item())
                loss = loss + et
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        r = {c: element_pearson(val[c], predict_multi(model, x_val[c], device)[:, k]) for k, c in enumerate(CELLS)}
        mean_r = float(np.mean(list(r.values())))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "label_loss": float(np.mean(l_lab)), "validation_element_pearson": r, "mean": mean_r}
        if l_tea:
            entry["teacher_loss"] = float(np.mean(l_tea))
        if l_mut:
            entry["mutant_loss"] = float(np.mean(l_mut))
        if l_eff:
            entry["teacher_effect_loss"] = float(np.mean(l_eff))
        pv = practice_readout(both_orientations(model, pr.seq1, device), both_orientations(model, pr.seq2, device), pr, readout)
        entry["practice13"] = pv
        entry["practice13_mean_effect"] = float(np.mean([v["effect"] for v in pv.values()]))
        entry["practice13_mean_activity"] = float(np.mean([v["activity"] for v in pv.values()]))
        entry["minutes"] = (time.time() - t0) / 60
        log.append(entry)
        print(f"  epoch {epoch:>2}: loss {entry['train_loss']:.4f} (label {entry['label_loss']:.4f}"
              + (f", teacher {entry['teacher_loss']:.4f}" if l_tea else "") + (f", mutants {entry['mutant_loss']:.4f}" if l_mut else "")
              + "), fold 9 " + ", ".join(f"{c} {v:.3f}" for c, v in r.items()) + f", mean {mean_r:.3f}; chr13 effect {entry['practice13_mean_effect']:.3f}"
              + f" activity {entry['practice13_mean_activity']:.3f}, {entry['minutes']:.1f} min", flush=True)
        if mean_r > best:
            best = mean_r
            torch.save(model.state_dict(), run_dir / "best.pt")
        (run_dir / "log.json").write_text(json.dumps({"config": config, "log": log}, indent=1))
        if args.max_steps and step >= args.max_steps:
            break
    print(f"done in {(time.time() - t0) / 60:.0f} min: best fold-9 mean Pearson {best:.3f}", flush=True)


if __name__ == "__main__":
    main()

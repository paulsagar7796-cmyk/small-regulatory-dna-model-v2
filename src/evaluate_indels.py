"""Indel (insertion / deletion) variant-effect test: score model 2, model 1, the published MPRA-LegNet, the 6-mer baseline and
MPAC on natural indels. Tables: src/prepare_indels.py. Scoring is exactly src/evaluate_locked.py (per cell type HepG2 and K562:
Pearson between the measured effect log2Skew and the predicted effect = version 2 minus version 1, over variants with both
versions measured, log2Skew present and at least one version active, log2FC >= 1; the two cell types averaged; the same 2,000
resamples of 100,000-letter genome blocks for every model and every subset).

    .venv/bin/python src/evaluate_indels.py --check-pipeline   the scoring code on the SINGLE-LETTER practice table must reproduce
                                                               model 2 (1e-4) and MPAC (1e-3) from earlier result files
    .venv/bin/python src/evaluate_indels.py --practice         practice chromosomes 6, 13, 20 -> results/practice/indels_practice.json
    .venv/bin/python src/evaluate_indels.py --locked           NOT for practice work. Refuses unless predeclarations/PREDECLARATION_INDELS.md
                                                               contains **Status: CONFIRMED**, its students block says "fixed": true, and
                                                               results/locked/INDELS.json does not exist. Run once.

Framing (fixed, not tuned). A piece of length 200 - k (0 <= k <= 54):
  our LegNet-family models (230 letters): orientation 0 = LEFT + piece + RIGHT + RIGHT_FLANK[:k]; orientation 1 = LEFT +
  reverse_complement(piece) + RIGHT + RIGHT_FLANK[:k]; predictions averaged over the two orientations. RIGHT_FLANK is the
  construct's own sequence after the piece (the models were trained with a slide of up to 21 letters filled with it).
  MPAC (600 letters): left = MPRA_UPSTREAM[-(200 + k // 2):], right = MPRA_DOWNSTREAM[:200 + k - k // 2]; forward = left + piece +
  right; reverse = left + reverse_complement(piece) + right; prediction = mean of the 10-network ensemble on both.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
import modeling_mpac as mm  # noqa: E402
from evaluate_locked import PRE as PRE_MORE  # noqa: E402
from evaluate_practice import BLOCK, MIN_ACTIVITY, MORE, TABLE, model2  # noqa: E402
from evaluate_scan import LEFT, N_BOOT, RIGHT, models, pieces  # noqa: E402
from mpac_reference import CELL_INDEX, MPAC_DIR, load_ensemble, pick_device  # noqa: E402
from mpra_data import encode  # noqa: E402
from mpra_train import RUNS, SETTINGS  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402
from mpra_train_v2 import RIGHT_FLANK, build  # noqa: E402

PRE = Path("predeclarations/PREDECLARATION_INDELS.md")
INDELS = Path("data/indels")
PRACTICE_OUT = Path("results/practice/indels_practice.json")
PIPELINE_OUT = Path("results/practice/indels_pipeline_check.json")
LOCKED_OUT = Path("results/locked/INDELS.json")
NEAR = 27  # shared 24-letter words with one training piece = near-copy
COMP = str.maketrans("ACGT", "TGCA")


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


# ---------------------------------------------------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------------------------------------------------
def frame_legnet(seqs: list[str]) -> np.ndarray:
    """(n, 2, 230) uint8 codes. Piece of length 200 - k: LEFT + piece + RIGHT + RIGHT_FLANK[:k]; orientation 1 has the
    piece reverse-complemented between the same adapters."""
    f, r = [], []
    for s in seqs:
        k = 200 - len(s)
        assert 0 <= k <= 54, f"piece of length {len(s)}"
        tail = RIGHT + RIGHT_FLANK[:k]
        f.append(LEFT + s + tail)
        r.append(LEFT + rc(s) + tail)
    out = np.zeros((len(seqs), 2, 230), np.uint8)
    out[:, 0], out[:, 1] = encode(pd.Series(f)), encode(pd.Series(r))
    return out


def _mpac_inputs(seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    fwd, rev = [], []
    for s in seqs:
        k = 200 - len(s)
        assert 0 <= k <= 54, f"piece of length {len(s)}"
        left, right = mm.MPRA_UPSTREAM[-(200 + k // 2):], mm.MPRA_DOWNSTREAM[:200 + k - k // 2]
        fwd.append(left + s + right)
        rev.append(left + rc(s) + right)
    return encode(pd.Series(fwd)), encode(pd.Series(rev))


@torch.no_grad()
def mpac_predict(ens, seqs: list[str], device, batch_size: int = 256) -> np.ndarray:
    """(n, n_outputs) in MPAC's own output order: the ensemble mean on the forward and the reverse-complemented insert,
    averaged (the way MPACEnsemble.predict combines them). The flanks grow with the indel size so the input is 600 letters."""
    fwd, rev = _mpac_inputs(seqs)
    assert fwd.shape[1] == rev.shape[1] == 600 and fwd.max() <= 3 and rev.max() <= 3
    oh = lambda c: torch.nn.functional.one_hot(torch.from_numpy(c).long(), 4).permute(0, 2, 1).float()  # noqa: E731  (A, C, G, T = mm.STANDARD_NT)
    out = []
    for a in range(0, len(seqs), batch_size):
        p = ens(oh(fwd[a:a + batch_size]).to(device)) + ens(oh(rev[a:a + batch_size]).to(device))
        out.append((p / 2.0).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------------------------------------------------
def run_legnet(fn, t: pd.DataFrame) -> dict:
    """fn: (m, 230) codes -> {cell: (m,)} . Returns {cell: (pred of seq1, pred of seq2)}, orientation-averaged."""
    got = {}
    for v in ("seq1", "seq2"):
        x = frame_legnet(t[v].tolist())
        a = fn(x.reshape(-1, 230))
        got[v] = {c: np.asarray(a[c]).reshape(len(t), 2).mean(1) for c in TABLE}
    return {c: (got["seq1"][c], got["seq2"][c]) for c in TABLE}


def run_mpac(t: pd.DataFrame, device) -> dict:
    config = json.loads((MPAC_DIR / "config.json").read_text())
    cols = {c: config["output_names"].index(n) for c, n in CELL_INDEX.items()}
    out = {c: (np.full(len(t), np.nan), np.full(len(t), np.nan)) for c in TABLE}
    for chrom in sorted(t.chr.astype(str).unique()):
        ens = load_ensemble(chrom, device)  # the 10 networks whose test pair holds this chromosome
        idx = np.flatnonzero((t.chr.astype(str) == chrom).to_numpy())
        for v, col in enumerate(("seq1", "seq2")):
            p = mpac_predict(ens, t[col].to_numpy()[idx].tolist(), device)
            for c, k in cols.items():
                out[c][v][idx] = p[:, k]
        del ens
    assert all(np.isfinite(a).all() for c in TABLE for a in out[c]), "missing MPAC predictions"
    return out


def run_sknsh(student: dict, t: pd.DataFrame, device) -> dict:
    """Model 2's SK-N-SH output (model 1 and the published model have none): {name: (p1, p2)}."""
    res = {}
    for name, run in (("model2", student["main"]), ("model2_seed2", student["seed2"])):
        outs = json.loads((Path(RUNS) / run / "log.json").read_text())["config"]["outputs"]
        m = build("legnet", len(outs)).to(device)
        m.load_state_dict(torch.load(Path(RUNS) / run / "best.pt", map_location=device, weights_only=True))
        k = outs.index("new_SKNSH")
        p = [predict_multi(m, torch.from_numpy(frame_legnet(t[v].tolist()).reshape(-1, 230)), device)[:, k].reshape(len(t), 2).mean(1) for v in ("seq1", "seq2")]
        res[name] = tuple(p)
    return res


def framing_asserts(device) -> dict:
    """1. frame_legnet equals evaluate_scan.pieces on 1,000 200-letter practice pieces. 2. mpac_predict equals ensemble.predict
    on 200-letter practice pieces of one chromosome (to 1e-5)."""
    t = pd.read_parquet(MORE / "practice.parquet", columns=["chr", "seq1"])
    s = t.seq1.iloc[:1000].tolist()
    assert (frame_legnet(s) == pieces(s)).all(), "frame_legnet differs from evaluate_scan.pieces"
    ch = str(t.chr.iloc[0])
    seqs = t.seq1[t.chr.astype(str) == ch].iloc[:300].tolist()
    ens = load_ensemble(ch, device)
    mine = mpac_predict(ens, seqs, device)
    ref = ens.predict(seqs, batch_size=256, device=device).numpy()
    diff = float(np.abs(mine - ref).max())
    assert diff <= 1e-5, f"mpac_predict differs from ensemble.predict by {diff}"
    return {"frame_legnet_equals_pieces_on": len(s), "mpac_chromosome": ch, "mpac_pieces": len(seqs), "mpac_max_abs_difference": diff,
            "device": str(device)}


# ---------------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------------
def _corr_rows(y: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Pearson of y (n,) with each row of d (m, n); nan when degenerate."""
    if len(y) < 3:
        return np.full(len(d), np.nan)
    yc, dc = y - y.mean(), d - d.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (dc @ yc) / (np.sqrt((dc ** 2).sum(1)) * np.sqrt((yc ** 2).sum()))


def score(t: pd.DataFrame, pred: dict, subsets: dict, boot: bool = True) -> dict:
    """pred: name -> {cell: (p1, p2)}. subsets: name -> boolean mask over t. Returns the evaluate_locked-style numbers per
    model and subset, with block-bootstrap intervals (the same resamples for all) and model2-minus-this intervals."""
    names = list(pred)
    block = (t.chr.astype(str) + ":" + (t.pos // BLOCK).astype(str)).to_numpy()
    blocks = {b: np.flatnonzero(block == b) for b in np.unique(block)}
    bn, rng = list(blocks), np.random.default_rng(SETTINGS["seed"])
    draws = [np.concatenate([blocks[b] for b in rng.choice(bn, len(bn))]) for _ in range(N_BOOT)] if boot else []
    res = {n: {} for n in names}
    for sname, smask in subsets.items():
        bootm = {n: np.zeros(len(draws)) for n in names}
        for cell, tab in TABLE.items():
            a1, a2, skew = (t[f"{k}_{tab}"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
            ok = ~np.isnan(a1) & ~np.isnan(a2) & smask
            e = ok & ~np.isnan(skew) & (np.maximum(a1, a2) >= MIN_ACTIVITY)
            D = np.stack([pred[n][cell][1] - pred[n][cell][0] for n in names])
            r = _corr_rows(skew[e], D[:, e])
            act = np.concatenate([a1[ok], a2[ok]])
            for i, n in enumerate(names):
                p1, p2 = pred[n][cell]
                res[n].setdefault(sname, {})[cell] = {
                    "effect": float(r[i]), "effect_variants": int(e.sum()),
                    "effect_spearman": float(spearmanr(skew[e], D[i, e])[0]) if e.sum() > 2 else float("nan"),
                    "activity": float(np.corrcoef(act, np.concatenate([p1[ok], p2[ok]]))[0, 1]) if ok.sum() > 2 else float("nan"),
                    "activity_variants": int(ok.sum())}
            for j, dr in enumerate(draws):
                dr = dr[e[dr]]
                rr = _corr_rows(skew[dr], D[:, dr])
                for i, n in enumerate(names):
                    bootm[n][j] += rr[i] / len(TABLE)
        for n in names:
            m = res[n][sname]
            for k in ("effect", "effect_spearman", "activity"):
                m["mean_" + k] = float(np.mean([m[c][k] for c in TABLE]))
            if boot:
                good = np.isfinite(bootm[n])
                m["mean_effect_ci95"] = [float(v) for v in np.percentile(bootm[n][good], [2.5, 97.5])] if good.sum() > 100 else None
                m["bootstrap_draws_degenerate"] = int((~good).sum())
                if "model2" in bootm:
                    dd = bootm["model2"] - bootm[n]
                    dd = dd[np.isfinite(dd)]
                    m["model2_minus_this_ci95"] = [float(v) for v in np.percentile(dd, [2.5, 97.5])] if len(dd) > 100 else None
    return res


def subsets_for(t: pd.DataFrame) -> dict:
    """Pre-set subsets. PRIMARY: size 1-21 and no near-copy (best_shared_words < 27)."""
    size, bsw, kind = t["size"].to_numpy(), t.best_shared_words.to_numpy(), t.kind.to_numpy()
    clean, small = bsw < NEAR, size <= 21
    return {"primary_size1-21_no_near_copy": small & clean,
            "size1-21_all": small,
            "size1-54_all": np.ones(len(t), bool),
            "deletions_primary": small & clean & (kind == "deletion"),
            "insertions_primary": small & clean & (kind == "insertion"),
            "size1_no_near_copy": (size == 1) & clean,
            "size2-21_no_near_copy": (size >= 2) & small & clean,
            "size22-54_no_near_copy": (size >= 22) & clean,
            "near_copies_only_size1-21": small & ~clean}


# ---------------------------------------------------------------------------------------------------------------------
def build_fns(student: dict, device, names: list[str] | None = None) -> dict:
    fns = {}
    if names is None or any(n in names for n in ("student_v2", "student_v2_seed2", "published", "baseline")):
        base = models(device, {})
        fns.update({k: base[k] for k in ("student_v2", "student_v2_seed2", "published", "baseline") if names is None or k in names})
    for n, run in (("model2", student["main"]), ("model2_seed2", student["seed2"])):
        if names is None or n in names:
            fns[n] = model2(run, device)["m2_new"]
    return fns


ORDER = ["model2", "model2_seed2", "student_v2", "student_v2_seed2", "published", "baseline", "mpac"]


def predict_all(t: pd.DataFrame, student: dict, device, mpac_device, names: list[str] | None = None) -> dict:
    fns = build_fns(student, device, names)
    pred = {n: run_legnet(fns[n], t) for n in ORDER if n in fns}
    if names is None or "mpac" in names:
        pred["mpac"] = run_mpac(t, mpac_device)
    return pred


def check_pipeline(student: dict, device, mpac_device) -> None:
    """Check 2: the scoring code on the single-letter practice table reproduces earlier results."""
    asserts = framing_asserts(mpac_device)
    print("framing asserts passed:", asserts, flush=True)
    t = pd.read_parquet(MORE / "practice.parquet")
    pred = predict_all(t, student, device, mpac_device, ["model2", "mpac"])
    res = score(t, pred, {"all": np.ones(len(t), bool)}, boot=False)
    want_m2 = json.loads(Path("results/practice/locked_script_check.json").read_text())["models"]["model2"]["mean_effect"]
    want_mp = json.loads(Path("results/practice/mpac_reference.json").read_text())["models"]["mpac"]["mean_effect"]
    got_m2, got_mp = res["model2"]["all"]["mean_effect"], res["mpac"]["all"]["mean_effect"]
    ok = abs(got_m2 - want_m2) < 1e-4 and abs(got_mp - want_mp) < 1e-3
    print(f"model 2 mean effect: here {got_m2:.6f} vs earlier {want_m2:.6f} (diff {got_m2 - want_m2:+.2e}, tolerance 1e-4)")
    print(f"MPAC    mean effect: here {got_mp:.6f} vs earlier {want_mp:.6f} (diff {got_mp - want_mp:+.2e}, tolerance 1e-3)")
    PIPELINE_OUT.write_text(json.dumps({"pass": bool(ok), "framing": asserts, "model2": {"here": got_m2, "earlier": want_m2},
                                        "mpac": {"here": got_mp, "earlier": want_mp}, "variants": len(t)}, indent=2))
    if not ok:
        sys.exit("STOP: the scoring pipeline does not reproduce the earlier numbers.")
    print("pipeline reproduction: PASS")


def pipeline_ok() -> bool:
    return PIPELINE_OUT.exists() and bool(json.loads(PIPELINE_OUT.read_text()).get("pass"))


def run(table: Path, student: dict, out: Path) -> None:
    device, mpac_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"), pick_device()
    print("devices:", device, mpac_device, flush=True)
    asserts = framing_asserts(mpac_device)
    print("framing asserts passed:", asserts, flush=True)
    t = pd.read_parquet(table)
    assert (t["size"] == (t.ref.str.len() - t.alt.str.len()).abs()).all()
    pred = predict_all(t, student, device, mpac_device)
    sub = subsets_for(t)
    res = score(t, pred, sub)
    sk = run_sknsh(student, t, device)
    sk_res = {}
    for n, (p1, p2) in sk.items():
        sk_res[n] = {}
        for sname, smask in sub.items():
            a1, a2, skew = (t[f"{k}_SKNSH"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
            e = ~np.isnan(a1) & ~np.isnan(a2) & ~np.isnan(skew) & (np.maximum(a1, a2) >= MIN_ACTIVITY) & smask
            sk_res[n][sname] = {"effect": float(np.corrcoef(skew[e], (p2 - p1)[e])[0, 1]) if e.sum() > 2 else None, "effect_variants": int(e.sum())}
    out_d = {"table": str(table), "variants": len(t), "framing_checks": asserts, "student": student,
             "subset_sizes": {k: int(v.sum()) for k, v in sub.items()},
             "subset_definitions": {"primary_size1-21_no_near_copy": "size <= 21 and best_shared_words < 27",
                                    "size1-21_all": "size <= 21, near-copies included", "size1-54_all": "all sizes, near-copies included",
                                    "deletions_primary": "primary filter, deletions", "insertions_primary": "primary filter, insertions",
                                    "size1_no_near_copy": "size 1 and best_shared_words < 27", "size2-21_no_near_copy": "size 2 to 21 and best_shared_words < 27",
                                    "size22-54_no_near_copy": "size 22 to 54 and best_shared_words < 27",
                                    "near_copies_only_size1-21": "size <= 21 and best_shared_words >= 27"},
             "models": res, "SKNSH": sk_res,
             "notes": "bootstrap: 2,000 resamples of 100,000-letter genome blocks, same resamples for all models and subsets"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_d, indent=2))
    report(out_d)


def report(r: dict) -> None:
    fmt = lambda c: "n/a" if c is None else f"{c[0]:.3f} to {c[1]:.3f}"  # noqa: E731
    for sname, size in r["subset_sizes"].items():
        print(f"\n== {sname}  ({size} variants)")
        for n in ORDER:
            m = r["models"][n][sname]
            print(f"{n:<17} effect {m['mean_effect']:.4f} (95% {fmt(m['mean_effect_ci95'])}) " + " ".join(f"{c} {m[c]['effect']:.3f}" for c in TABLE)
                  + f"  spearman {m['mean_effect_spearman']:.3f}  activity {m['mean_activity']:.3f}  scored {[m[c]['effect_variants'] for c in TABLE]}"
                  + (f"  model2 minus this {fmt(m['model2_minus_this_ci95'])}" if n != "model2" else ""))
    print("\nSK-N-SH", json.dumps(r["SKNSH"]["model2"]["primary_size1-21_no_near_copy"]), json.dumps(r["SKNSH"]["model2_seed2"]["primary_size1-21_no_near_copy"]))


def main() -> None:
    more = json.loads(re.search(r"<!-- students: (.*?) -->", PRE_MORE.read_text()).group(1))
    if "--check-pipeline" in sys.argv:
        check_pipeline(more, torch.device("mps" if torch.backends.mps.is_available() else "cpu"), pick_device())
    elif "--practice" in sys.argv:
        if not pipeline_ok():
            sys.exit("Stopped: run --check-pipeline first (results/practice/indels_pipeline_check.json must say pass).")
        run(INDELS / "practice.parquet", more, PRACTICE_OUT)
    elif "--locked" in sys.argv:
        text = PRE.read_text() if PRE.exists() else ""
        student = json.loads(re.search(r"<!-- students: (.*?) -->", text).group(1)) if "<!-- students:" in text else {}
        if "**Status: CONFIRMED**" not in text or not student.get("fixed"):
            sys.exit("Stopped: the pre-declaration is missing, not CONFIRMED, or the students are not fixed.")
        if LOCKED_OUT.exists():
            sys.exit("Stopped: the locked indel test has already been run. It is run once.")
        if not pipeline_ok():
            sys.exit("Stopped: results/practice/indels_pipeline_check.json missing or not pass.")
        run(INDELS / "locked.parquet", student, LOCKED_OUT)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

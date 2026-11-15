"""Model 2 (more data): the locked test on chromosomes 5, 14 and 22. RUN ONCE.

    .venv/bin/python src/evaluate_locked.py --practice   (the same code on the practice chromosomes: a check of this script)
    .venv/bin/python src/evaluate_locked.py --locked     (the test; refuses unless the pre-declaration is CONFIRMED,
                                                           the students are fixed there, and no result file exists yet)

Rules: predeclarations/PREDECLARATION_MORE_DATA.md. Measure: per cell type (HepG2, K562), Pearson between the measured
effect (log2Skew) and the predicted effect over variants with at least one active version (log2FC >= 1); the two cell
types averaged; 2,000 resamples of 100,000-letter genome blocks, the same resample for every model.
Model 2 is read through its new outputs (fixed in the pre-declaration).
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
from evaluate_practice import BLOCK, MIN_ACTIVITY, MORE, TABLE, model2  # noqa: E402
from evaluate_scan import N_BOOT, models, pieces  # noqa: E402
from mpra_data import CELLS, TRAIN_FOLDS, encode  # noqa: E402
from mpra_train import RUNS, SETTINGS  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402
from mpra_train_v2 import build  # noqa: E402

PRE = Path("predeclarations/PREDECLARATION_MORE_DATA.md")
WORD = 24


def words(codes: np.ndarray) -> np.ndarray:
    """(n, L) codes -> (n, L-23) canonical 24-letter words as integers (a word and its reverse complement are one word)."""
    c = codes.astype(np.uint64)
    n, L = c.shape
    f, r = np.zeros((n, L - WORD + 1), np.uint64), np.zeros((n, L - WORD + 1), np.uint64)
    for j in range(WORD):
        f = f * np.uint64(4) + c[:, j : L - WORD + 1 + j]
        r = r + (np.uint64(3) - c[:, j : L - WORD + 1 + j]) * np.uint64(4**j)
    return np.minimum(f, r)


def training_words() -> np.ndarray:
    """Every 24-letter word of every training piece (new table and lentiMPRA folds 1 to 8)."""
    nt = pd.read_parquet(MORE / "train.parquet", columns=["seq1", "seq2"])
    parts = [np.unique(words(encode(nt[c]))) for c in ("seq1", "seq2")]
    for c in CELLS:
        d = pd.read_csv(Path("data/mpra") / f"{c}.tsv", sep="\t", usecols=["seq", "fold", "rev"])
        parts.append(np.unique(words(encode(d.seq[(d.rev == 0) & d.fold.isin(TRAIN_FOLDS)].drop_duplicates()))))
    return np.unique(np.concatenate(parts))


def evaluate(table: Path, student: dict, device) -> dict:
    t = pd.read_parquet(table)
    fns = models(device, {})  # student_v2 (model 1), student_v2_seed2, published, baseline
    for name, run in (("model2", student["main"]), ("model2_seed2", student["seed2"])):
        fns[name] = model2(run, device)["m2_new"]
    x1, x2 = pieces(t.seq1.tolist()), pieces(t.seq2.tolist())
    pred = {}
    for n, fn in fns.items():
        a, b = (fn(x.reshape(-1, x.shape[-1])) for x in (x1, x2))
        pred[n] = {c: (a[c].reshape(len(t), 2).mean(1), b[c].reshape(len(t), 2).mean(1)) for c in TABLE}
    tw = training_words()
    fresh = ~(np.isin(words(encode(t.seq1)), tw).any(1) | np.isin(words(encode(t.seq2)), tw).any(1))
    block = (t.chr.astype(str) + ":" + (t.pos // BLOCK).astype(str)).to_numpy()
    blocks = {b: np.flatnonzero(block == b) for b in np.unique(block)}
    names, rng = list(blocks), np.random.default_rng(SETTINGS["seed"])
    draws = [np.concatenate([blocks[b] for b in rng.choice(names, len(names))]) for _ in range(N_BOOT)]
    res, boot = {n: {} for n in fns}, {n: np.zeros(N_BOOT) for n in fns}
    for cell, tab in TABLE.items():
        a1, a2, skew = (t[f"{k}_{tab}"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
        ok = ~np.isnan(a1) & ~np.isnan(a2)
        e = ok & ~np.isnan(skew) & (np.maximum(a1, a2) >= MIN_ACTIVITY)
        for n in fns:
            p1, p2 = pred[n][cell]
            d = p2 - p1
            res[n][cell] = {"effect": float(np.corrcoef(skew[e], d[e])[0, 1]), "effect_variants": int(e.sum()),
                            "effect_spearman": float(spearmanr(skew[e], d[e])[0]),
                            "effect_no_shared_word": float(np.corrcoef(skew[e & fresh], d[e & fresh])[0, 1]),
                            "no_shared_word_variants": int((e & fresh).sum()),
                            "activity": float(np.corrcoef(np.concatenate([a1[ok], a2[ok]]), np.concatenate([p1[ok], p2[ok]]))[0, 1])}
            for j, dr in enumerate(draws):
                dr = dr[e[dr]]
                boot[n][j] += np.corrcoef(skew[dr], d[dr])[0, 1] / len(TABLE)
    for n in fns:
        for k in ("effect", "effect_spearman", "effect_no_shared_word", "activity"):
            res[n]["mean_" + k] = float(np.mean([res[n][c][k] for c in TABLE]))
        res[n]["mean_effect_ci95"] = [float(v) for v in np.percentile(boot[n], [2.5, 97.5])]
        res[n]["minus_model1_ci95"] = [float(v) for v in np.percentile(boot[n] - boot["student_v2"], [2.5, 97.5])]
        res[n]["minus_published_ci95"] = [float(v) for v in np.percentile(boot[n] - boot["published"], [2.5, 97.5])]
    # SK-N-SH: model 2 only (model 1 and the published model have no such output)
    sk = {}
    for name, run in (("model2", student["main"]), ("model2_seed2", student["seed2"])):
        outs = json.loads((Path(RUNS) / run / "log.json").read_text())["config"]["outputs"]
        m = build("legnet", len(outs)).to(device)
        m.load_state_dict(torch.load(Path(RUNS) / run / "best.pt", map_location=device, weights_only=True))
        k = outs.index("new_SKNSH")
        p1, p2 = (predict_multi(m, torch.from_numpy(x.reshape(-1, x.shape[-1])), device)[:, k].reshape(len(t), 2).mean(1) for x in (x1, x2))
        a1, a2, skew = (t[f"{q}_SKNSH"].to_numpy() for q in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
        e = ~np.isnan(a1) & ~np.isnan(a2) & ~np.isnan(skew) & (np.maximum(a1, a2) >= MIN_ACTIVITY)
        sk[name] = {"effect": float(np.corrcoef(skew[e], (p2 - p1)[e])[0, 1]), "effect_variants": int(e.sum())}
    return {"variants": len(t), "blocks": len(names), "models": res, "SKNSH": sk}


def verdict(res: dict, marks: dict) -> dict:
    m = res["models"]
    gain = m["model2"]["mean_effect"] - m["student_v2"]["mean_effect"]
    deficit = max(m["student_v2"][c]["effect"] - m["model2"][c]["effect"] for c in TABLE)
    gap = abs(m["model2"]["mean_effect"] - m["model2_seed2"]["mean_effect"])
    v = {"1_ci_above_zero": bool(gain > 0 and m["model2"]["minus_model1_ci95"][0] > 0), "2_min_gain": bool(gain >= marks["min_gain"]),
         "3_cell_deficit": bool(deficit <= marks["max_cell_deficit"]), "4_seed_gap": bool(gap <= marks["max_seed_gap"])}
    return {"gain": gain, "largest_cell_deficit": deficit, "seed_gap": gap, "marks": v, "pass": all(v.values())}


def main() -> None:
    text = PRE.read_text()
    marks = json.loads(re.search(r"<!-- marks: (.*?) -->", text).group(1))
    student = json.loads(re.search(r"<!-- students: (.*?) -->", text).group(1))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if "--practice" in sys.argv:
        student = student if student.get("fixed") else {"main": sys.argv[sys.argv.index("--main") + 1], "seed2": sys.argv[sys.argv.index("--seed2") + 1]}
        res, out = evaluate(MORE / "practice.parquet", student, device), Path("results/practice/locked_script_check.json")
    elif "--locked" in sys.argv:
        out = Path("results/locked/RESULT.json")
        if "**Status: CONFIRMED**" not in text or not student.get("fixed"):
            sys.exit("Stopped: the pre-declaration is not CONFIRMED or the students are not fixed.")
        if out.exists():
            sys.exit("Stopped: the locked test has already been run. It is run once.")
        res = evaluate(MORE / "locked.parquet", student, device)
    else:
        sys.exit(__doc__)
    res["student"], res["marks"] = student, marks
    res["verdict"] = verdict(res, marks)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for n, m in res["models"].items():
        print(f"{n:<18} effect {m['mean_effect']:.4f} " + " ".join(f"{c} {m[c]['effect']:.3f}" for c in TABLE)
              + f"  spearman {m['mean_effect_spearman']:.3f}  no shared word {m['mean_effect_no_shared_word']:.3f}  activity {m['mean_activity']:.3f}"
              + f"  minus model 1 {[round(v, 3) for v in m['minus_model1_ci95']]}")
    print("variants", res["variants"], {c: (res["models"]["model2"][c]["effect_variants"], res["models"]["model2"][c]["no_shared_word_variants"]) for c in TABLE}, "SK-N-SH", res["SKNSH"])
    print("verdict", json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()

"""Extra tests for model 2 (predeclarations/EXTRA_TESTS_MODEL2.md). Nothing is chosen from these results.

    python src/extra_tests_model2.py overlap              near-copy counts for tests 1 to 3 (sequences only)
    python src/extra_tests_model2.py fold --practice      fold 9 (check: model 2 0.844, model 1 0.842)
    python src/extra_tests_model2.py fold --locked        fold 10, third read. RUN ONCE.
    python src/extra_tests_model2.py scan --practice      scan practice half (check: model 2 0.699, model 1 0.627)
    python src/extra_tests_model2.py scan --locked        scan locked half, second read. RUN ONCE.
    python src/extra_tests_model2.py kircher              the four Kircher elements (check inside: model 1 0.639, published 0.594). RUN ONCE.

Results go to results/extra_tests/. Locked runs refuse if their result file exists.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from baseline_kmer import features  # noqa: E402
from evaluate_locked import words  # noqa: E402
from evaluate_practice import MORE, model2  # noqa: E402
from evaluate_scan import N_BOOT, SCAN_CELLS, V2_MAIN, V2_SEED2, backgrounds, models, scan, score  # noqa: E402
from evaluate_variants import MAIN as KIRCHER, effect, pieces as kpieces, table  # noqa: E402
from legnet_reference import MAIN_MODEL, WEIGHTS, load_model, predictor  # noqa: E402
from malinois_body import build_arch  # noqa: E402
from mpra_data import CELLS, TEST_FOLD, TRAIN_FOLDS, VALIDATION_FOLD, encode, load  # noqa: E402
from mpra_train import RUNS, SETTINGS, checkpoint  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402
from mpra_train_v2 import build  # noqa: E402

OUT = Path("results/extra_tests")
PRE = Path("predeclarations/EXTRA_TESTS_MODEL2.md")
M2 = {"model2": "more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed20261113",
      "model2_seed2": "more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed2"}
M1 = {"model1": V2_MAIN["run"], "model1_seed2": V2_SEED2["run"]}
NEAR = 27
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def ci(a) -> list:
    return [float(v) for v in np.percentile(a, [2.5, 97.5])]


def raw_model2(run: str):
    """codes -> (m, 8) predictions and the output names."""
    cfg = json.loads((Path(RUNS) / run / "log.json").read_text())["config"]
    m = build_arch(cfg.get("arch", "legnet"), len(cfg["outputs"]), width=float(cfg.get("width", 1.0))).to(DEV)
    m.load_state_dict(torch.load(Path(RUNS) / run / "best.pt", map_location=DEV, weights_only=True))
    return (lambda codes: predict_multi(m, torch.from_numpy(codes), DEV)), cfg["outputs"]


def raw_model1(run: str):
    m = build("legnet", len(CELLS)).to(DEV)
    m.load_state_dict(torch.load(checkpoint(run), map_location=DEV))
    return (lambda codes: predict_multi(m, torch.from_numpy(codes), DEV)), list(CELLS)


# ---------------------------------------------------------------- near-copy counter (as in rescore_nearcopy.py)
def near_copy_best(test_codes: np.ndarray) -> np.ndarray:
    """For each (n, 200) test piece: the largest number of its distinct canonical 24-letter words shared with ONE
    new-table training piece (seq1 or seq2 of data/more_data/train.parquet)."""
    tr = pd.read_parquet(MORE / "train.parquet", columns=["seq1", "seq2"])
    n = len(test_codes)
    hw = words(test_codes)
    key = np.unique(np.stack([hw.ravel(), np.repeat(np.arange(n, dtype=np.uint64), hw.shape[1])], 1), axis=0)
    W, P = key[:, 0], key[:, 1].astype(np.int64)
    best = np.zeros(n, np.int64)
    for col in ("seq1", "seq2"):
        codes = encode(tr[col])
        for a in range(0, len(codes), 2000):
            q = np.sort(words(codes[a:a + 2000]), axis=1)
            first = np.concatenate([np.ones((len(q), 1), bool), q[:, 1:] != q[:, :-1]], 1).ravel()
            rows, w = np.repeat(np.arange(len(q)), q.shape[1])[first], q.ravel()[first]
            lo, hi = np.searchsorted(W, w, "left"), np.searchsorted(W, w, "right")
            c = hi - lo
            use = c > 0
            r, l, c = rows[use], lo[use], c[use]
            if not len(r):
                continue
            idx = np.repeat(l, c) + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
            u, k = np.unique(np.repeat(r, c).astype(np.int64) * n + P[idx], return_counts=True)
            np.maximum.at(best, u % n, k)
    return best


def fold_near_copy_elements(fold: int) -> set:
    """Elements of a lentiMPRA fold (any cell type) whose 200-letter insert has a near-copy in model 2's new table."""
    seqs = {}
    for c in CELLS:
        d = load(c, [fold])
        for e, s in zip(d.element, d.seq):
            seqs.setdefault(e, s[15:215])
    names = list(seqs)
    best = near_copy_best(encode(pd.Series([seqs[e] for e in names])))
    return {e for e, b in zip(names, best) if b >= NEAR}, len(names), best


def overlap() -> None:
    res = {}
    near10, n10, b10 = fold_near_copy_elements(TEST_FOLD)
    res["fold10"] = {"elements": n10, "near_copy": len(near10), "share_any_word": int((b10 > 0).sum())}
    for split in ("practice", "locked"):  # sequences only; no labels are used
        s = scan(split)
        seq = backgrounds(s)
        ids = list(seq)
        best = near_copy_best(encode(pd.Series([seq[i] for i in ids])))
        res[f"scan_{split}"] = {"backgrounds": len(ids), "near_copy": int((best >= NEAR).sum()), "share_any_word": int((best > 0).sum()),
                                "near_copy_ids": [i for i, b in zip(ids, best) if b >= NEAR]}
    # Note: the table's "pos" column is hg19 (checked 2026-11-17: SORT1's near-copies are chr1:1098172xx-1098178xx),
    # so coordinates are not compared with the hg38 Kircher positions; training pieces are matched by words instead.
    tr = pd.read_parquet(MORE / "train.parquet", columns=["variant", "seq1", "seq2"])
    tw = {col: words(encode(tr[col])) for col in ("seq1", "seq2")}
    from data import genome
    for name in KIRCHER:
        d = table(name)
        ch, pos = str(d.Chromosome[0]), d.Position.to_numpy()
        g = genome(ch)
        lo, hi = int(pos.min()), int(pos.max())
        tiles = np.stack([np.array(g[p - 1 - 100: p - 1 + 100]) for p in np.unique(pos)])  # the offset-0 window of each position
        best = near_copy_best(tiles)
        span = np.unique(words(np.array(g[lo - 1 - 100: hi + 100]).reshape(1, -1)))  # element plus 100 letters each side
        k = np.maximum(*[np.array([len(np.intersect1d(r, span)) for r in tw[c]]) for c in tw])
        res[f"kircher_{name}"] = {"chrom": ch, "start_hg38": lo, "end_hg38": hi, "windows": len(tiles),
                                  "near_copy_windows": int((best >= NEAR).sum()), "max_shared_words": int(best.max()),
                                  "training_pieces_sharing_27plus_words_with_element": int((k >= NEAR).sum()),
                                  "training_pieces_sharing_any_word_with_element": int((k > 0).sum()),
                                  "those_variants": tr.variant[k > 0].tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlap.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "near_copy_ids"} for k, v in res.items()}, indent=2))


# ---------------------------------------------------------------- test 1: lentiMPRA whole pieces
def fold_test(fold: int) -> dict:
    near, _, _ = fold_near_copy_elements(fold)
    fns = {k: raw_model2(r)[0] for k, r in M2.items()}
    fns.update({k: raw_model1(r)[0] for k, r in M1.items()})
    rng = np.random.default_rng(SETTINGS["seed"])
    out = {"fold": fold, "cells": {}}
    boot = {k: [] for k in ("model2_minus_model1", "model2_minus_published")}
    for k, cell in enumerate(CELLS):
        te = load(cell, [fold])
        x = encode(te.seq)
        cols = {"y": te.mean_value.to_numpy(), **{n: f(x)[:, k] for n, f in fns.items()},  # column k = lentiMPRA output k
                "published": predictor([load_model(cell, WEIGHTS / cell / MAIN_MODEL, device=DEV)], DEV)(x)}
        tr = load(cell, TRAIN_FOLDS)
        cols["baseline"] = Ridge(alpha=100.0).fit(features(encode(tr.seq)), tr.mean_value.to_numpy()).predict(features(x))
        g = pd.DataFrame({"element": te.element.to_numpy(), **cols}).groupby("element").mean()
        r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
        d1, d2 = [], []
        y = g.y.values
        for _ in range(N_BOOT):
            i = rng.integers(0, len(g), len(g))
            a = r(y[i], g.model2.values[i])
            d1.append(a - r(y[i], g.model1.values[i]))
            d2.append(a - r(y[i], g.published.values[i]))
        boot["model2_minus_model1"].append(d1)
        boot["model2_minus_published"].append(d2)
        keep = ~g.index.isin(near)
        out["cells"][cell] = {"elements": len(g), "near_copy_elements": int((~keep).sum()),
                              **{c: r(g.y, g[c]) for c in cols if c != "y"},
                              "no_near_copy": {c: r(g.y[keep], g[c][keep]) for c in cols if c != "y"},
                              "model2_minus_model1_ci95": ci(d1), "model2_minus_published_ci95": ci(d2)}
    names = [c for c in out["cells"]["HepG2"] if c not in ("elements", "near_copy_elements", "no_near_copy") and not c.endswith("ci95")]
    out["mean"] = {n: float(np.mean([out["cells"][c][n] for c in CELLS])) for n in names}
    out["mean_no_near_copy"] = {n: float(np.mean([out["cells"][c]["no_near_copy"][n] for c in CELLS])) for n in names}
    for k, v in boot.items():
        out[f"mean_{k}_ci95"] = ci(np.mean(v, axis=0))
    return out


# ---------------------------------------------------------------- test 2: scan
def scan_test(split: str) -> dict:
    more = {}
    for k, run in M2.items():
        for ro, fn in model2(run, DEV).items():
            more[f"{k}_{ro[3:]}"] = fn  # model2_new, model2_lenti, model2_both, model2_seed2_new, ...
    res, d = score(split, {}, DEV, more=more)
    near = set(json.loads((OUT / "overlap.json").read_text())[f"scan_{split}"]["near_copy_ids"])
    overall = lambda t, col: float(np.mean([t[t.cell == c][col].mean() for c in SCAN_CELLS.values()]))
    rng = np.random.default_rng(SETTINGS["seed"])
    parents, by_parent = d.parent.unique(), {p: t for p, t in d.groupby("parent")}
    names = [n for n in res["models"]]
    diffs = {n: [] for n in names}
    for _ in range(N_BOOT):
        t = pd.concat([by_parent[p] for p in rng.choice(parents, len(parents))])
        for n in names:
            diffs[n].append(overall(t, n) - overall(t, "student_v2"))
    keep = ~d.sat_ref.isin(near)
    for n in names:
        res["models"][n]["minus_model1_ci95"] = ci(diffs[n])
        res["models"][n]["no_near_copy"] = overall(d[keep], n)
    res["near_copy_backgrounds_scored"] = int((~keep).sum())
    return res, d


# ---------------------------------------------------------------- test 3: Kircher
def kircher_test() -> dict:
    fns = models(DEV, {})  # student_v2 (model 1), student_v2_seed2, published, baseline (+ v1 if present)
    for k, run in M2.items():
        m = model2(run, DEV)
        fns[f"{k}_new"], fns[f"{k}_lenti"] = m["m2_new"], m["m2_lenti"]
    rng = np.random.default_rng(SETTINGS["seed"])
    out, boot = {"elements": {}}, {"model2_new_minus_model1": [], "model2_new_minus_published": []}
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    for name, cell in KIRCHER.items():
        d = table(name)
        ref, new = kpieces(d.Chromosome[0], d.Position.to_numpy(), d.alt_code.to_numpy())
        for n, fn in fns.items():
            d[n] = effect(lambda c, fn=fn: fn(c)[cell], ref, new)
        s = d[d.alt_code >= 0]
        y = s.Value.to_numpy()
        b1, b2 = [], []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            a = r(y[i], s.model2_new.values[i])
            b1.append(a - r(y[i], s.student_v2.values[i]))
            b2.append(a - r(y[i], s.published.values[i]))
        boot["model2_new_minus_model1"].append(b1)
        boot["model2_new_minus_published"].append(b2)
        out["elements"][name] = {"cell": cell, "changes": len(s), **{n: r(y, s[n]) for n in fns},
                                 "model2_new_minus_model1_ci95": ci(b1), "model2_new_minus_published_ci95": ci(b2)}
        d.drop(columns="alt_code").to_parquet(OUT / f"kircher_predictions_{name}.parquet", index=False)
    out["mean"] = {n: float(np.mean([v[n] for v in out["elements"].values()])) for n in fns}
    for k, v in boot.items():
        out[f"mean_{k}_ci95"] = ci(np.mean(v, axis=0))
    return out


def main() -> None:
    if "RECORDED before scoring" not in PRE.read_text():
        sys.exit("Stopped: the record is missing.")
    OUT.mkdir(parents=True, exist_ok=True)
    what, locked = sys.argv[1], "--locked" in sys.argv
    if what == "overlap":
        return overlap()
    if what == "fold":
        f = OUT / ("fold10.json" if locked else "fold9_check.json")
        if locked and f.exists():
            sys.exit("Stopped: fold 10 already scored.")
        res = fold_test(TEST_FOLD if locked else VALIDATION_FOLD)
    elif what == "scan":
        f = OUT / ("scan_locked.json" if locked else "scan_practice_check.json")
        if locked and f.exists():
            sys.exit("Stopped: scan locked half already scored.")
        res, d = scan_test("locked" if locked else "practice")
        d.to_parquet(f.with_suffix(".per_background.parquet"), index=False)
    elif what == "kircher":
        f = OUT / "kircher.json"
        if f.exists():
            sys.exit("Stopped: Kircher already scored.")
        res = kircher_test()
    else:
        sys.exit(__doc__)
    f.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

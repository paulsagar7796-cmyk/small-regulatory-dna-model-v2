"""Correction for the leak-guard flaw found on 2026-11-16 (the cap-8 shortcut in prepare_more_data.py let repeat-family
near-copies stay in training). Re-scores the FIXED models of the model 2 test on the variants that have no near-copy in
the new-table training pieces (fewer than 27 shared 24-letter words with any one training piece, counted exactly).
No model or setting is chosen from this. The maintainer approved one run on the locked chromosomes on 2026-11-16.

    python src/rescore_nearcopy.py --practice | --locked      (--locked refuses if its result file exists)
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "src/third_party")
import numpy as np, pandas as pd, torch
from evaluate_practice import BLOCK, MIN_ACTIVITY, MORE, TABLE, model2
from evaluate_scan import N_BOOT, models, pieces
from evaluate_locked import PRE, words
from mpra_data import encode
from mpra_train import SETTINGS

which = "locked" if "--locked" in sys.argv else "practice" if "--practice" in sys.argv else sys.exit(__doc__)
out = Path(f"results/{which}/NEARCOPY_RESCORE.json")
if which == "locked" and out.exists():
    sys.exit("Stopped: already run.")
t = pd.read_parquet(MORE / f"{which}.parquet")
tr = pd.read_parquet(MORE / "train.parquet", columns=["seq1", "seq2"])
n = len(t)
hw = words(encode(t.seq1))                                   # (n, 177) canonical words of each held-out piece (version 1)
key = np.unique(np.stack([hw.ravel(), np.repeat(np.arange(n, dtype=np.uint64), hw.shape[1])], 1), axis=0)  # a word counts once per piece
W, P = key[:, 0], key[:, 1].astype(np.int64)
best = np.zeros(n, np.int64)
for col in ("seq1", "seq2"):
    codes = encode(tr[col])
    for a in range(0, len(codes), 2000):
        q = np.sort(words(codes[a:a + 2000]), axis=1)
        first = np.concatenate([np.ones((len(q), 1), bool), q[:, 1:] != q[:, :-1]], 1).ravel()
        rows, w = np.repeat(np.arange(len(q)), q.shape[1])[first], q.ravel()[first]
        lo, hi = np.searchsorted(W, w, "left"), np.searchsorted(W, w, "right")
        c = hi - lo; use = c > 0; r, l, c = rows[use], lo[use], c[use]
        if not len(r):
            continue
        idx = np.repeat(l, c) + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
        u, k = np.unique(np.repeat(r, c).astype(np.int64) * n + P[idx], return_counts=True)
        np.maximum.at(best, u % n, k)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
student = json.loads(re.search(r"<!-- students: (.*?) -->", PRE.read_text()).group(1))
fns = {k: v for k, v in models(device, {}).items() if k in ("student_v2", "published", "baseline")}
fns["model2"], fns["model2_seed2"] = model2(student["main"], device)["m2_new"], model2(student["seed2"], device)["m2_new"]
x1, x2 = pieces(t.seq1.tolist()), pieces(t.seq2.tolist())
pred = {}
for m, fn in fns.items():
    a, b = (fn(x.reshape(-1, x.shape[-1])) for x in (x1, x2))
    pred[m] = {c: b[c].reshape(n, 2).mean(1) - a[c].reshape(n, 2).mean(1) for c in TABLE}
block = (t.chr.astype(str) + ":" + (t.pos // BLOCK).astype(str)).to_numpy()
blocks = {b: np.flatnonzero(block == b) for b in np.unique(block)}
names, rng = list(blocks), np.random.default_rng(SETTINGS["seed"])
draws = [np.concatenate([blocks[b] for b in rng.choice(names, len(names))]) for _ in range(N_BOOT)]
res = {"table": which, "variants": n, "near_copy_variants": int((best >= 27).sum()), "share_any_word": int((best > 0).sum()), "subsets": {}}
for name, s in {"all": best >= 0, "no_near_copy": best < 27, "near_copy": best >= 27}.items():
    r, boot = {}, {m: np.zeros(N_BOOT) for m in pred}
    for cell, tab in TABLE.items():
        a1, a2, skew = (t[f"{k}_{tab}"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
        e = s & ~np.isnan(a1) & ~np.isnan(a2) & ~np.isnan(skew) & (np.maximum(a1, a2) >= MIN_ACTIVITY)
        for m in pred:
            r.setdefault(m, {})[cell] = float(np.corrcoef(skew[e], pred[m][cell][e])[0, 1])
            for j, dr in enumerate(draws):
                dr = dr[e[dr]]
                boot[m][j] += np.corrcoef(skew[dr], pred[m][cell][dr])[0, 1] / len(TABLE)
        r.setdefault("scored", {})[cell] = int(e.sum())
    for m in pred:
        r[m]["mean"] = float(np.mean([r[m][c] for c in TABLE]))
        r[m]["model2_minus_this_ci95"] = [float(v) for v in np.percentile(boot["model2"] - boot[m], [2.5, 97.5])]
    res["subsets"][name] = r
    print(f"{name:<13} scored {r['scored']}  " + "  ".join(f"{m} {r[m]['mean']:.4f}" for m in pred)
          + f"  gain over model 1 {r['model2']['mean'] - r['student_v2']['mean']:.4f} CI {[round(v, 3) for v in r['student_v2']['model2_minus_this_ci95']]}")
print("variants", n, "near-copy", res["near_copy_variants"])
out.write_text(json.dumps(res, indent=2))

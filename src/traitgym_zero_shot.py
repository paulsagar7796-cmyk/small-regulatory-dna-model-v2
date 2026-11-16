"""Test 4 of predeclarations/EXTRA_TESTS_MODEL2.md: TraitGym Mendelian, zero-shot, models 1 and 2. RUN ONCE.

    python src/traitgym_zero_shot.py --check    centre-letter check and CADD reproduction only (no model scores)
    python src/traitgym_zero_shot.py            the scoring (refuses if results/extra_tests/traitgym.json exists)

Window: 200 hg38 letters with the variant at 0-based index `at` (primary at = 100), i.e. 1-based positions
pos-at .. pos-at+199. The window goes between the training adapters; reference and alternative are predicted in
both orientations (insert forward and reversed; predict_multi also averages each piece with its reverse complement).
Metric and bootstrap: copied from the lab repo's d1/evaluate.py (pooled AUPRC, stratified bootstrap with seed
20261111 and 2,000 draws, chromosome-weighted AUPRC).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from data import genome  # noqa: E402
from extra_tests_model2 import M1, M2, OUT, raw_model1, raw_model2  # noqa: E402

RAW = Path("data/raw")
N_BOOT, SEED = 2000, 20261111  # as d1/evaluate.py
OFFSETS = (60, 80, 100, 120, 140)
LEFT, RIGHT = "AGGACCGGATCAACT", "CATTGCGTGAACCGA"
_L = np.array(["ACGT".index(b) for b in LEFT], np.uint8)
_R = np.array(["ACGT".index(b) for b in RIGHT], np.uint8)


# --- copied from d1/evaluate.py ---
def chrom_weighted(y, s, chrom):
    total, weight_sum = 0.0, 0
    for c in np.unique(chrom):
        m = chrom == c
        if y[m].sum() == 0:
            continue
        total += m.sum() * average_precision_score(y[m], s[m])
        weight_sum += m.sum()
    return total / weight_sum


def stratified_indices(y, rng):
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    return np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])


def block_indices(chrom, rng):
    blocks = np.unique(chrom)
    picked = rng.choice(blocks, blocks.size)
    return np.concatenate([np.flatnonzero(chrom == b) for b in picked])
# --- end of copy ---


def windows(v: pd.DataFrame):
    """(n, offsets, 2 alleles, 2 orientations, 230) codes, and the centre-letter check."""
    n = len(v)
    x = np.zeros((n, len(OFFSETS), 2, 2, 230), np.uint8)
    match = np.zeros(n, bool)
    for i, r in enumerate(v.itertuples()):
        g = genome(str(r.chrom))
        p0 = int(r.pos) - 1
        match[i] = "ACGTN"[g[p0]] == r.ref
        for j, at in enumerate(OFFSETS):
            a = np.array(g[p0 - at: p0 - at + 200])
            b = a.copy()
            b[at] = "ACGT".index(r.alt)
            for k, s in enumerate((a, b)):
                x[i, j, k, 0] = np.concatenate([_L, s, _R])
                x[i, j, k, 1] = np.concatenate([_L, np.where(s < 4, 3 - s, 4)[::-1], _R])
    return x, match


def main() -> None:
    v = pd.read_parquet(RAW / "traitgym_test.parquet")
    y, chrom = v.label.to_numpy(bool), v.chrom.to_numpy()
    x, match = windows(v)
    print(f"{len(v)} variants, {y.sum()} positives; centre letter equals reference for {match.sum()} of {len(v)}", flush=True)
    if match.sum() < len(v):
        sys.exit("Stopped: centre letters do not all match the reference letters.")
    has_n = int((x[:, OFFSETS.index(100)] == 4).any(axis=(1, 2, 3)).sum())
    print("variants whose primary window holds an N:", has_n)
    cadd = pd.read_parquet(RAW / "features/CADD.parquet").RawScore.astype(float)
    cadd = cadd.fillna(cadd.mean()).to_numpy()
    scores = {"CADD (check)": cadd}
    if "--check" not in sys.argv:
        out_file = OUT / "traitgym.json"
        if out_file.exists():
            sys.exit("Stopped: already scored.")
        flat = x.reshape(-1, 230)
        for name, (fn, outs), keep in [
                ("model1", raw_model1(M1["model1"]), ["HepG2", "K562"]),
                ("model1_seed2", raw_model1(M1["model1_seed2"]), ["HepG2", "K562"]),
                ("model2", raw_model2(M2["model2"]), ["new_HEPG2", "new_K562"]),
                ("model2_seed2", raw_model2(M2["model2_seed2"]), ["new_HEPG2", "new_K562"])]:
            p = fn(flat).reshape(len(v), len(OFFSETS), 2, 2, -1).mean(axis=3)  # orientations averaged
            ch = p[:, :, 1] - p[:, :, 0]  # (n, offsets, outputs): alt minus ref
            ks = [outs.index(o) for o in keep]
            c100 = ch[:, OFFSETS.index(100)]
            scores[f"{name} PRIMARY"] = np.abs(c100[:, ks]).mean(1)
            scores[f"{name} secondary: all outputs mean"] = np.abs(c100).mean(1)
            scores[f"{name} secondary: max over outputs"] = np.abs(c100).max(1)
            scores[f"{name} secondary: 5 offsets"] = np.abs(ch[:, :, ks]).mean(2).mean(1)
    rng = np.random.default_rng(SEED)
    strat = [stratified_indices(y, rng) for _ in range(N_BOOT)]
    boot = {n: np.array([average_precision_score(y[i], s[i]) for i in strat]) for n, s in scores.items()}
    res = {"variants": len(v), "positives": int(y.sum()), "random_auprc": float(y.mean()), "centre_letter_matches": int(match.sum()),
           "primary_window_with_N": has_n, "models": {}}
    for n, s in scores.items():
        res["models"][n] = {"pooled_auprc": float(average_precision_score(y, s)),
                            "pooled_ci95": [float(q) for q in np.percentile(boot[n], [2.5, 97.5])],
                            "chrom_weighted_auprc": float(chrom_weighted(y, s, chrom))}
        print(f"{n:<45} pooled {res['models'][n]['pooled_auprc']:.4f} CI {[round(q, 3) for q in res['models'][n]['pooled_ci95']]}"
              f"  chrom-weighted {res['models'][n]['chrom_weighted_auprc']:.4f}")
    if "model2 PRIMARY" in scores:
        d = boot["model2 PRIMARY"] - boot["model1 PRIMARY"]
        res["model2_minus_model1_primary_ci95"] = [float(q) for q in np.percentile(d, [2.5, 97.5])]
        print("model 2 minus model 1 (primary), CI", res["model2_minus_model1_primary_ci95"])
        out_file.write_text(json.dumps(res, indent=2))
    else:
        (OUT / "traitgym_check.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

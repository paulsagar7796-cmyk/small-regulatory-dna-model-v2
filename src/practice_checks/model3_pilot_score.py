"""Score model 3 pilots on practice chromosome 13 exactly as the AlphaGenome teacher check did
(src/practice_checks/alphagenome_teacher_score.py): effect Pearson, active variants (max allele log2FC >= 1), mean of HepG2 and
K562, 1,000 resamples of 100,000-letter blocks. Compared with model 2 (two taught students, z-averaged as in the check), MPAC
(practice-fold networks, member mean) and AlphaGenome-FT. Practice data only.

    .venv/bin/python src/practice_checks/model3_pilot_score.py runs/m3_P0_... runs/m3_P1_... ..."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from evaluate_scan import pieces  # noqa: E402
from malinois_body import build_arch  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402

TABLE = {"HepG2": "HEPG2", "K562": "K562"}
M2 = ["runs/more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed20261113",
      "runs/more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed2"]


def f32(a):
    a = np.asarray(a)
    if a.dtype.kind == "V" and a.dtype.itemsize == 2:
        return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def main():
    full = pd.read_parquet("data/more_data/practice.parquet")
    sel = np.flatnonzero((full.chr.astype(str) == "13").to_numpy())
    pr = full.iloc[sel].reset_index(drop=True)
    p13 = pd.read_parquet("data/model3/practice13.parquet")
    assert len(p13) == len(pr) and (p13.seq1.to_numpy() == pr.seq1.to_numpy()).all(), "practice13 table differs from chr13 of practice.parquet"
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x1, x2 = (torch.from_numpy(pieces(s.tolist()).reshape(-1, 230)) for s in (pr.seq1, pr.seq2))

    def student(run):
        cfg = json.loads((Path(run) / "log.json").read_text())["config"]
        outs = cfg["outputs"]
        m = build_arch(cfg.get("arch", "legnet"), len(outs), width=cfg.get("width", 1.0))
        m = m.to(dev)
        m.load_state_dict(torch.load(Path(run) / "best.pt", map_location=dev, weights_only=True))
        p1, p2 = predict_multi(m, x1, dev), predict_multi(m, x2, dev)
        return {c: (p2[:, outs.index(f"new_{t}")] - p1[:, outs.index(f"new_{t}")]).reshape(len(pr), 2).mean(1) for c, t in TABLE.items()}

    z = lambda v: (v - v.mean()) / v.std()
    mp = np.load("results/practice/mpac_single_practice_preds.npy")
    ag = {k: f32(v) for k, v in np.load("results/practice/alphagenome_chr13_flanks.npz").items()}
    m2 = [student(r) for r in M2]
    eff = {"MPAC": {c: (mp[:, i, sel, 1] - mp[:, i, sel, 0]).mean(0) for i, c in enumerate(TABLE)},
           "AG": {c: ag[f"{c}_v2"] - ag[f"{c}_v1"] for c in TABLE},
           "model2": {c: (z(m2[0][c]) + z(m2[1][c])) / 2 for c in TABLE},
           "model2_seed1": m2[0]}
    for run in sys.argv[1:]:
        eff[Path(run).name] = student(run)
    masks, skews = {}, {}
    for c, t in TABLE.items():
        a1, a2, sk = (pr[f"{k}_{t}"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
        masks[c] = ~np.isnan(a1) & ~np.isnan(a2) & ~np.isnan(sk) & (np.maximum(a1, a2) >= 1.0)
        skews[c] = sk
    block = (pr.chr.astype(str) + ":" + (pr.pos // 100_000).astype(str)).to_numpy()
    ub = np.unique(block)
    bidx = {b: np.flatnonzero(block == b) for b in ub}
    rng = np.random.default_rng(20261113)
    draws = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(1000)]

    def sc(e, d=None):
        r = []
        for c in TABLE:
            i = np.flatnonzero(masks[c]) if d is None else d[masks[c][d]]
            r.append(np.corrcoef(skews[c][i], e[c][i])[0, 1])
        return float(np.mean(r)), [round(float(v), 4) for v in r]

    res = {}
    for n, e in eff.items():
        res[n] = {"mean": round(sc(e)[0], 4), "HepG2_K562": sc(e)[1]}
        if n.startswith("m3_"):
            for ref in ("MPAC", "model2_seed1", "AG"):
                dd = [sc(e, d)[0] - sc(eff[ref], d)[0] for d in draws]
                res[n][f"minus_{ref}"] = round(sc(e)[0] - sc(eff[ref])[0], 4)
                res[n][f"minus_{ref}_ci95"] = [round(float(v), 4) for v in np.percentile(dd, [2.5, 97.5])]
    res["active_variants"] = {c: int(masks[c].sum()) for c in TABLE}
    print(json.dumps(res, indent=1))
    Path("results/practice/model3_pilots_chr13.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""Step 2 of the AlphaGenome teacher check (predeclarations/TEACHER_CHECK_ALPHAGENOME.md). Main environment, repo root:

    .venv/bin/python src/practice_checks/alphagenome_teacher_score.py <alphagenome npz> <mpac_single_practice_preds.npy>

The MPAC file is the output of src/practice_checks/mpac_single_practice.py (member, cell [HepG2, K562], variant, version)
over all practice variants in the order of data/more_data/practice.parquet."""
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
STUDENTS = ["more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed20261113",
            "more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed2"]


def main():
    raw, mp = np.load(sys.argv[1]), np.load(sys.argv[2])

    def f32(a):  # AlphaGenome runs in bfloat16; numpy stores it as raw 2-byte values ('V2')
        a = np.asarray(a)
        if a.dtype.kind == "V" and a.dtype.itemsize == 2:
            return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return a.astype(np.float32)
    ag = {k: f32(raw[k]) for k in raw.files}
    full = pd.read_parquet("data/more_data/practice.parquet")
    sel = np.flatnonzero((full.chr.astype(str) == "13").to_numpy())
    pr = full.iloc[sel].reset_index(drop=True)
    assert len(pr) == len(ag["HepG2_v1"])
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x1, x2 = (torch.from_numpy(pieces(s.tolist()).reshape(-1, 230)) for s in (pr.seq1, pr.seq2))
    ours = []
    for run in STUDENTS:
        outs = json.loads((Path("runs") / run / "log.json").read_text())["config"]["outputs"]
        m = build_arch("legnet", len(outs)).to(dev)
        m.load_state_dict(torch.load(Path("runs") / run / "best.pt", map_location=dev, weights_only=True))
        p1, p2 = predict_multi(m, x1, dev), predict_multi(m, x2, dev)
        ours.append({c: (p2[:, outs.index(f"new_{t}")] - p1[:, outs.index(f"new_{t}")]).reshape(len(pr), 2).mean(1)
                     for c, t in TABLE.items()})
    z = lambda v: (v - v.mean()) / v.std()
    eff = {"OURS": {c: (z(ours[0][c]) + z(ours[1][c])) / 2 for c in TABLE},
           "MPAC": {c: (mp[:, i, sel, 1] - mp[:, i, sel, 0]).mean(0) for i, c in enumerate(TABLE)},
           "AG": {c: ag[f"{c}_v2"] - ag[f"{c}_v1"] for c in TABLE}}
    eff["OURS+MPAC"] = {c: (z(eff["OURS"][c]) + z(eff["MPAC"][c])) / 2 for c in TABLE}
    eff["OURS+MPAC+AG"] = {c: (z(eff["OURS"][c]) + z(eff["MPAC"][c]) + z(eff["AG"][c])) / 3 for c in TABLE}
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

    res = {n: {"mean": round(sc(e)[0], 4), "HepG2_K562": sc(e)[1]} for n, e in eff.items()}
    diff = lambda a, b: [round(float(v), 4) for v in np.percentile([sc(eff[a], d)[0] - sc(eff[b], d)[0] for d in draws], [2.5, 97.5])]
    gain = res["OURS+MPAC+AG"]["mean"] - res["OURS+MPAC"]["mean"]
    ci = diff("OURS+MPAC+AG", "OURS+MPAC")
    res["rule"] = {"gain": round(gain, 4), "gain_ci95": ci, "1_ci_above_zero": ci[0] > 0, "2_gain_at_least_0.005": gain >= 0.005,
                   "qualifies": bool(ci[0] > 0 and gain >= 0.005)}
    res["AG_minus_MPAC_ci95"] = diff("AG", "MPAC")
    res["active_variants"] = {c: int(masks[c].sum()) for c in TABLE}
    res["activity_sanity"] = {c: float(ag[f"{c}_activity"]) for c in TABLE}
    print(json.dumps(res, indent=1))
    Path("results/practice/alphagenome_teacher_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

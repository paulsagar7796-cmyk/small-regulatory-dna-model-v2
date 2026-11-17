"""Test-time shift averaging on PRACTICE chromosomes only (6, 13, 20). Run from the v2 repo root.
For each model: predictions averaged over a set of slides (the training slide: piece moves left s letters, the
construct's right flank comes into view), both insert orientations and both strands, as in evaluate_practice.
Effect Pearson per cell (HepG2, K562) on variants with max allele log2FC >= 1; block-bootstrap CI of the
difference against the same model with no slide."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src"); sys.path.insert(0, "src/third_party")
from evaluate_scan import pieces
from mpra_train_multi import predict_multi
from mpra_train_v2 import RIGHT_FLANK
from mpra_data import encode
from malinois_body import build_arch

dev = torch.device("mps")
pr = pd.read_parquet("data/more_data/practice.parquet")
x1, x2 = (torch.from_numpy(pieces(s.tolist()).reshape(-1, 230)) for s in (pr.seq1, pr.seq2))
flank = torch.from_numpy(encode(pd.Series([RIGHT_FLANK]))[0])
def slide(c, s):
    if s == 0: return c
    ext = torch.cat([c, flank[:s].expand(len(c), -1)], 1)
    return ext[:, s:s + 230].contiguous()
TABLE = {"HepG2": "HEPG2", "K562": "K562"}
block = (pr.chr + ":" + (pr.pos // 100_000).astype(str)).to_numpy()
ub = np.unique(block); bidx = {b: np.flatnonzero(block == b) for b in ub}
rng = np.random.default_rng(20261113)
draws = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(1000)]
masks, skews = {}, {}
for c, t in TABLE.items():
    a1, a2, sk = (pr[f"{k}_{t}"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
    masks[c] = ~np.isnan(a1) & ~np.isnan(a2) & ~np.isnan(sk) & (np.maximum(a1, a2) >= 1.0); skews[c] = sk

SETS = {"none": [0], "4 slides": [0, 7, 14, 21], "8 slides": [0, 3, 6, 9, 12, 15, 18, 21]}
RUNS = sys.argv[1:]
out = {}
for run in RUNS:
    cfg = json.loads((Path("runs") / run / "log.json").read_text())["config"]; outs = cfg["outputs"]
    m = build_arch(cfg.get("arch", "legnet"), len(outs), width=float(cfg.get("width", 1.0))).to(dev)
    m.load_state_dict(torch.load(Path("runs") / run / "best.pt", map_location=dev, weights_only=True))
    ks = {c: outs.index(f"new_{t}") for c, t in TABLE.items()}
    per_shift = {}
    for s in sorted({s for v in SETS.values() for s in v}):
        p1, p2 = (predict_multi(m, slide(x, s), dev) for x in (x1, x2))
        per_shift[s] = {c: (p2[:, k] - p1[:, k]).reshape(len(pr), 2).mean(1) for c, k in ks.items()}
    eff = {n: {c: np.mean([per_shift[s][c] for s in v], 0) for c in TABLE} for n, v in SETS.items()}
    def score(e, idx=None):
        r = []
        for c in TABLE:
            i = np.flatnonzero(masks[c]) if idx is None else idx[masks[c][idx]]
            r.append(np.corrcoef(skews[c][i], e[c][i])[0, 1])
        return np.mean(r), r
    res = {}
    for n in SETS:
        mean, per = score(eff[n])
        diff = [score(eff[n], d)[0] - score(eff["none"], d)[0] for d in draws] if n != "none" else [0]
        res[n] = {"mean": round(mean, 4), "HepG2": round(per[0], 4), "K562": round(per[1], 4),
                  "minus_none_ci95": [round(float(v), 4) for v in np.percentile(diff, [2.5, 97.5])]}
        print(run[-45:], n, res[n], flush=True)
    out[run] = res
Path("results_tta_shift_practice.json").write_text(json.dumps(out, indent=2))

"""Practice only: model 2 (taught) averaged with MPAC (10-network mean), effects z-scored per cell before averaging."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src"); sys.path.insert(0, "src/third_party")
from evaluate_scan import pieces
from mpra_train_multi import predict_multi
from malinois_body import build_arch
S = Path(sys.argv[1]); dev = torch.device("mps")
pr = pd.read_parquet("data/more_data/practice.parquet")
mp = np.load(S / "mpac_single_practice_preds.npy")  # member, cell(cols order), n, version
cfgd = json.loads(Path("data/models/MPAC/config.json").read_text())["output_names"]
TABLE = {"HepG2": "HEPG2", "K562": "K562"}
order = {"HepG2": 0, "K562": 1}  # axis order as saved by mpac_single_practice.py (CELL_INDEX order)
mpac_eff = {c: (mp[:, order[c], :, 1] - mp[:, order[c], :, 0]).mean(0) for c in TABLE}
eff = {}
for name, run in [("taught", "more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed20261113"),
                  ("taught_s2", "more_HEPG2-K562-SKNSH-HCT116-A549_distill_t3_lam1.0_mut0.05_e25_seed2")]:
    cfg = json.loads((Path("runs") / run / "log.json").read_text())["config"]; outs = cfg["outputs"]
    m = build_arch("legnet", len(outs)).to(dev)
    m.load_state_dict(torch.load(Path("runs") / run / "best.pt", map_location=dev, weights_only=True))
    p1, p2 = (predict_multi(m, torch.from_numpy(pieces(s.tolist()).reshape(-1, 230)), dev) for s in (pr.seq1, pr.seq2))
    eff[name] = {c: (p2[:, outs.index(f"new_{t}")] - p1[:, outs.index(f"new_{t}")]).reshape(len(pr), 2).mean(1) for c, t in TABLE.items()}
masks, skews = {}, {}
for c, t in TABLE.items():
    a1, a2, sk = (pr[f"{k}_{t}"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
    masks[c] = ~np.isnan(a1) & ~np.isnan(a2) & ~np.isnan(sk) & (np.maximum(a1, a2) >= 1.0); skews[c] = sk
z = lambda v: (v - v.mean()) / v.std()
block = (pr.chr + ":" + (pr.pos // 100_000).astype(str)).to_numpy(); ub = np.unique(block); bidx = {b: np.flatnonzero(block == b) for b in ub}
rng = np.random.default_rng(20261113); draws = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(1000)]
def sc(e, d=None):
    return np.mean([np.corrcoef(skews[c][(i := (np.flatnonzero(masks[c]) if d is None else d[masks[c][d]]))], e[c][i])[0, 1] for c in TABLE])
combos = {"mpac": mpac_eff, "model2": eff["taught"],
          "model2 + mpac": {c: z(eff["taught"][c]) + z(mpac_eff[c]) for c in TABLE},
          "model2 x2 seeds + mpac": {c: (z(eff["taught"][c]) + z(eff["taught_s2"][c])) / 2 + z(mpac_eff[c]) for c in TABLE}}
for n, e in combos.items():
    ci = np.percentile([sc(e, d) - sc(mpac_eff, d) for d in draws], [2.5, 97.5]) if n != "mpac" else [0, 0]
    print(f"{n:<24} {sc(e):.4f}  minus MPAC CI [{ci[0]:.4f}, {ci[1]:.4f}]")

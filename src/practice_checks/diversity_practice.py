"""Practice only: how similar are ensemble members' predicted effects? MPAC's 10 networks vs our 3 untaught seeds."""
import json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src"); sys.path.insert(0, "src/third_party")
from evaluate_scan import pieces
from mpra_train_multi import predict_multi
from malinois_body import build_arch
S = Path(sys.argv[1]); dev = torch.device("mps")
pr = pd.read_parquet("data/more_data/practice.parquet")
mp = np.load(S / "mpac_single_practice_preds.npy")
TABLE = {"HepG2": "HEPG2", "K562": "K562"}
masks, skews = {}, {}
for c, t in TABLE.items():
    a1, a2, sk = (pr[f"{k}_{t}"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
    masks[c] = ~np.isnan(a1) & ~np.isnan(a2) & ~np.isnan(sk) & (np.maximum(a1, a2) >= 1.0); skews[c] = sk
mpac = [{c: mp[j, i, :, 1] - mp[j, i, :, 0] for i, c in enumerate(TABLE)} for j in range(10)]
ours = []
x1, x2 = (torch.from_numpy(pieces(s.tolist()).reshape(-1, 230)) for s in (pr.seq1, pr.seq2))
for seed in ("20261113", "2", "3"):
    run = f"more_HEPG2-K562-SKNSH-HCT116-A549_e25_seed{seed}"
    outs = json.loads((Path("runs") / run / "log.json").read_text())["config"]["outputs"]
    m = build_arch("legnet", len(outs)).to(dev)
    m.load_state_dict(torch.load(Path("runs") / run / "best.pt", map_location=dev, weights_only=True))
    p1, p2 = predict_multi(m, x1, dev), predict_multi(m, x2, dev)
    ours.append({c: (p2[:, outs.index(f"new_{t}")] - p1[:, outs.index(f"new_{t}")]).reshape(len(pr), 2).mean(1) for c, t in TABLE.items()})
def pair_r(ms):
    return np.mean([np.mean([np.corrcoef(a[c][masks[c]], b[c][masks[c]])[0, 1] for c in TABLE]) for a, b in itertools.combinations(ms, 2)])
def acc(ms):
    return np.mean([np.corrcoef(skews[c][masks[c]], np.mean([m[c] for m in ms], 0)[masks[c]])[0, 1] for c in TABLE])
print(f"MPAC  members: mean pairwise r of predicted effects {pair_r(mpac):.3f}; single {np.mean([acc([m]) for m in mpac]):.4f}; first 3 avg {acc(mpac[:3]):.4f}; all 10 {acc(mpac):.4f}")
print(f"ours  seeds  : mean pairwise r of predicted effects {pair_r(ours):.3f}; single {np.mean([acc([m]) for m in ours]):.4f}; 3 avg {acc(ours):.4f}")

"""E1: how much of MPAC's practice score comes from averaging 10 networks? PRACTICE chromosomes 6, 13, 20 only.
For each chromosome, its fold's 10 clean networks are run one at a time (same one centred window as our reference run).
Reports: each single network's effect Pearson (mean of HepG2, K562), the mean of singles, and averages of the first k
networks for k = 1, 2, 3, 5, 10. Run from the v2 repo root."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src"); sys.path.insert(0, "src/third_party")
import mpac_reference as R
from safetensors.torch import load_file

dev = R.pick_device()
pr = pd.read_parquet("data/more_data/practice.parquet")
config = json.loads((R.MPAC_DIR / "config.json").read_text())
cols = {c: config["output_names"].index(n) for c, n in R.CELL_INDEX.items()}
TABLE = {"HepG2": "HEPG2", "K562": "K562"}
preds = np.full((10, 2, len(pr), 2), np.nan)  # member, cell, variant, version
prov = R.provenance()
for chrom in sorted(pr.chr.astype(str).unique()):
    _, members = R.verify_fold(chrom, prov)
    idx = np.flatnonzero((pr.chr.astype(str) == chrom).to_numpy())
    for j, r in enumerate(members):
        net = R.mm.MPACModel(**config); net.load_state_dict(load_file(str(R.MPAC_DIR / r["file"])))
        ens = R.mm.MPACEnsemble([net.eval()]).to(dev)
        for v, col in enumerate(("seq1", "seq2")):
            p = ens.predict(pr[col].to_numpy()[idx].tolist(), batch_size=256, device=dev).numpy()
            for ci, (c, k) in enumerate(cols.items()):
                preds[j, ci, idx, v] = p[:, k]
        print("chr", chrom, "member", j, flush=True)
np.save("mpac_single_practice_preds.npy", preds)
masks, skews = [], []
for c, t in TABLE.items():
    a1, a2, sk = (pr[f"{k}_{t}"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
    masks.append(~np.isnan(a1) & ~np.isnan(a2) & ~np.isnan(sk) & (np.maximum(a1, a2) >= 1.0)); skews.append(sk)
order = [list(cols).index(c) for c in TABLE]
def score(p):  # p: (2 cells, n, 2 versions) in cols order
    r = [np.corrcoef(skews[i][masks[i]], (p[order[i], :, 1] - p[order[i], :, 0])[masks[i]])[0, 1] for i in range(2)]
    return round(float(np.mean(r)), 4), [round(float(x), 4) for x in r]
res = {"singles": [score(preds[j]) for j in range(10)]}
res["mean_of_singles"] = round(float(np.mean([s[0] for s in res["singles"]])), 4)
res["first_k_average"] = {k: score(preds[:k].mean(0)) for k in (1, 2, 3, 5, 10)}
print(json.dumps(res, indent=1))
Path("mpac_single_practice.json").write_text(json.dumps(res, indent=1))

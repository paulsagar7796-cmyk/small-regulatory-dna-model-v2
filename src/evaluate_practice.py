"""Model 2 (more data): score a trained run on PRACTICE data only, beside model 1 and the published model.

    .venv/bin/python src/evaluate_practice.py --run <run folder name> [--tag T]
    .venv/bin/python src/evaluate_practice.py --run none     (model 1 and the published model only, measure B)

Two practice measures, both for single-letter changes:
  A. the saturation scan's practice half (odd chromosomes and X), exactly as in model 1's tests
     (src/evaluate_scan.py; the scan's locked half is closed and is not loaded);
  B. the natural variants on the practice chromosomes (6, 13, 20): Pearson between the measured effect
     (log2Skew = version 2 minus version 1) and the predicted one, over variants where at least one version is
     active (log2FC >= 1), in HepG2 and K562; and the activity Pearson over both versions.
Model 2 is read out three ways: its lentiMPRA outputs ("m2_lenti"), its new plasmid-MPRA outputs ("m2_new"),
and the two averaged ("m2_both"). Which read-out to use is a choice to make on practice data.
The locked chromosomes (5, 14, 22), the scan's locked half and lentiMPRA fold 10 are never loaded.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from evaluate_scan import N_BOOT, SCAN_CELLS, models, pieces, score  # noqa: E402
from mpra_data import CELLS  # noqa: E402
from mpra_train import RUNS, SETTINGS  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402
from malinois_body import build_arch  # noqa: E402

OUT, MORE = Path("results/practice"), Path("data/more_data")
TABLE = {"HepG2": "HEPG2", "K562": "K562"}  # our cell names -> the variant table's
MIN_ACTIVITY, BLOCK = 1.0, 100_000


def model2(run: str, device) -> dict:
    cfg = json.loads((Path(RUNS) / run / "log.json").read_text())["config"]
    outs = cfg["outputs"]
    m = build_arch(cfg.get("arch", "legnet"), len(outs), width=float(cfg.get("width", 1.0))).to(device)  # runs before --arch existed: legnet
    m.load_state_dict(torch.load(Path(RUNS) / run / "best.pt", map_location=device, weights_only=True))

    def reader(index: dict):
        def fn(codes):
            p = predict_multi(m, torch.from_numpy(codes), device)
            return {c: np.mean([p[:, k] for k in ks], axis=0) for c, ks in index.items()}
        return fn

    lenti = {c: [CELLS.index(c)] for c in SCAN_CELLS.values()}
    fns = {"m2_lenti": reader(lenti)}
    if all(f"new_{t}" in outs for t in TABLE.values()):
        new = {c: [outs.index(f"new_{TABLE[c]}")] for c in SCAN_CELLS.values()}
        fns["m2_new"] = reader(new)
        fns["m2_both"] = reader({c: lenti[c] + new[c] for c in lenti})  # outputs are in standard units, so a plain mean
    return fns


def natural_variants(fns: dict) -> dict:
    pr = pd.read_parquet(MORE / "practice.parquet")
    x1, x2 = pieces(pr.seq1.tolist()), pieces(pr.seq2.tolist())
    block = (pr.chr + ":" + (pr.pos // BLOCK).astype(str)).to_numpy()
    blocks = {b: np.flatnonzero(block == b) for b in np.unique(block)}
    names = list(blocks)
    pred = {}
    for name, fn in fns.items():
        a, b = (fn(x.reshape(-1, x.shape[-1])) for x in (x1, x2))
        pred[name] = {c: (a[c].reshape(len(pr), 2).mean(1), b[c].reshape(len(pr), 2).mean(1)) for c in TABLE}
    res, rng = {n: {} for n in fns}, np.random.default_rng(SETTINGS["seed"])
    draws = [np.concatenate([blocks[b] for b in rng.choice(names, len(names))]) for _ in range(N_BOOT)]
    boot = {n: np.zeros(N_BOOT) for n in fns}
    for cell, tab in TABLE.items():
        a1, a2, skew = (pr[f"{k}_{tab}"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
        ok = ~np.isnan(a1) & ~np.isnan(a2)
        e = ok & ~np.isnan(skew) & (np.maximum(a1, a2) >= MIN_ACTIVITY)
        for n in fns:
            p1, p2 = pred[n][cell]
            res[n][cell] = {"activity": float(np.corrcoef(np.concatenate([a1[ok], a2[ok]]), np.concatenate([p1[ok], p2[ok]]))[0, 1]),
                            "effect": float(np.corrcoef(skew[e], (p2 - p1)[e])[0, 1]), "effect_variants": int(e.sum())}
            for j, d in enumerate(draws):
                d = d[e[d]]
                boot[n][j] += np.corrcoef(skew[d], (p2 - p1)[d])[0, 1] / len(TABLE)
    for n in fns:
        res[n]["mean_effect"] = float(np.mean([res[n][c]["effect"] for c in TABLE]))
        res[n]["mean_activity"] = float(np.mean([res[n][c]["activity"] for c in TABLE]))
        res[n]["mean_effect_minus_model1_ci95"] = [float(v) for v in np.percentile(boot[n] - boot["student_v2"], [2.5, 97.5])]
    return res


def main() -> None:
    run = sys.argv[sys.argv.index("--run") + 1]
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else run
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    if run == "none":  # reference numbers only: model 1, the published model and the baseline on measure B
        out = {"run": None, "natural_variants_practice_chromosomes": natural_variants(models(device, {}))}
        (OUT / "references.json").write_text(json.dumps(out, indent=2))
        for n, m in out["natural_variants_practice_chromosomes"].items():
            print(f"natural variants     {n:<18} effect {m['mean_effect']:.4f}  activity {m['mean_activity']:.4f}",
                  {c: (round(m[c]["effect"], 3), m[c]["effect_variants"]) for c in TABLE})
        return
    m2 = model2(run, device)
    res, d = score("practice", {}, device, more=m2)  # measure A
    rng = np.random.default_rng(SETTINGS["seed"])
    parents, by_parent = d.parent.unique(), {p: t for p, t in d.groupby("parent")}
    overall = lambda t, col: float(np.mean([t[t.cell == c][col].mean() for c in SCAN_CELLS.values()]))
    diffs = {n: [] for n in m2}
    for _ in range(N_BOOT):
        t = pd.concat([by_parent[p] for p in rng.choice(parents, len(parents))])
        for n in m2:
            diffs[n].append(overall(t, n) - overall(t, "student_v2"))
    for n in m2:
        res["models"][n]["minus_model1_ci95"] = [float(v) for v in np.percentile(diffs[n], [2.5, 97.5])]
    fns = models(device, {})
    fns.update(m2)
    out = {"run": run, "scan_practice_half": res, "natural_variants_practice_chromosomes": natural_variants(fns)}
    d.to_parquet(OUT / f"{tag}_scan_per_background.parquet", index=False)
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=2))
    for n, m in res["models"].items():
        print(f"scan practice half   {n:<18} {m['mean_pearson']:.4f}  HepG2 {m['HepG2']:.3f}  K562 {m['K562']:.3f}", m.get("minus_model1_ci95", ""))
    for n, m in out["natural_variants_practice_chromosomes"].items():
        print(f"natural variants     {n:<18} effect {m['mean_effect']:.4f}  activity {m['mean_activity']:.4f}", m["mean_effect_minus_model1_ci95"])


if __name__ == "__main__":
    main()

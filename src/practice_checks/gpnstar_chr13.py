"""Does GPN-Star (evolution, mammalian 'M' scores; data/teacher_full) add information to the model-3 students on PRACTICE
chromosome 13? No training, no download, no locked data.

1. Each chr13 practice variant is placed in hg38 by finding its 200-letter version 1 in our hg38 chr13 copy (the table's
   positions are hg19). Unique exact matches only.
2. GPN-Star score of the change, oriented like the measured skew: g = log P(allele 2) - log P(allele 1)
   (= llr if the hg38 letter is allele 1, -llr if it is allele 2).
3. (a) g alone against the measured effect (log2Skew), signed and as sizes; (b) student effect combined with GPN-Star:
       C1  e * (1 + w * |g| / mean|g|)        (bigger effects where evolution says the position matters)
       C2  e + w * g * sd(e) / sd(g)           (GPN-Star's direction added)
   w is chosen by 2-fold cross-validation over alternating 100,000-letter blocks of chr13 (chosen on one half, applied to the
   other), so chr13 never picks the weight it is scored with. Interval: 2,000 block resamples of (combined - student).
Scored as model3_pilot_score.py: active variants (max allele log2FC >= 1), Pearson per cell (HepG2, K562), mean.

    .venv/bin/python src/practice_checks/gpnstar_chr13.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

sys.path[:0] = ["src", "src/third_party"]
from evaluate_scan import pieces  # noqa: E402
from malinois_body import build_arch  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402

RUNS = {"P1": "runs/m3_P1_e25_seed20261113", "4M_s1": "runs/m3_P1_e25_seed20261113_w1.75_amp", "4M_s2": "runs/m3_P1_e25_seed2_w1.75_amp"}
TABLE = {"HepG2": "HEPG2", "K562": "K562"}
LETTERS = np.array(list("ACGTN"))
W_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
OUT = Path("results/practice/gpnstar_chr13.json")


def place_in_hg38(pr: pd.DataFrame) -> pd.DataFrame:
    g = np.load("data/genome/chr13.npy")
    genome = np.frombuffer(b"ACGTN", dtype=np.uint8)[np.minimum(g, 4)].tobytes().decode("ascii")
    rows = []
    for v in pr.itertuples():
        diff = [i for i, (a, b) in enumerate(zip(v.seq1, v.seq2)) if a != b]
        if len(diff) != 1:
            rows.append((v.Index, np.nan, "", "", "not one letter"))
            continue
        i = diff[0]
        start = genome.find(v.seq1)
        if start < 0 or genome.find(v.seq1, start + 1) >= 0:
            rows.append((v.Index, np.nan, "", "", "no unique match"))
            continue
        rows.append((v.Index, start + i + 1, v.seq1[i], v.seq2[i], "ok"))  # 1-based hg38 position
    m = pd.DataFrame(rows, columns=["row", "pos38", "a1", "a2", "status"]).set_index("row")
    return pr.join(m)


def gpn_scores(t: pd.DataFrame) -> np.ndarray:
    ok = t.status == "ok"
    want = np.unique(t.pos38[ok].astype(np.int64))
    files = sorted(Path("data/teacher_full").glob("chr13_rg*.parquet"))
    tab = ds.dataset([str(f) for f in files], format="parquet").to_table(filter=ds.field("pos").isin(want.tolist())).to_pandas()
    look = {(int(r.pos), r.ref, r.alt): float(r.llr_calibrated) for r in tab.itertuples()}
    refs = {int(r.pos): r.ref for r in tab.itertuples()}
    g = np.full(len(t), np.nan)
    for k, r in enumerate(t.itertuples()):
        if r.status != "ok":
            continue
        ref = refs.get(int(r.pos38))
        if ref == r.a1:
            g[k] = look.get((int(r.pos38), r.a1, r.a2), np.nan)
        elif ref == r.a2:
            s = look.get((int(r.pos38), r.a2, r.a1), np.nan)
            g[k] = -s
    return g


def student_effects(run: str, x1, x2, n: int, device) -> dict:
    cfg = json.loads((Path(run) / "log.json").read_text())["config"]
    outs = cfg["outputs"]
    m = build_arch(cfg.get("arch", "legnet"), len(outs), width=cfg.get("width", 1.0)).to(device)
    m.load_state_dict(torch.load(Path(run) / "best.pt", map_location=device, weights_only=True))
    p1, p2 = predict_multi(m, x1, device), predict_multi(m, x2, device)
    return {c: (p2[:, outs.index(f"new_{t}")] - p1[:, outs.index(f"new_{t}")]).reshape(n, 2).mean(1) for c, t in TABLE.items()}


def main() -> None:
    pr = pd.read_parquet("data/model3/practice13.parquet").reset_index(drop=True)
    t = place_in_hg38(pr)
    g = gpn_scores(t)
    print(f"placed {int((t.status == 'ok').sum())} of {len(t)} variants uniquely in hg38; GPN-Star score for {int(np.isfinite(g).sum())}", flush=True)

    masks, skews = {}, {}
    for c, tb in TABLE.items():
        a1, a2, sk = (t[f"{k}_{tb}"].to_numpy() for k in ("allele1.log2FC", "allele2.log2FC", "log2Skew"))
        masks[c] = ~np.isnan(a1) & ~np.isnan(a2) & ~np.isnan(sk) & (np.maximum(a1, a2) >= 1.0) & np.isfinite(g)
        skews[c] = sk
    block = (t.pos // 100_000).to_numpy()
    fold = block % 2
    ub = np.unique(block)
    bidx = {b: np.flatnonzero(block == b) for b in ub}
    rng = np.random.default_rng(20261113)
    draws = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]

    def score(e: dict, sel=None) -> float:
        r = []
        for c in TABLE:
            i = np.flatnonzero(masks[c]) if sel is None else sel[masks[c][sel]]
            r.append(np.corrcoef(skews[c][i], e[c][i])[0, 1])
        return float(np.mean(r))

    res = {"placed_uniquely": int((t.status == "ok").sum()), "with_gpnstar_score": int(np.isfinite(g).sum()),
           "active_with_score": {c: int(masks[c].sum()) for c in TABLE}}
    ga = {c: g for c in TABLE}
    res["gpnstar_alone"] = {"signed_pearson": round(score(ga), 4),
                            "size_pearson_abs_vs_abs": round(float(np.mean([np.corrcoef(np.abs(skews[c][masks[c]]), np.abs(g[masks[c]]))[0, 1] for c in TABLE])), 4)}
    print("GPN-Star alone:", res["gpnstar_alone"], flush=True)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x1, x2 = (torch.from_numpy(pieces(s.tolist()).reshape(-1, 230)) for s in (t.seq1, t.seq2))
    absg = np.abs(np.nan_to_num(g))
    u = absg / np.mean(absg[np.isfinite(g)])
    gz = np.nan_to_num(g)
    for name, run in RUNS.items():
        e = student_effects(run, x1, x2, len(t), dev)
        combos = {"C1": lambda w: {c: e[c] * (1 + w * u) for c in TABLE},
                  "C2": lambda w: {c: e[c] + w * gz * np.std(e[c][masks[c]]) / np.std(gz[masks[c]]) for c in TABLE}}
        out = {"student": round(score(e), 4)}
        for cname, make in combos.items():
            comb = {c: np.empty(len(t)) for c in TABLE}
            chosen = []
            for f in (0, 1):  # choose w on fold f, apply to the other fold
                tr, te = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
                w = max(W_GRID, key=lambda w: score(make(w), tr))
                chosen.append(w)
                cw = make(w)
                for c in TABLE:
                    comb[c][te] = cw[c][te]
            d = [score(comb, s) - score(e, s) for s in draws]
            out[cname] = {"cross_validated": round(score(comb), 4), "minus_student": round(score(comb) - score(e), 4),
                          "ci95": [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])], "w_chosen_per_half": chosen,
                          "best_w_on_all_chr13_optimistic": max(W_GRID, key=lambda w: score(make(w))),
                          "best_on_all_chr13_optimistic": round(max(score(make(w)) for w in W_GRID), 4)}
        res[name] = out
        print(name, json.dumps(out), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

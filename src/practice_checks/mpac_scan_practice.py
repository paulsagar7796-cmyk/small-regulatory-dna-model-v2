"""MPAC on the saturation scan's PRACTICE half, scored exactly like src/evaluate_scan.py, for a paired comparison with the
model-3 students (their per-background tables from src/evaluate_practice.py: results/practice/scancheck_<run>_scan_per_background.parquet).

MPAC as in src/mpac_reference.py: for a piece on chromosome N, the 10 networks of the fold whose test pair holds N (verified:
none trained or validated on N); MPAC's own flanks and strand averaging; one centred window. MPAC has no held-out networks for
chromosome X, so X pieces are left out; so are chromosomes whose MPAC fold is not downloaded (listed in the output). Every model
is compared on the same remaining backgrounds.
Check first: P1's plasmid read-out run through this same loop must reproduce its stored per-background values.
Practice data only (the scan's locked half is never loaded).

    .venv/bin/python src/practice_checks/mpac_scan_practice.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path[:0] = ["src", "src/third_party"]
from evaluate_practice import model2  # noqa: E402
from evaluate_scan import MIN_BASELINE, N_BOOT, SCAN_CELLS, backgrounds, pieces, scan  # noqa: E402
import modeling_mpac as mm  # noqa: E402
from mpac_reference import CELL_INDEX, MPAC_DIR, load_ensemble, pick_device, provenance  # noqa: E402
from mpra_train import SETTINGS  # noqa: E402

RUNS = {"P1": "m3_P1_e25_seed20261113", "4M_s1": "m3_P1_e25_seed20261113_w1.75_amp", "4M_s2": "m3_P1_e25_seed2_w1.75_amp",
        "pyramid": "m3_P1_e25_seed20261113_pyramid_resume22"}
OUT = Path("results/practice")


def main() -> None:
    device = pick_device()
    s = scan("practice")
    seq = backgrounds(s)
    config = json.loads((MPAC_DIR / "config.json").read_text())
    cols = {c: config["output_names"].index(n) for c, n in CELL_INDEX.items()}
    p1 = model2(RUNS["P1"], device)["m2_new"]
    rows, skipped = [], {}
    chrom_of = s.drop_duplicates("sat_ref").set_index("sat_ref").chr.astype(str)
    prov = provenance()
    for chrom in sorted(chrom_of.unique(), key=lambda c: (c == "X", c)):
        if chrom == "X":
            skipped["X"] = "MPAC has no held-out networks for X"
            continue
        fold = mm.fold_for_chromosome(prov, chrom)
        if len(list((MPAC_DIR / fold).glob("*.safetensors"))) != 10:  # not downloaded: leave the chromosome out for every model
            skipped[chrom] = f"fold {fold} not in data/models/MPAC"
            print(f"chromosome {chrom}: skipped ({skipped[chrom]})", flush=True)
            continue
        ens = load_ensemble(chrom, device)
        for ref_id in chrom_of.index[chrom_of == chrom]:
            g = s[s.sat_ref == ref_id]
            one = g[g["Cell Type"] == "HEPG2"]
            ch = one[one.oligomut != "m0"]
            original = seq[ref_id]
            assert len(original) == 200 and all(original[p - 1] == m[1] for p, m in zip(ch.mut_pos, ch.oligomut)), ref_id
            changed = [original[: p - 1] + b + original[p:] for p, b in zip(ch.mut_pos, ch.mut_base)]
            mp = ens.predict([original] + changed, batch_size=256, device=device).numpy()  # (1 + n, outputs), strands averaged
            x = pieces([original] + changed)
            pp = p1(x.reshape(-1, x.shape[-1]))
            for cell_tab, cell in SCAN_CELLS.items():
                gc = g[g["Cell Type"] == cell_tab]
                base = float(gc[gc.oligomut == "m0"].log2FC.iloc[0])
                if base < MIN_BASELINE:
                    continue
                y = gc[gc.oligomut != "m0"].set_index("oligomut").log2Skew.reindex(ch.oligomut.to_numpy())
                ok = y.notna().to_numpy()
                row = {"sat_ref": ref_id, "parent": g.sat_ref_parent.iloc[0], "cell": cell, "chr": chrom, "changes": int(ok.sum())}
                e_m = (mp[1:, cols[cell]] - mp[0, cols[cell]])[ok]
                e_p = pp[cell].reshape(len(x), 2).mean(axis=1)
                e_p = (e_p[1:] - e_p[0])[ok]
                yy = y.to_numpy()[ok]
                row["MPAC"], row["MPAC_spearman"] = float(np.corrcoef(yy, e_m)[0, 1]), float(spearmanr(yy, e_m).statistic)
                row["P1_check"] = float(np.corrcoef(yy, e_p)[0, 1])
                rows.append(row)
        print(f"chromosome {chrom}: done", flush=True)
    d = pd.DataFrame(rows)
    for k, r in RUNS.items():
        t = pd.read_parquet(OUT / f"scancheck_{r}_scan_per_background.parquet")[["sat_ref", "cell", "m2_new"]].rename(columns={"m2_new": k})
        d = d.merge(t, on=["sat_ref", "cell"], how="left")
    assert d[list(RUNS)].notna().all().all(), "a student has no stored value for some background"
    gap = float((d.P1_check - d.P1).abs().max())
    print(f"check: P1 through this loop vs its stored values, largest difference {gap:.2e} over {len(d)} backgrounds", flush=True)
    assert gap < 1e-3, "this loop does not reproduce the scan recipe"

    overall = lambda t, c: float(np.mean([t[t.cell == x][c].mean() for x in SCAN_CELLS.values()]))
    names = ["MPAC"] + list(RUNS)
    rng = np.random.default_rng(SETTINGS["seed"])
    parents = d.parent.unique()
    byp = {p: t for p, t in d.groupby("parent")}
    boot = {n: [] for n in names}
    for _ in range(N_BOOT):  # resample pieces; both alleles and both cell types of a piece stay together (as evaluate_scan.py)
        t = pd.concat([byp[p] for p in rng.choice(parents, len(parents))])
        for n in names:
            boot[n].append(overall(t, n))
    res = {"what": "saturation scan, practice half, autosomal backgrounds only (MPAC has no held-out networks for X); plasmid "
                   "read-out for the students; MPAC = the 10 clean networks per chromosome, one centred window",
           "backgrounds": int(len(d)), "pieces": int(len(parents)), "by_cell": d.groupby("cell").size().to_dict(),
           "chromosomes_scored": sorted(d.chr.unique().tolist()), "chromosomes_skipped": skipped,
           "check_P1_max_abs_difference": gap, "device": str(device),
           "score": {n: {"mean_pearson": round(overall(d, n), 4), **{c: round(float(d[d.cell == c][n].mean()), 4) for c in SCAN_CELLS.values()}}
                     for n in names},
           "minus_MPAC": {}}
    res["score"]["MPAC"]["mean_spearman"] = round(overall(d, "MPAC_spearman"), 4)
    for n in RUNS:
        diff = np.array(boot[n]) - np.array(boot["MPAC"])
        res["minus_MPAC"][n] = {"diff": round(overall(d, n) - overall(d, "MPAC"), 4),
                                "ci95": [round(float(v), 4) for v in np.percentile(diff, [2.5, 97.5])]}
    d.to_parquet(OUT / "mpac_scan_practice_per_background.parquet", index=False)
    (OUT / "mpac_scan_practice.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

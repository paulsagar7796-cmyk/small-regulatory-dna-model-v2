"""D4c: single-letter changes on fresh data (the Siraj and others saturation-mutagenesis scan).
Data, split, recipe and pass marks: predeclarations/PREDECLARATION_SCAN.md.

    .venv/bin/python src/evaluate_scan.py --practice [--tag T] [--extra name=arch:width:run_dir,...]   (a+b+c after the = averages several models)
        scores the PRACTICE half only (odd chromosomes and X). The locked half is never loaded into memory.
    .venv/bin/python src/evaluate_scan.py
        the locked half (even chromosomes). Refuses unless the pre-declaration is CONFIRMED and the student is fixed.
        Run ONCE.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from baseline_kmer import features  # noqa: E402
from legnet_reference import MAIN_MODEL, WEIGHTS, load_model, predictor  # noqa: E402
from mpra_data import CELLS, TRAIN_FOLDS, encode, load  # noqa: E402
from mpra_model import ActivityStudent  # noqa: E402
from mpra_train import SETTINGS, checkpoint  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402
from mpra_train_v2 import build  # noqa: E402

PREDECLARATION = Path("predeclarations/PREDECLARATION_SCAN.md")
DATA, OUT = Path("data/siraj"), Path("results/scan")
SCAN_CELLS = {"HEPG2": "HepG2", "K562": "K562"}  # the table's names -> ours
LEFT, RIGHT = "AGGACCGGATCAACT", "CATTGCGTGAACCGA"  # the adapters around every training piece
MIN_BASELINE, N_BOOT = 1.0, 2000
V2_MAIN = {"arch": "legnet", "width": 1.0, "run": "distill_legnet_t3_lam1.0_mut0.05_e25_seed20261113"}
V2_SEED2 = {"arch": "legnet", "width": 1.0, "run": "distill_legnet_t3_lam1.0_mut0.05_e25_seed2"}


def scan(split: str) -> pd.DataFrame:
    """The scan rows of one half. The other half is dropped before anything else is done."""
    s = pd.read_parquet(DATA / "satmut.parquet")
    s = s[s.is_haplo == 0]
    odd = s.chr.map(lambda c: c == "X" or int(c) % 2 == 1)
    return s[odd if split == "practice" else ~odd].reset_index(drop=True)


def backgrounds(s: pd.DataFrame) -> dict:
    """sat_ref -> the 200-letter original piece (allele R = 'Allele 1 Oligo', allele A = 'Allele 2 Oligo'; checked)."""
    o = pd.read_parquet(DATA / "oligos.parquet").set_index("Variant")
    b = s.drop_duplicates("sat_ref")
    return {r.sat_ref: o.at[r.var1, "Allele 1 Oligo" if r.allele == "R" else "Allele 2 Oligo"] for r in b.itertuples()}


def pieces(seqs: list[str]) -> np.ndarray:
    """(n, 2 orientations, 230) codes: the piece between the adapters, and the piece inserted the other way round."""
    c = encode(pd.Series(seqs))
    l, r = encode(pd.Series([LEFT]))[0], encode(pd.Series([RIGHT]))[0]
    out = np.zeros((len(c), 2, c.shape[1] + 30), dtype=np.uint8)
    out[:, 0] = np.concatenate([np.tile(l, (len(c), 1)), c, np.tile(r, (len(c), 1))], axis=1)
    out[:, 1] = np.concatenate([np.tile(l, (len(c), 1)), (3 - c)[:, ::-1], np.tile(r, (len(c), 1))], axis=1)
    return out


def models(device, extra: dict) -> dict:
    """name -> function mapping (m, 230) codes to {our cell name: (m,) predictions}."""
    fns = {}

    def multi(spec):
        if isinstance(spec, list):  # several models averaged (practice only: an upper reference for one student)
            parts = [multi(s) for s in spec]
            return lambda codes: {c: np.mean([f(codes)[c] for f in parts], axis=0) for c in SCAN_CELLS.values()}
        m = (ActivityStudent(n_out=len(CELLS)) if spec["arch"] == "v1" else build(spec["arch"], len(CELLS), width=spec.get("width", 1.0))).to(device)
        m.load_state_dict(torch.load(checkpoint(spec["run"]), map_location=device))

        def fn(codes):
            p = predict_multi(m, torch.from_numpy(codes), device)
            return {c: p[:, CELLS.index(c)] for c in SCAN_CELLS.values()}
        return fn

    fns["student_v2"], fns["student_v2_seed2"] = multi(V2_MAIN), multi(V2_SEED2)
    try:
        fns["student_v1"] = multi({"arch": "v1", "run": f"multi_c128_seed{SETTINGS['seed']}"})
    except FileNotFoundError:  # the earlier v1 student (reported only) is not shipped with the repository
        print("note: the v1 student's checkpoint is not present; it is left out", flush=True)
    for name, spec in extra.items():
        fns[name] = multi(spec)
    pub = {c: predictor([load_model(c, WEIGHTS / c / MAIN_MODEL, device)], device) for c in SCAN_CELLS.values()}
    fns["published"] = lambda codes: {c: pub[c](codes) for c in SCAN_CELLS.values()}
    ridge = {}
    for c in SCAN_CELLS.values():
        tr = load(c, TRAIN_FOLDS)
        ridge[c] = Ridge(alpha=100.0).fit(features(encode(tr.seq)), tr.mean_value.to_numpy())
    fns["baseline"] = lambda codes: {c: ridge[c].predict(features(codes)) for c in SCAN_CELLS.values()}
    return fns


def score(split: str, extra: dict, device, more: dict | None = None) -> dict:
    s = scan(split)
    seq = backgrounds(s)
    overlap = pd.read_parquet(DATA / "satmut_backgrounds_overlap.parquet").set_index("sat_ref").shared
    fns = models(device, extra)
    fns.update(more or {})  # model 2's outputs (src/evaluate_practice.py)
    rows = []
    for ref_id, g in s.groupby("sat_ref", sort=False):
        one = g[g["Cell Type"] == "HEPG2"]
        ch = one[one.oligomut != "m0"]
        original = seq[ref_id]
        assert len(original) == 200 and all(original[p - 1] == m[1] for p, m in zip(ch.mut_pos, ch.oligomut)), ref_id
        changed = [original[: p - 1] + b + original[p:] for p, b in zip(ch.mut_pos, ch.mut_base)]
        x = pieces([original] + changed)  # (1 + n, 2, 230)
        pred = {name: fn(x.reshape(-1, x.shape[-1])) for name, fn in fns.items()}
        for cell_tab, cell in SCAN_CELLS.items():
            gc = g[g["Cell Type"] == cell_tab]
            base = float(gc[gc.oligomut == "m0"].log2FC.iloc[0])
            if base < MIN_BASELINE:
                continue
            y = gc[gc.oligomut != "m0"].set_index("oligomut").log2Skew.reindex(ch.oligomut.to_numpy())
            ok = y.notna().to_numpy()
            row = {"sat_ref": ref_id, "parent": g.sat_ref_parent.iloc[0], "cell": cell, "changes": int(ok.sum()),
                   "baseline_activity": base, "shares_training_dna": bool(overlap.get(ref_id, 0) > 0)}
            for name, p in pred.items():
                e = p[cell].reshape(len(x), 2).mean(axis=1)
                e = (e[1:] - e[0])[ok]
                row[name] = float(np.corrcoef(y.to_numpy()[ok], e)[0, 1])
                row[name + "_spearman"] = float(spearmanr(y.to_numpy()[ok], e).statistic)
            rows.append(row)
    d = pd.DataFrame(rows)
    names = list(fns)

    def overall(t: pd.DataFrame, col: str) -> float:
        return float(np.mean([t[t.cell == c][col].mean() for c in SCAN_CELLS.values()]))

    res = {"split": split, "pieces": int(d.parent.nunique()), "scored_backgrounds_by_cell": d.groupby("cell").size().to_dict(),
           "models": {n: {"mean_pearson": overall(d, n), "mean_spearman": overall(d, n + "_spearman"),
                          **{c: float(d[d.cell == c][n].mean()) for c in SCAN_CELLS.values()},
                          "shares_training_dna": overall(d[d.shares_training_dna], n),
                          "no_shared_dna": overall(d[~d.shares_training_dna], n)} for n in names}}
    rng = np.random.default_rng(SETTINGS["seed"])
    parents = d.parent.unique()
    by_parent = {p: t for p, t in d.groupby("parent")}
    boot = {n: [] for n in names}
    for _ in range(N_BOOT):  # resample pieces; both alleles and both cell types of a piece stay together
        t = pd.concat([by_parent[p] for p in rng.choice(parents, len(parents))])
        for n in names:
            boot[n].append(overall(t, n))
    q = lambda a: [float(v) for v in np.percentile(a, [2.5, 97.5])]
    for n in names:
        res["models"][n]["mean_pearson_ci95"] = q(boot[n])
        if n != "published":
            res["models"][n]["minus_published_ci95"] = q(np.array(boot[n]) - np.array(boot["published"]))
    return res, d


def main() -> None:
    practice = "--practice" in sys.argv
    text = PREDECLARATION.read_text()
    extra = {}
    if "--extra" in sys.argv:
        for item in sys.argv[sys.argv.index("--extra") + 1].split(","):
            name, spec = item.split("=")
            specs = [dict(zip(("arch", "width", "run"), s.split(":"))) for s in spec.split("+")]  # a+b+c = averaged
            for s in specs:
                s["width"] = float(s["width"])
            extra[name] = specs[0] if len(specs) == 1 else specs
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    if practice:
        res, d = score("practice", extra, device)
        tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "practice"
        d.to_parquet(OUT / f"{tag}_per_background.parquet", index=False)
        (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return
    if "**Status: CONFIRMED**" not in text:
        sys.exit("Stopped: predeclarations/PREDECLARATION_SCAN.md is not CONFIRMED yet.")
    students = json.loads(text.split("<!-- students:")[1].split("-->")[0])
    if not students.get("fixed"):
        sys.exit("Stopped: the student is not fixed yet in the pre-declaration.")
    if (OUT / "results.json").exists():
        sys.exit("Stopped: the locked half has already been scored once (results/scan/results.json exists).")
    global V2_MAIN, V2_SEED2
    V2_MAIN, V2_SEED2 = students["main"], students["other_seed"]
    marks = json.loads(text.split("<!-- marks:")[1].split("-->")[0])
    res, d = score("locked", {}, device)
    res["marks"], res["students"] = marks, students
    d.to_parquet(OUT / "locked_per_background.parquet", index=False)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

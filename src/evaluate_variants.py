"""D4b one-shot test: does the Path B student predict the measured effect of single-letter changes?
Data: Kircher and others 2019 (data/satmut). Rules, recipe and pass marks: predeclarations/PREDECLARATION_VARIANTS.md.

Refuses to run unless that file says CONFIRMED and names the student in a line like
    <!-- student: {"kind": "multi", "seed": 20261113, "other_seed": 2} -->
("multi" = one shared body with three outputs; "single" = one student per cell type).

Run from the repository root:
    .venv/bin/python src/evaluate_variants.py --check   (builds the inputs and checks reference letters; compares nothing)
    .venv/bin/python src/evaluate_variants.py           (the test; run ONCE)
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
from baseline_kmer import features  # noqa: E402
from data import genome  # noqa: E402  (src/data.py)
from mpra_data import CELLS, TRAIN_FOLDS, encode, load  # noqa: E402
from mpra_model import ActivityStudent, count_parameters  # noqa: E402
from mpra_train import SETTINGS, checkpoint  # noqa: E402
from mpra_train_multi import predict_multi  # noqa: E402

PREDECLARATION = Path("predeclarations/PREDECLARATION_VARIANTS.md")
SATMUT, OUT = Path("data/satmut"), Path("results/variants")
MAIN = {"F9": "HepG2", "LDLR": "HepG2", "SORT1": "HepG2", "PKLR-48h": "K562"}  # the four switches of the pass marks
REPEATS = {"LDLR.2": "HepG2", "SORT1.2": "HepG2", "PKLR-24h": "K562"}  # repeat experiments, reported only
LEFT, RIGHT = "AGGACCGGATCAACT", "CATTGCGTGAACCGA"  # the adapters around every training piece
INSERT, OFFSETS, MIN_TAGS, N_BOOT = 200, (-60, -30, 0, 30, 60), 10, 2000
_L = np.array(["ACGT".index(b) for b in LEFT], dtype=np.uint8)
_R = np.array(["ACGT".index(b) for b in RIGHT], dtype=np.uint8)


def pieces(chrom: str, pos: np.ndarray, alt: np.ndarray):
    """For each change: original and changed 230-letter pieces, shape (n, offsets, 2 orientations, 230).
    pos is 1-based. alt is a letter code 0..3, or -1 for a 1-letter deletion."""
    g = genome(chrom)
    n = len(pos)
    ref = np.zeros((n, len(OFFSETS), 2, INSERT + 30), dtype=np.uint8)
    new = np.zeros_like(ref)
    for i in range(n):
        p0 = int(pos[i]) - 1
        for j, off in enumerate(OFFSETS):
            at = INSERT // 2 + off  # where the changed letter sits inside the 200-letter insert
            start = p0 - at
            w = np.array(g[start : start + INSERT + 1])  # one spare letter, used only by deletions
            assert w.max() < 4, "unknown letter in the genome window"
            a = w[:INSERT].copy()
            if alt[i] >= 0:
                b = a.copy()
                b[at] = alt[i]
            else:
                b = np.delete(w, at)
            for k, (src, dst) in enumerate(((a, ref), (b, new))):
                dst[i, j, 0] = np.concatenate([_L, src, _R])
                dst[i, j, 1] = np.concatenate([_L, (3 - src)[::-1], _R])  # the insert the other way round
    return ref, new


def table(name: str) -> pd.DataFrame:
    d = pd.read_csv(SATMUT / f"GRCh38_{name}.tsv", sep="\t", dtype={"Chromosome": str})
    d = d[d.Tags >= MIN_TAGS].reset_index(drop=True)
    g = genome(d.Chromosome[0])
    assert (np.array(list("ACGT"))[g[d.Position.to_numpy() - 1]] == d.Ref.to_numpy()).all(), f"{name}: reference letters differ"
    d["alt_code"] = d.Alt.map({"A": 0, "C": 1, "G": 2, "T": 3, "-": -1}).astype(int)
    return d


def effect(fn, ref: np.ndarray, new: np.ndarray) -> np.ndarray:
    """fn maps (m, 230) codes to m predictions. Returns changed minus original, averaged over offsets and orientations."""
    shape = ref.shape[:3]
    return (fn(new.reshape(-1, ref.shape[-1])) - fn(ref.reshape(-1, ref.shape[-1]))).reshape(shape).mean(axis=(1, 2))


def ci(a) -> list[float]:
    return [float(q) for q in np.percentile(a, [2.5, 97.5])]


def main() -> None:
    check_only = "--check" in sys.argv
    text = PREDECLARATION.read_text()
    if not check_only and ("**Status: CONFIRMED**" not in text or "<!-- student:" not in text):
        sys.exit("Stopped: the pre-declaration must be CONFIRMED and must name the student before scoring.")
    marks = json.loads(text.split("<!-- marks:")[1].split("-->")[0])
    student = (json.loads(text.split("<!-- student:")[1].split("-->")[0]) if "<!-- student:" in text
               else {"kind": "multi", "seed": SETTINGS["seed"], "other_seed": 2})  # default used by --check only
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def student_fn(cell: str, seed: int):
        multi = student["kind"] == "multi"
        m = ActivityStudent(n_out=len(CELLS) if multi else 1).to(device)
        if not check_only:
            run = f"multi_c128_seed{seed}" if multi else f"{cell}_c128_seed{seed}"
            m.load_state_dict(torch.load(checkpoint(run), map_location=device))
        k = CELLS.index(cell)

        def fn(codes: np.ndarray) -> np.ndarray:
            p = predict_multi(m, torch.from_numpy(codes), device)
            return p[:, k] if multi else p.reshape(-1)
        return fn, count_parameters(m)

    if check_only:
        for name, cell in {**MAIN, **REPEATS}.items():
            d = table(name)
            ref, new = pieces(d.Chromosome[0], d.Position.to_numpy(), d.alt_code.to_numpy())
            fn, _ = student_fn(cell, student["seed"])
            e = effect(fn, ref[:8], new[:8])  # an UNTRAINED model, only to check shapes
            differ = (ref != new).any(axis=-1).all()
            print(f"{name}: {len(d)} rows ({(d.alt_code >= 0).sum()} single-letter, {(d.alt_code < 0).sum()} deletions), "
                  f"pieces {ref.shape}, every changed piece differs: {bool(differ)}, effect shape {e.shape}")
        print("check finished: reference letters match, nothing was compared with the measurements")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SETTINGS["seed"])
    base = {}
    for cell in sorted(set(MAIN.values())):
        tr = load(cell, TRAIN_FOLDS)
        ridge = Ridge(alpha=100.0).fit(features(encode(tr.seq)), tr.mean_value.to_numpy())
        base[cell] = lambda codes, r=ridge: r.predict(features(codes))
    res = {"marks": marks, "student": student, "switches": {}, "repeat_experiments": {}}
    for group, key in ((MAIN, "switches"), (REPEATS, "repeat_experiments")):
        for name, cell in group.items():
            d = table(name)
            ref, new = pieces(d.Chromosome[0], d.Position.to_numpy(), d.alt_code.to_numpy())
            fn, res["parameters"] = student_fn(cell, student["seed"])
            fn2, _ = student_fn(cell, student["other_seed"])
            d["student"], d["student_other_seed"], d["baseline"] = effect(fn, ref, new), effect(fn2, ref, new), effect(base[cell], ref, new)
            d.drop(columns="alt_code").to_parquet(OUT / f"predictions_{name}.parquet", index=False)
            s, dele = d[d.alt_code >= 0], d[d.alt_code < 0]
            y, p, b = s.Value.to_numpy(), s.student.to_numpy(), s.baseline.to_numpy()
            rs, diff = [], []
            for _ in range(N_BOOT):  # resample changes, the same resample for student and baseline
                i = rng.integers(0, len(y), len(y))
                r = np.corrcoef(y[i], p[i])[0, 1]
                rs.append(r)
                diff.append(r - np.corrcoef(y[i], b[i])[0, 1])
            res[key][name] = {
                "cell": cell, "changes": len(s), "pearson": float(np.corrcoef(y, p)[0, 1]), "pearson_ci95": ci(rs),
                "spearman": float(spearmanr(y, p).statistic),
                "pearson_other_seed": float(np.corrcoef(y, s.student_other_seed)[0, 1]),
                "baseline_pearson": float(np.corrcoef(y, b)[0, 1]), "student_minus_baseline_ci95": ci(diff),
                "deletions": len(dele),
                "deletions_pearson": float(np.corrcoef(dele.Value, dele.student)[0, 1]) if len(dele) > 2 else None,
                "deletions_baseline_pearson": float(np.corrcoef(dele.Value, dele.baseline)[0, 1]) if len(dele) > 2 else None,
            }
    sw = res["switches"]
    res["mean_pearson"] = float(np.mean([v["pearson"] for v in sw.values()]))
    res["mean_pearson_other_seed"] = float(np.mean([v["pearson_other_seed"] for v in sw.values()]))
    res["mean_baseline_pearson"] = float(np.mean([v["baseline_pearson"] for v in sw.values()]))
    res["rule1_pass"] = all(v["pearson_ci95"][0] > 0 for v in sw.values())
    res["rule2_pass"] = res["mean_pearson"] >= marks["min_mean_pearson"]
    res["rule3_pass"] = res["mean_pearson"] > res["mean_baseline_pearson"]
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

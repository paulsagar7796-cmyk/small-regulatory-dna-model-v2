"""Reference run of MPAC (an outside model) on model 2's test variants. Rule: predeclarations/REFERENCE_MPAC.md.

    .venv/bin/python src/mpac_reference.py --practice       practice chromosomes 6, 13, 20: a check of this script (and the 0.80 sanity line)
    .venv/bin/python src/mpac_reference.py --check          MPAC's own predict() against a hand-built forward pass, on practice pieces only
    .venv/bin/python src/mpac_reference.py --locked         chromosomes 5, 14, 22, ONCE (refuses unless the practice file exists and
                                                            passed the sanity line, and no locked MPAC result exists yet)

MPAC = 110 networks of the Malinois design (Butts and others 2025), weights from the Hugging Face mirror saarantras1/MPAC
(safetensors only, in data/models/MPAC). For a variant on chromosome N the 10 networks whose TEST pair contains N are averaged
(they did not train on N and did not use it for validation). Input: our 200-letter version 1 and version 2 through the vendored
MPAC `predict` (its own flanks, 200 + 200 letters, and its own strand handling: reverse complement of the insert only,
re-flanked forward, averaged with the forward pass). One centred window per version (declared deviation from the preprint's
18 shifted windows). Scoring, blocks and resamples: exactly src/evaluate_locked.py.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
import modeling_mpac as mm  # noqa: E402  (vendored, read in full before use)
from evaluate_locked import evaluate  # noqa: E402
from evaluate_practice import MORE, TABLE  # noqa: E402

MPAC_DIR = Path("data/models/MPAC")
PRACTICE_OUT = Path("results/practice/mpac_reference.json")
LOCKED_OUT = Path("results/locked/MPAC_REFERENCE.json")
LOCKED_RESULT = Path("results/locked/RESULT.json")
SANITY_LINE = 0.80
CELL_INDEX = {"HepG2": "HepG2", "K562": "K562"}  # our names -> MPAC output names (order is read from config.json)
ALL_CHR = {str(i) for i in range(1, 23)}


def provenance() -> list:
    return json.loads((MPAC_DIR / "provenance.json").read_text())


def verify_fold(chrom: str, prov: list) -> tuple[str, list]:
    """The fold whose TEST pair holds `chrom`; its 10 members must all exist, and none may have trained on or been validated on it."""
    fold = mm.fold_for_chromosome(prov, chrom)
    members = sorted((r for r in prov if Path(r["file"]).parent.as_posix() == fold), key=lambda r: r["file"])
    assert len(members) == 10, f"{fold}: {len(members)} members"
    tests = {tuple(r["test_chrs"]) for r in members}
    assert len(tests) == 1 and chrom in next(iter(tests)), f"{fold}: test pair {tests}"
    vals = [tuple(r["val_chrs"]) for r in members]
    assert len(set(vals)) == 10, f"{fold}: validation pairs not distinct: {vals}"
    for r in members:
        # trained on = every autosome pair except its test pair and its validation pair
        assert chrom not in r["val_chrs"] and chrom in r["test_chrs"], f"{r['file']} may have seen chromosome {chrom}"
        trained_on = ALL_CHR - set(r["test_chrs"]) - set(r["val_chrs"])
        assert chrom not in trained_on and len(trained_on) == 18, f"{r['file']}: training chromosomes {sorted(trained_on)}"
        assert (MPAC_DIR / r["file"]).stat().st_size == 16_445_652, f"{r['file']}: unexpected size"
    return fold, members


def load_ensemble(chrom: str, device) -> mm.MPACEnsemble:
    prov, config = provenance(), json.loads((MPAC_DIR / "config.json").read_text())
    _, members = verify_fold(chrom, prov)
    nets = []
    for r in members:
        net = mm.MPACModel(**config)
        net.load_state_dict(load_file(str(MPAC_DIR / r["file"])))  # safetensors: no pickle
        nets.append(net.eval())
    return mm.MPACEnsemble(nets).to(device)


def make_mpac_fn(device, batch_size: int = 256):
    """fn(table) -> {"HepG2": (p1, p2), "K562": (p1, p2)}: the fold's 10-network average, version 1 and version 2 of each variant."""
    config = json.loads((MPAC_DIR / "config.json").read_text())
    cols = {c: config["output_names"].index(n) for c, n in CELL_INDEX.items()}  # read from config, not assumed
    cache = {}

    def fn(t):
        out = {c: (np.full(len(t), np.nan), np.full(len(t), np.nan)) for c in TABLE}
        for chrom in sorted(t.chr.astype(str).unique()):
            if chrom not in cache:
                cache.clear()  # one ensemble in memory at a time
                cache[chrom] = load_ensemble(chrom, device)
            ens = cache[chrom]
            idx = np.flatnonzero((t.chr.astype(str) == chrom).to_numpy())
            for v, col in enumerate(("seq1", "seq2")):
                p = ens.predict(t[col].to_numpy()[idx].tolist(), batch_size=batch_size, device=device).numpy()
                for c, k in cols.items():
                    out[c][v][idx] = p[:, k]
        assert all(np.isfinite(a).all() for c in TABLE for a in out[c]), "missing MPAC predictions"
        return out

    return fn


def pick_device():
    """MPS if it gives the same numbers as CPU on a small batch, else CPU."""
    if not torch.backends.mps.is_available():
        return torch.device("cpu")
    try:
        import pandas as pd
        t = pd.read_parquet(MORE / "practice.parquet", columns=["chr", "seq1"]).iloc[:16]
        chrom = str(t.chr.iloc[0])
        t = t[t.chr.astype(str) == chrom]
        a = load_ensemble(chrom, torch.device("cpu")).predict(t.seq1.tolist(), device=torch.device("cpu")).numpy()
        b = load_ensemble(chrom, torch.device("mps")).predict(t.seq1.tolist(), device=torch.device("mps")).numpy()
        return torch.device("mps") if np.abs(a - b).max() < 1e-3 else torch.device("cpu")
    except Exception as e:  # noqa: BLE001
        print("MPS not usable, using CPU:", type(e).__name__, str(e)[:100])
        return torch.device("cpu")


def students(text_path="predeclarations/PREDECLARATION_MORE_DATA.md") -> dict:
    s = json.loads(re.search(r"<!-- students: (.*?) -->", Path(text_path).read_text()).group(1))
    assert s.get("fixed"), "students are not fixed"
    return s


def report(res: dict) -> None:
    for n, m in res["models"].items():
        print(f"{n:<18} effect {m['mean_effect']:.4f} (95% {m['mean_effect_ci95'][0]:.3f} to {m['mean_effect_ci95'][1]:.3f}) "
              + " ".join(f"{c} {m[c]['effect']:.3f}" for c in TABLE)
              + f"  spearman {m['mean_effect_spearman']:.3f}  activity {m['mean_activity']:.3f} ("
              + " ".join(f"{c} {m[c]['activity']:.3f}" for c in TABLE) + f")  model 2 minus this {[round(v, 3) for v in m['model2_minus_this_ci95']]}")
    print("variants", res["variants"], "blocks", res["blocks"], {c: res["models"]["mpac"][c]["effect_variants"] for c in TABLE})


def reading(ci: list) -> str:
    """Fixed in REFERENCE_MPAC.md: interval of (model 2 minus MPAC)."""
    return "MPAC is ahead" if ci[1] < 0 else "model 2 is ahead" if ci[0] > 0 else "level"


def practice() -> None:
    device = pick_device()
    print("device", device)
    res = evaluate(MORE / "practice.parquet", students(), device, extra={"mpac": make_mpac_fn(device)})
    m = res["models"]["mpac"]
    res["mpac_practice"] = {"sanity_line": SANITY_LINE, "mpac_mean_activity": m["mean_activity"],
                            "mpac_activity_per_cell": {c: m[c]["activity"] for c in TABLE},
                            "sanity_pass": bool(m["mean_activity"] >= SANITY_LINE and min(m[c]["activity"] for c in TABLE) >= SANITY_LINE),
                            "reading_if_this_were_the_test": reading(m["model2_minus_this_ci95"]), "device": str(device),
                            "note": "practice chromosomes only; one centred 200-letter window per version"}
    res["student"] = students()
    PRACTICE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PRACTICE_OUT.write_text(json.dumps(res, indent=2))
    report(res)
    p = res["mpac_practice"]
    print(f"MPAC activity correlation: mean {p['mpac_mean_activity']:.3f}, per cell {p['mpac_activity_per_cell']}; sanity line {SANITY_LINE}: "
          + ("CLEARED" if p["sanity_pass"] else "NOT CLEARED (script treated as faulty)"))
    print("model 2 minus MPAC (95% interval):", [round(v, 4) for v in m["model2_minus_this_ci95"]], "->", p["reading_if_this_were_the_test"], "(practice, not the test)")


def locked() -> None:
    if not PRACTICE_OUT.exists() or not json.loads(PRACTICE_OUT.read_text()).get("mpac_practice", {}).get("sanity_pass"):
        sys.exit("Stopped: the practice check file is missing or MPAC's practice activity correlation did not clear the sanity line.")
    if LOCKED_OUT.exists():
        sys.exit("Stopped: the MPAC reference run on the locked chromosomes has already been done. It is run once.")
    ref = json.loads(LOCKED_RESULT.read_text())
    device = pick_device()
    res = evaluate(MORE / "locked.parquet", ref["student"], device, extra={"mpac": make_mpac_fn(device)})
    for n in ("model2", "student_v2", "published", "baseline"):
        a, b = res["models"][n]["mean_effect"], ref["models"][n]["mean_effect"]
        if abs(a - b) > 1e-6:
            sys.exit(f"VOID: {n} mean effect {a:.8f} does not reproduce RESULT.json {b:.8f}. Nothing written.")
    m = res["models"]["mpac"]
    res["reading"] = reading(m["model2_minus_this_ci95"])
    res["size_note"] = "model 2: one network of 1.32 million parameters; MPAC for one chromosome: 10 networks, about 41 million parameters in total"
    res["rule"] = "predeclarations/REFERENCE_MPAC.md"
    res["device"] = str(device)
    res["declared_deviation"] = "one centred 200-letter window per version, not the preprint's 18 shifted windows"
    LOCKED_OUT.write_text(json.dumps(res, indent=2))
    report(res)
    print("model 2 minus MPAC (95% interval):", [round(v, 4) for v in m["model2_minus_this_ci95"]], "->", res["reading"])


def check() -> None:
    """MPAC's own predict() against a forward pass built by hand from the constants (practice pieces only)."""
    import pandas as pd
    t = pd.read_parquet(MORE / "practice.parquet", columns=["chr", "seq1"])
    t = t[t.chr.astype(str) == "6"].iloc[:6]  # practice chromosome 6, one fold
    device = torch.device("cpu")
    prov, config = provenance(), json.loads((MPAC_DIR / "config.json").read_text())
    fold, members = verify_fold("6", prov)
    net = mm.MPACModel(**config)
    net.load_state_dict(load_file(str(MPAC_DIR / members[0]["file"])))
    net.eval()
    seqs = t.seq1.tolist()
    wrapper = net.predict(seqs, device=device).numpy()

    left, right = mm.MPRA_UPSTREAM[-200:], mm.MPRA_DOWNSTREAM[:200]
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    onehot = lambda s: torch.tensor([[float(ch == b) for ch in s] for b in "ACGT"])  # noqa: E731  (rows A, C, G, T)
    manual = []
    for s in seqs:
        fwd = left + s + right
        rev = left + "".join(comp[ch] for ch in reversed(s)) + right  # reverse complement of the INSERT only, flanks as before
        x = torch.stack([onehot(fwd), onehot(rev)])
        with torch.no_grad():
            p = net(x)
        manual.append(p.mean(0).numpy())
    manual = np.array(manual)
    # the other strand convention (flip the whole 600-letter construct), for contrast only
    other = []
    for s in seqs:
        fwd = left + s + right
        rc_all = "".join(comp[ch] for ch in reversed(fwd))
        with torch.no_grad():
            other.append(net(torch.stack([onehot(fwd), onehot(rc_all)])).mean(0).numpy())
    other = np.array(other)
    print(f"fold {fold}, member {members[0]['file']}, {len(seqs)} practice pieces; output order {config['output_names']}")
    print("max |wrapper - manual forward pass|           :", float(np.abs(wrapper - manual).max()))
    print("max |wrapper - whole-construct-flip variant|   :", float(np.abs(wrapper - other).max()), "(different convention, for contrast)")
    print("flank lengths", len(left), len(right), "; left flank = last 200 letters of MPRA_UPSTREAM, right = first 200 of MPRA_DOWNSTREAM")
    print("strand convention used by the wrapper: reverse complement of the 200-letter insert only, re-flanked in forward orientation, averaged with the forward pass")


if __name__ == "__main__":
    if "--practice" in sys.argv:
        practice()
    elif "--check" in sys.argv:
        check()
    elif "--locked" in sys.argv:
        locked()
    else:
        sys.exit(__doc__)

"""Reference run: the published MPRA-LegNet model on the same single-letter-change test as our student (D4b).
Same data, same filter, same scoring recipe (src/evaluate_variants.py). Not a pass/fail test for us: it tells
us where our 0.533 stands. The reading rule is recorded in predeclarations/PREDECLARATION_VARIANTS.md before this is run.

Weights: zenodo.org/records/10558183 (CC BY 4.0), final_dump.zip, models/{HepG2,K562}/.../best_model_test10_val*.ckpt
Main comparison: best_model_test10_val9 (the same split as our student: fold 9 validation, fold 10 never trained on).
Also reported: the average of all 9 models that held out fold 10.

Run from the repository root:
    .venv/bin/python src/legnet_reference.py --sanity   (fold 9 only: checks the weights are loaded correctly)
    .venv/bin/python src/legnet_reference.py            (the reference run)
"""

import collections
import json
import pickle
import pickletools
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from evaluate_variants import MAIN, N_BOOT, OUT, PREDECLARATION, REPEATS, ci, effect, pieces, table  # noqa: E402
from legnet_model import LegNet  # noqa: E402
from mpra_data import VALIDATION_FOLD, element_pearson, encode, load  # noqa: E402
from mpra_train import SETTINGS  # noqa: E402

WEIGHTS = Path("data/models/MPRALegNet")
MAIN_MODEL = "best_model_test10_val9.ckpt"
_EXPECTED = {"collections OrderedDict", "torch._utils _rebuild_tensor_v2", "torch FloatStorage", "torch LongStorage",
             "__builtin__ getattr", "torch.optim.lr_scheduler OneCycleLR", "torch.optim.adamw AdamW",
             "collections defaultdict", "__builtin__ dict", "torch._utils _rebuild_parameter"}
_SWAP = [0, 2, 1, 3]  # our codes are A0 C1 G2 T3; LegNet's channels are A0 G1 C2 T3


class _Stub:
    """Stands in for the optimiser and scheduler objects stored in the checkpoint; nothing of theirs is run."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _RestrictedUnpickler(pickle.Unpickler):
    _REAL = {("collections", "OrderedDict"): collections.OrderedDict, ("collections", "defaultdict"): collections.defaultdict,
             ("__builtin__", "dict"): dict, ("builtins", "dict"): dict}

    def find_class(self, module, name):
        if (module, name) in self._REAL:
            return self._REAL[(module, name)]
        if (module, name) in {("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"),
                              ("torch", "FloatStorage"), ("torch", "LongStorage")}:
            return super().find_class(module, name)
        if name == "getattr":
            return lambda *a: None
        return _Stub  # optimiser and scheduler classes: never constructed for real


class _RestrictedPickle:
    Unpickler = _RestrictedUnpickler
    load = staticmethod(pickle.load)
    __name__ = "pickle"


def load_model(cell: str, ckpt: Path, device) -> LegNet:
    cfg = json.loads((WEIGHTS / cell / "config.json").read_text())
    assert not cfg["use_reverse_channel"]
    m = LegNet(in_ch=4, stem_ch=cfg["stem_ch"], stem_ks=cfg["stem_ks"], ef_ks=cfg["ef_ks"],
               ef_block_sizes=cfg["ef_block_sizes"], resize_factor=cfg["resize_factor"], pool_sizes=cfg["pool_sizes"])
    # The checkpoints also hold optimiser and scheduler state, which the safe loader refuses by default.
    # Check (without executing anything) that the file names only the expected objects, then allow exactly those.
    with zipfile.ZipFile(ckpt) as z:
        pkl = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    ops = list(pickletools.genops(pkl))
    assert not any(op.name in ("STACK_GLOBAL", "INST", "OBJ") for op, _, _ in ops), f"{ckpt}: unexpected pickle instruction"
    named = {arg for op, arg, _ in ops if op.name == "GLOBAL"}
    assert named <= _EXPECTED, f"{ckpt}: unexpected objects {named - _EXPECTED}"
    state = torch.load(ckpt, map_location="cpu", weights_only=False, pickle_module=_RestrictedPickle)["state_dict"]
    m.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}, strict=True)
    return m.to(device).eval()


def predictor(models: list, device):
    @torch.no_grad()
    def fn(codes: np.ndarray, batch: int = 1024) -> np.ndarray:
        out = np.zeros(len(codes), dtype=np.float64)
        for i in range(0, len(codes), batch):
            c = torch.from_numpy(codes[i : i + batch].astype(np.int64)).to(device)
            rc = (3 - c).flip(1)
            for m in models:
                for x in (c, rc):  # the authors average the piece and its reverse complement
                    oh = torch.nn.functional.one_hot(x, 4).float()[..., _SWAP].permute(0, 2, 1)
                    out[i : i + batch] += m(oh).float().cpu().numpy() / (2 * len(models))
        return out
    return fn


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cells = sorted(set(MAIN.values()))
    single = {c: [load_model(c, WEIGHTS / c / MAIN_MODEL, device)] for c in cells}
    if "--sanity" in sys.argv:
        for c in cells:
            va = load(c, [VALIDATION_FOLD])
            r = element_pearson(va, predictor(single[c], device)(encode(va.seq)))
            n = sum(p.numel() for p in single[c][0].parameters())
            print(f"{c}: {MAIN_MODEL}, {n:,} parameters, fold {VALIDATION_FOLD} element Pearson {r:.3f}")
        return
    if "<!-- reference: MPRALegNet -->" not in PREDECLARATION.read_text():
        sys.exit("Stopped: record the reference run in the pre-declaration first.")
    every = {c: [load_model(c, f, device) for f in sorted((WEIGHTS / c).glob("best_model_test10_val*.ckpt"))] for c in cells}
    assert all(len(v) == 9 for v in every.values())
    rng = np.random.default_rng(SETTINGS["seed"])
    res = {"parameters": sum(p.numel() for p in single[cells[0]][0].parameters()), "switches": {}, "repeat_experiments": {}}
    boot_diff = []
    for group, key in ((MAIN, "switches"), (REPEATS, "repeat_experiments")):
        for name, cell in group.items():
            d = table(name)
            ref, new = pieces(d.Chromosome[0], d.Position.to_numpy(), d.alt_code.to_numpy())
            d["legnet"] = effect(predictor(single[cell], device), ref, new)
            d["legnet_9"] = effect(predictor(every[cell], device), ref, new)
            ours = pd.read_parquet(OUT / f"predictions_{name}.parquet")
            assert (ours.Position.to_numpy() == d.Position.to_numpy()).all() and (ours.Alt.to_numpy() == d.Alt.to_numpy()).all()
            d["student"] = ours.student.to_numpy()
            d.drop(columns="alt_code").to_parquet(OUT / f"legnet_predictions_{name}.parquet", index=False)
            s, dele = d[d.alt_code >= 0], d[d.alt_code < 0]
            y, a, b = s.Value.to_numpy(), s.student.to_numpy(), s.legnet.to_numpy()
            r = lambda u, v: float(np.corrcoef(u, v)[0, 1])
            diffs = []
            for _ in range(N_BOOT):  # resample changes, the same resample for both models
                i = rng.integers(0, len(y), len(y))
                diffs.append(r(y[i], a[i]) - r(y[i], b[i]))
            if key == "switches":
                boot_diff.append(diffs)
            res[key][name] = {"cell": cell, "changes": len(s), "student": r(y, a), "legnet": r(y, b),
                              "legnet_9_models": r(y, s.legnet_9), "student_minus_legnet_ci95": ci(diffs),
                              "deletions_student": r(dele.Value, dele.student), "deletions_legnet": r(dele.Value, dele.legnet)}
    sw = res["switches"]
    for k in ("student", "legnet", "legnet_9_models"):
        res[f"mean_{k}"] = float(np.mean([v[k] for v in sw.values()]))
    res["mean_student_minus_legnet"] = res["mean_student"] - res["mean_legnet"]
    res["mean_student_minus_legnet_ci95"] = ci(np.mean(boot_diff, axis=0))
    lo, hi = res["mean_student_minus_legnet_ci95"]
    res["reading"] = "student ahead" if lo > 0 else "student behind" if hi < 0 else "no clear difference"
    (OUT / "legnet_reference.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

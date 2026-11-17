"""Tests of src/train_model3.py. Run from the repository root (pytest is not needed):

    .venv/bin/python src/test_train_model3.py            # unit tests of the maths + smoke runs of the three modes + equality check
    .venv/bin/python src/test_train_model3.py --fast     # unit tests only
Fake data are written to a temporary folder in the same layout as data/model3 (make_fake); nothing in data/ is touched.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
import train_model3 as m3  # noqa: E402

PY = sys.executable
CELLS3 = ["HEPG2", "K562", "SKNSH"]
ALL5 = CELLS3 + ["HCT116", "A549"]


# ------------------------------- unit tests (hand-computed values) -------------------------------

def test_blend():
    # y = 1 (error variance 0.25), t = 3 (error variance 1): (1/0.25 + 3/1) / (1/0.25 + 1/1) = 7 / 5
    assert np.isclose(m3.blend(1.0, 3.0, 0.25, 1.0), 1.4)
    assert np.isclose(m3.blend(1.0, 3.0, 1.0, 1.0), 2.0)  # equal variances: the plain mean


def test_gate():
    # sqrt(0.25 + 1) = 1.1180; k = 3 -> 3.354
    assert not m3.gated(1.0, 3.0, 0.25, 1.0, 3.0)  # |y - t| = 2
    assert m3.gated(0.0, 4.0, 0.25, 1.0, 3.0)  # 4 > 3.354
    t = m3.make_target(np.array([0.0, 1.0]), np.array([4.0, 3.0]), 0.25, 1.0, gate_k=3.0)
    assert np.allclose(t, [0.0, 1.4]), t  # gated row: the label alone; the other row: the blend
    t = m3.make_target(np.array([0.0, 1.0]), np.array([4.0, 3.0]), 0.25, 1.0, gate_k=0.0)
    assert np.allclose(t, [(0 + 4.0) / 5, 1.4])  # no gate (P1): (0/0.25 + 4/1) / 5 = 0.8


def test_disagreement():
    d = m3.disagreement(np.array([1.0, 0.2]), np.array([-0.5, 0.2]))
    assert np.allclose(d, [1.5, 0.0])
    st2 = 0.04 * (1 + d ** 2)  # 0.04 * 3.25 = 0.13 ; 0.04
    assert np.allclose(st2, [0.13, 0.04])
    # inverse-variance blend with that st2: sy2 = 0.01, y = 0, t = 1 -> weight of t = (1/0.13) / (1/0.01 + 1/0.13)
    assert np.isclose(m3.blend(0.0, 1.0, 0.01, 0.13), (1 / 0.13) / (100 + 1 / 0.13))
    assert np.isclose(1 / (1 + 1.5 ** 2), 0.3076923076923077)  # the mutated-copy down-weight for d = 1.5


def test_make_target_missing():
    nan = np.nan
    y = np.array([1.0, nan, 1.0, nan])
    t = np.array([3.0, 3.0, nan, nan])
    out = m3.make_target(y, t, 0.25, 1.0)
    assert np.allclose(out[[0, 2]], [1.4, 1.0]) and np.isnan(out[1]) and np.isnan(out[3])  # no label: no loss; no teacher: label only
    out = m3.make_target(y, t, 0.25, 1.0, teacher_unlabelled=True)
    assert np.isclose(out[1], 3.0) and np.isnan(out[3])


def test_mix_and_calibrate():
    assert np.isclose(m3.mix_teachers(np.array(1.0), np.array(-1.0), 3, 1), 0.5)  # (3 * 1 + 1 * -1) / 4
    m = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = 2 + 3 * m
    a, b = m3.calibrate(y, m)
    assert np.allclose([a[0], b[0]], [2.0, 3.0])
    y2 = y.copy()
    y2[1, 0] = np.nan  # a missing label is skipped
    a, b = m3.calibrate(y2, m)
    assert np.allclose([a[0], b[0]], [2.0, 3.0])


def test_error_variance():
    y = np.array([[1.0], [0.0], [1.0], [0.0]])
    t = np.zeros((4, 1)) + 0.5  # label - teacher = +-0.5 -> variance 0.25
    sy2 = np.full((4, 1), 0.05)
    st2, raw = m3.teacher_error_variance(y, t, sy2)
    assert np.isclose(raw[0], 0.20) and np.isclose(st2[0], 0.20)  # 0.25 - 0.05
    st2, raw = m3.teacher_error_variance(y, t, np.full((4, 1), 0.4))
    assert np.isclose(raw[0], -0.15) and np.isclose(st2[0], 0.01)  # floored at 0.01


def test_weighted_masked_mse():
    pred = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 5.0]])
    target = torch.tensor([[0.0, float("nan")], [0.0, float("nan")], [float("nan"), 3.0]])
    # output 0: errors 1, 4 -> mean 2.5 ; output 1: error 4 -> 4 ; mean over outputs 3.25
    assert np.isclose(m3.weighted_masked_mse(pred, target, torch.ones(3, 2)).item(), 3.25)
    w = torch.tensor([[1.0, 1.0], [0.5, 1.0], [1.0, 0.25]])
    # output 0: (1 * 1 + 0.5 * 4) / 2 = 1.5 ; output 1: 0.25 * 4 / 1 = 1 ; mean 1.25
    assert np.isclose(m3.weighted_masked_mse(pred, target, w).item(), 1.25)
    from train_more_data import masked_mse
    assert np.isclose(m3.weighted_masked_mse(pred, target, torch.ones(3, 2)).item(), masked_mse(pred, target).item())


def test_pipeline_against_independent_calculation():
    """build_teacher_targets on 4 rows x 3 cells (the cells repeat the same numbers), checked against a plain re-derivation."""
    ag = np.tile(np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32), (1, 3))
    mp = np.tile(np.array([[0.0], [2.0], [1.0], [3.0]], dtype=np.float32), (1, 3))
    y = np.tile(np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32), (1, 3))
    sy = np.full((4, 3), 0.2, dtype=np.float32)
    args = Namespace(w_ag=3.0, w_mpac=1.0, gate_k=3.0, teacher_unlabelled=False)
    zs = lambda v: (v - v.mean()) / v.std()
    zA, zM = zs(ag[:, 0].astype(np.float64)), zs(mp[:, 0].astype(np.float64))
    mix = (3 * zA + zM) / 4
    b = np.mean((mix - mix.mean()) * (y[:, 0] - y[:, 0].mean())) / np.var(mix)
    a = y[:, 0].mean() - b * mix.mean()
    t = a + b * mix
    sy2 = 0.04
    st2 = max(np.var(y[:, 0] - t) - sy2, 0.01)
    d = np.abs(zA - zM)
    out1 = m3.build_teacher_targets("P1", y, sy, {"ag": ag, "mpac": mp}, None, args)
    exp1 = (y[:, 0] / sy2 + t / st2) / (1 / sy2 + 1 / st2)
    assert np.allclose(out1["target"][:, 0], exp1, atol=1e-5), (out1["target"][:, 0], exp1)
    assert np.allclose(out1["info"]["calibration_slope"], b) and np.allclose(out1["info"]["st2"], st2)
    out2 = m3.build_teacher_targets("P2", y, sy, {"ag": ag, "mpac": mp}, None, args)
    st2r = st2 * (1 + d ** 2)
    exp2 = (y[:, 0] / sy2 + t / st2r) / (1 / sy2 + 1 / st2r)
    exp2 = np.where(np.abs(y[:, 0] - t) > 3 * np.sqrt(sy2 + st2r), y[:, 0], exp2)
    assert np.allclose(out2["target"][:, 0], exp2, atol=1e-5)
    out0 = m3.build_teacher_targets("P0", y, sy, {"ag": ag, "mpac": mp}, None, args)
    assert np.allclose(out0["t_labelled"][:, 0], t, atol=1e-5)
    # mutated copies: same parameters; P2 weight 1 / (1 + d^2)
    mu = {"ag": ag[::-1].copy(), "mpac": mp[::-1].copy()}
    outm = m3.build_teacher_targets("P2", y, sy, {"ag": ag, "mpac": mp}, mu, args)
    mA, mM = (mu["ag"][:, 0] - ag[:, 0].mean()) / ag[:, 0].std(), (mu["mpac"][:, 0] - mp[:, 0].mean()) / mp[:, 0].std()
    assert np.allclose(outm["t_mut"][:, 0], a + b * (3 * mA + mM) / 4, atol=1e-5)
    assert np.allclose(outm["w_mut"][:, 0], 1 / (1 + (mA - mM) ** 2), atol=1e-5)


def test_mutant_stream():
    s = m3.MutantStream(10, torch.Generator().manual_seed(0))
    got = torch.cat([s.take(4), s.take(4), s.take(4)])  # 12 rows from 10: wraps into a new permutation
    assert len(got) == 12 and sorted(got[:10].tolist()) == list(range(10))


# ------------------------------- fake data in the layout of data/model3 -------------------------------

def make_fake(d: Path, n_var: int = 1500, n_prac: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    L = np.array(list("ACGT"))

    def pieces(n):
        s = rng.integers(0, 4, (n, 200))
        s2 = s.copy()
        s2[:, 100] = (s[:, 100] + rng.integers(1, 4, n)) % 4
        return s, s2

    def to_str(a):
        return ["".join(L[r]) for r in a]

    def motif(a, k):  # activity grows with the number of letter k at 20 positions
        return (a[:, 60:140] == k).sum(1)

    def table(n, chrom):
        s1, s2 = pieces(n)
        df = pd.DataFrame({"variant": [f"chr{chrom}:{i}:A:C" for i in range(n)], "chr": str(chrom), "pos": np.arange(n) * 1000 + 5000,
                           "seq1": to_str(s1), "seq2": to_str(s2), "part": "train"})
        act = {}
        for j, c in enumerate(ALL5):
            for v, s in ((1, s1), (2, s2)):
                col = 0.15 * (motif(s, j % 4) - 20) + rng.normal(0, 0.3, n)
                col = np.where(rng.random(n) < (0.03 if c in CELLS3 else 0.4), np.nan, col)
                df[f"allele{v}.log2FC_{c}"] = col
                df[f"allele{v}.log2FC.SE_{c}"] = np.where(np.isnan(col), np.nan, 0.1 + 0.2 * rng.random(n))
                act[(c, v)] = col
            df[f"log2Skew_{c}"] = act[(c, 2)] - act[(c, 1)] + rng.normal(0, 0.1, n)
            df[f"log2Skew.SE_{c}"] = 0.2
        return df, s1, s2, act

    tr, s1, s2, act = table(n_var, 1)
    pr, _, _, _ = table(n_prac, 13)
    tr.to_parquet(d / "train.parquet")
    pr.to_parquet(d / "practice13.parquet")
    pr.to_parquet(d / "practice.parquet")  # the name train_more_data.py reads (equality check)
    pd.DataFrame({"seq": []}).to_parquet(d / "lentimpra_excluded.parquet")
    # mutated copies: the same 10 positions in both versions
    m1, m2 = s1.copy(), s2.copy()
    for i in range(n_var):
        pos = rng.choice(200, 10, replace=False)
        sub = rng.integers(1, 4, 10)
        m1[i, pos] = (s1[i, pos] + sub) % 4
        m2[i, pos] = (s2[i, pos] + sub) % 4
    pd.DataFrame({"seq1_mut": to_str(m1), "seq2_mut": to_str(m2)}).to_parquet(d / "mutants.parquet")
    names = {"HEPG2": "HepG2", "K562": "K562", "SKNSH": "SKNSH"}
    ag, mp, agm, mpm = {}, {}, {}, {}
    for c in CELLS3:
        for v, s, m in ((1, s1, m1), (2, s2, m2)):
            truth = 0.15 * (motif(s, ALL5.index(c) % 4) - 20)
            truthm = 0.15 * (motif(m, ALL5.index(c) % 4) - 20)
            k = f"{names[c]}_v{v}"
            ag[k] = (truth + rng.normal(0, 0.2, n_var)).astype(np.float32)
            mp[k] = (2 * truth + 1 + rng.normal(0, 0.4, n_var)).astype(np.float32)  # another scale: z-scoring must handle it
            agm[k] = (truthm + rng.normal(0, 0.2, n_var)).astype(np.float32)
            mpm[k + "_mut"] = (2 * truthm + 1 + rng.normal(0, 0.4, n_var)).astype(np.float32)
            ag[k][rng.random(n_var) < 0.01] = np.nan  # a few rows without a teacher
    np.savez(d / "teacher_ag_train.npz", **ag)
    np.savez(d / "teacher_mpac.npz", **mp, **mpm)
    np.savez(d / "teacher_ag_mutants.npz", **agm)


def run_m3(d: Path, *extra: str) -> dict:
    cmd = [PY, "src/train_model3.py", "--data_dir", str(d), "--epochs", "1", "--max_steps", "30", "--batch", "512", "--mut_batch", "256", "--tag", "unittest", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    mode = extra[extra.index("--mode") + 1]
    seed = extra[extra.index("--seed") + 1] if "--seed" in extra else "20261113"
    run = Path("runs") / f"m3_{mode}_e1_seed{seed}_unittest"
    out = json.loads((run / "log.json").read_text())
    shutil.rmtree(run)
    return out | {"stdout": p.stdout}


def test_smoke_runs(d: Path):
    for mode in ("P0", "P1", "P2"):
        out = run_m3(d, "--mode", mode, "--lenti_excluded", "none")
        e = out["log"][0]
        assert np.isfinite(e["train_loss"]) and "practice13_mean_effect" in e and "practice13_mean_activity" in e
        cfg = out["config"]
        assert "calibration_slope" in cfg["teacher"] and len(cfg["teacher"]["st2"]) == 3
        print(f"  smoke {mode}: loss {e['train_loss']:.4f}, label {e['label_loss']:.4f}, teacher {e.get('teacher_loss')}, mutants {e.get('mutant_loss')}, "
              f"chr13 effect {e['practice13_mean_effect']:.3f}, gated {cfg['teacher'].get('gated_share')}, "
              f"mean mut weight {cfg['teacher'].get('mut_mean_weight')}")
        assert (mode == "P0") == ("mutant_loss" not in e) and (mode == "P0") == ("teacher_loss" in e)
    # the width option builds the 4M body (a full wide smoke run is slow on a laptop GPU that is shared; run with --wide)
    from malinois_body import build_arch
    assert sum(p.numel() for p in build_arch("legnet", 8, width=1.75).parameters()) > 3_900_000
    if "--wide" in sys.argv:
        out = run_m3(d, "--mode", "P1", "--lenti_excluded", "none", "--width", "1.75", "--seed", "2")
        assert out["config"]["parameters"] > 3_000_000


def test_equals_plain_label_training(d: Path):
    """P0 with lam = 0 and no mutated copies: the first-step loss equals train_more_data.py's (plain label masked MSE)."""
    code = ("import sys; sys.argv=['train_more_data.py','--epochs','1','--max_steps','1','--batch','512','--tag','eqref']; sys.path.insert(0,'src');"
            f"import train_more_data as t; from pathlib import Path; t.MORE=Path({str(d)!r}); t.main()")
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    run = Path("runs/more_HEPG2-K562-SKNSH-HCT116-A549_e1_eqref_seed20261113")
    b = json.loads((run / "log.json").read_text())["log"][0]["train_loss"]
    shutil.rmtree(run)
    cmd = [PY, "src/train_model3.py", "--data_dir", str(d), "--epochs", "1", "--max_steps", "1", "--batch", "512", "--tag", "eq1", "--mode", "P0", "--lam", "0",
           "--mut_weight", "0", "--lenti_excluded", "none"]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    run = Path("runs/m3_P0_e1_seed20261113_eq1")
    a1 = json.loads((run / "log.json").read_text())["log"][0]["train_loss"]
    shutil.rmtree(run)
    print(f"  first-step loss: model 3 P0 (lam 0) {a1:.6f}, plain label training (train_more_data.py) {b:.6f}")
    assert abs(a1 - b) < 1e-5, (a1, b)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and fn.__code__.co_argcount == 0:
            fn()
            print("ok", name)
    if "--fast" not in sys.argv:
        tmp = Path(tempfile.mkdtemp())
        try:
            make_fake(tmp)
            test_smoke_runs(tmp)
            print("ok test_smoke_runs")
            test_equals_plain_label_training(tmp)
            print("ok test_equals_plain_label_training")
        finally:
            shutil.rmtree(tmp)

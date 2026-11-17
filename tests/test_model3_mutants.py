"""Invariants of src/model3_mutants.py. Run: ~/PurvoLabs/small-dna-model/.venv/bin/python -I tests/test_model3_mutants.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model3_mutants as mm  # noqa: E402


def make(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 4, size=(n, 200), dtype=np.uint8)
    b = a.copy()
    pos = rng.integers(0, 200, size=n)
    b[np.arange(n), pos] = (a[np.arange(n), pos] + rng.integers(1, 4, size=n)) % 4
    return mm.to_strings(a), mm.to_strings(b), a, b, pos


def test_invariants():
    s1, s2, a, b, pos = make()
    am, bm, mask = mm.mutate(s1, s2)
    n = len(s1)
    assert (mask.sum(1) == 10).all()                                  # exactly 10 of 200 positions
    assert not mask[np.arange(n), pos].any()                          # never the variant position
    assert ((am != a) == mask).all() and ((bm != b) == mask).all()    # changed exactly where masked, always to a different letter
    d = am != bm
    assert (d.sum(1) == 1).all() and d[np.arange(n), pos].all()       # copies differ only at the variant position
    assert (am[np.arange(n), pos] == a[np.arange(n), pos]).all()      # the variant letters are untouched
    assert (bm[np.arange(n), pos] == b[np.arange(n), pos]).all()
    assert (am[mask] == bm[mask]).all()                               # the same new letters in both versions
    assert am.max() <= 3 and am.dtype == np.uint8


def test_letters_roughly_uniform_and_positions_spread():
    s1, s2, a, b, pos = make(20000)
    am, bm, mask = mm.mutate(s1, s2)
    new = am[mask]; old = a[mask]
    shift = (new.astype(int) - old.astype(int)) % 4
    freq = np.bincount(shift, minlength=4) / len(shift)
    assert freq[0] == 0 and (np.abs(freq[1:] - 1 / 3) < 0.01).all()   # each of the 3 other letters about 1/3
    per_pos = mask.mean(0)
    assert per_pos.min() > 0.04 and per_pos.max() < 0.06              # about 5% at every position


def test_deterministic_and_string_roundtrip():
    s1, s2, *_ = make(300)
    x = mm.mutate(s1, s2)
    y = mm.mutate(s1, s2)
    assert all((p == q).all() for p, q in zip(x, y))
    z = mm.mutate(s1, s2, seed=1)
    assert not (x[0] == z[0]).all()
    out = mm.to_strings(x[0])
    assert all(len(s) == 200 and set(s) <= set("ACGT") for s in out)
    assert mm.to_codes(out).tolist() == x[0].tolist()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

"""Checks for src/pyramid_legnet.py. Run from the repository root:

    .venv/bin/python src/test_pyramid_legnet.py            # all checks + parameter table
    .venv/bin/python src/test_pyramid_legnet.py --speed    # also time training steps against LegNet on this machine
"""
import sys
import time
from collections import OrderedDict

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "src/third_party")
from malinois_body import build_arch  # noqa: E402
from pyramid_legnet import PRESETS, PyramidLegNet, build_pyramid, parameter_breakdown  # noqa: E402

torch.manual_seed(0)


def random_onehot(b, length=230):
    codes = torch.randint(0, 4, (b, length))
    return torch.nn.functional.one_hot(codes, 4).float().transpose(1, 2)


def test_off_equals_legnet():
    """All changes off: same parameter count as LegNet and identical outputs with LegNet's weights copied in."""
    leg = build_arch("legnet", 8)
    ours = build_pyramid("pyr_off", 8)
    n_leg = sum(p.numel() for p in leg.parameters())
    n_ours = sum(p.numel() for p in ours.parameters())
    assert n_leg == n_ours == 1_324_264, (n_leg, n_ours)
    s_leg, s_ours = leg.state_dict(), ours.state_dict()
    assert len(s_leg) == len(s_ours)
    for (k1, v1), (k2, v2) in zip(s_leg.items(), s_ours.items()):
        assert v1.shape == v2.shape, (k1, k2, v1.shape, v2.shape)
    ours.load_state_dict(OrderedDict(zip(s_ours.keys(), s_leg.values())))
    # give the BatchNorm layers non-trivial statistics first, so eval mode is a real test
    leg.train()
    with torch.no_grad():
        leg(random_onehot(64))
    ours.load_state_dict(OrderedDict(zip(s_ours.keys(), leg.state_dict().values())))
    leg.eval(), ours.eval()
    x = random_onehot(32)
    with torch.no_grad():
        a, b = leg(x), ours(x)
    assert a.shape == b.shape == (32, 8)
    assert torch.equal(a, b), (a - b).abs().max()
    print("ok  all changes off == LegNet: 1,324,264 parameters, identical outputs with LegNet's weights")
    full = build_pyramid("pyramid", 8)
    assert sum(p.numel() for p in full.stages.parameters()) == sum(p.numel() for p in leg.main.parameters()) == 1_219_680
    print("ok  the full preset keeps LegNet's trunk unchanged (1,219,680 parameters in the four blocks)")


def test_presets_forward_backward():
    for name in PRESETS:
        m = build_pyramid(name, 8)
        x = random_onehot(16)
        y = m(x)
        assert y.shape == (16, 8) and torch.isfinite(y).all(), name
        y.pow(2).mean().backward()
        bad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        # an adapter's `down` gets no gradient on the very first step (its `up` starts at zero); everything else must
        bad = [n for n in bad if ".down." not in n]
        assert not bad, (name, bad[:5])
    print("ok  every preset: (16, 4, 230) -> (16, 8), finite outputs and gradients")


def test_adapters_start_as_identity():
    m = build_pyramid("pyramid", 8).eval()
    h = torch.randn(8, 256)
    for ad in m.adapters:
        assert torch.equal(ad(h), h)
    print("ok  adapters start as the identity")


def test_output_groups_and_lenti_only():
    m = build_pyramid("pyramid", 8)
    assert m.groups == [3, 5] and [o.out_features for o in m.outs] == [3, 5]
    m3 = build_pyramid("pyramid", 3)  # --new_cells none: lentiMPRA outputs only
    assert m3.groups == [3] and m3(random_onehot(4)).shape == (4, 3)
    print("ok  output order: 3 lentiMPRA columns then 5 plasmid columns; works with 3 outputs only")


def test_other_lengths_and_predict_multi():
    m = build_pyramid("pyramid", 8)
    for length in (200, 230, 251):
        assert m(random_onehot(4, length)).shape == (4, 8)
    from mpra_train_multi import predict_multi
    codes = torch.randint(0, 5, (10, 230), dtype=torch.uint8)  # includes N (code 4)
    p = predict_multi(m, codes, torch.device("cpu"))
    assert p.shape == (10, 8)
    print("ok  any input length; works with predict_multi (both strands, N letters)")


def table():
    print("\nparameters (8 outputs)")
    rows = []
    for name in PRESETS:
        b = parameter_breakdown(build_pyramid(name, 8))
        rows.append((name, b))
    parts = ["stem", "stages", "mapper", "proj", "head_shared", "out", "adapters", "outs"]
    print(f"{'preset':<12}" + "".join(f"{p:>12}" for p in parts) + f"{'total':>12}")
    for name, b in rows:
        print(f"{name:<12}" + "".join(f"{b.get(p, 0):>12,}" for p in parts) + f"{b['total']:>12,}")
    for wd in (1.25, 1.5):
        b = parameter_breakdown(PyramidLegNet(8, width=wd))
        print(f"pyramid w{wd:<5}" + "".join(f"{b.get(p, 0):>12,}" for p in parts) + f"{b['total']:>12,}")


def speed():
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\ntraining-step time, batch 2048 (as P1: 1024 pieces + 1024 mutated copies), device {dev}")
    base = None
    for name in ("legnet", "pyramid", "pyr_stem", "pyr_taps"):
        m = (build_arch("legnet", 8) if name == "legnet" else build_pyramid(name, 8)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x = random_onehot(2048).to(dev)
        for i in range(13):
            if i == 3:
                if dev.type == "mps":
                    torch.mps.synchronize()
                t0 = time.time()
            loss = m(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        dt = (time.time() - t0) / 10
        base = base or dt
        print(f"  {name:<10} {dt * 1000:7.0f} ms per step  ({dt / base:.2f}x LegNet)")


if __name__ == "__main__":
    test_off_equals_legnet()
    test_presets_forward_backward()
    test_adapters_start_as_identity()
    test_output_groups_and_lenti_only()
    test_other_lengths_and_predict_multi()
    table()
    if "--speed" in sys.argv:
        speed()

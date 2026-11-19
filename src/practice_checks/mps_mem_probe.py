"""Memory probe for src/train_model3.py on the Mac GPU (MPS). Runs the real script unchanged (via runpy) and, every --every
training steps, prints: live tensor memory (torch.mps.current_allocated_memory), everything the Metal driver holds for the
process including PyTorch's cache (torch.mps.driver_allocated_memory) and the process footprint (macOS `footprint`).
Optional --empty_every N calls torch.mps.empty_cache() every N steps (to test the fix).
Run from the v2 repository root:
    .venv/bin/python src/practice_checks/mps_mem_probe.py --every 25 [--empty_every 0] -- --mode P1 --arch pyramid --max_steps 400 --tag memprobe
"""
import argparse
import os
import re
import runpy
import subprocess
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=25)
ap.add_argument("--empty_every", type=int, default=0)
ap.add_argument("--short_epochs", type=int, default=0, help="cut each epoch to this many full batches plus the real (short) last batch")
ap.add_argument("--empty_after_eval", action="store_true", help="torch.mps.empty_cache() after each epoch-end evaluation (the fix to test)")
own, _, rest = " ".join(sys.argv[1:]).partition(" -- ")
args = ap.parse_args(own.split())
assert "--tag" in rest, "always give --tag, so the real run folder is never written"

pid = str(os.getpid())
t0 = time.time()


def footprint_gb() -> float:
    out = subprocess.run(["footprint", pid], capture_output=True, text=True).stdout
    m = re.search(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB)", out)
    if not m:
        return float("nan")
    return float(m.group(1)) / {"KB": 2**20, "MB": 2**10, "GB": 1}[m.group(2)]


def report(label: str) -> None:
    torch.mps.synchronize()
    print(f"[mem] {label:<14} t {time.time() - t0:6.0f}s  live {torch.mps.current_allocated_memory() / 2**30:6.3f} GB  "
          f"driver {torch.mps.driver_allocated_memory() / 2**30:6.3f} GB  footprint {footprint_gb():6.3f} GB", flush=True)


step = 0
orig_clip = torch.nn.utils.clip_grad_norm_


def counted_clip(*a, **k):  # called exactly once per training step by train_model3.py
    global step
    r = orig_clip(*a, **k)
    step += 1
    if step in (1, 2, 5, 10) or step % args.every == 0:
        report(f"step {step}")
    if args.empty_every and step % args.empty_every == 0:
        torch.mps.empty_cache()
    return r


torch.nn.utils.clip_grad_norm_ = counted_clip

sys.path[:0] = ["src", "src/third_party"]
import mpra_train_multi  # noqa: E402  (patch before train_model3 imports it by name)

orig_predict = mpra_train_multi.predict_multi
n_eval = 0


def counted_predict(model, codes, device, batch=1024):
    global n_eval
    r = orig_predict(model, codes, device, batch)
    n_eval += 1
    if n_eval % 7 == 0:  # one epoch-end evaluation = 3 fold-9 calls + 4 chr13 calls
        report(f"eval {n_eval // 7}")
        if args.empty_after_eval:
            torch.mps.empty_cache()
            report(f"emptied {n_eval // 7}")
    return r


TOTAL_ROWS = 735_086  # lentiMPRA rows + both versions of every training variant (train_model3.py prints it)
orig_randperm = torch.randperm


def short_randperm(n, *a, **k):  # only the main epoch order (n == TOTAL_ROWS) is cut; the mutated-copy stream is untouched
    p = orig_randperm(n, *a, **k)
    if args.short_epochs and n == TOTAL_ROWS:
        return p[: args.short_epochs * 1024 + TOTAL_ROWS % 1024]
    return p


torch.randperm = short_randperm


mpra_train_multi.predict_multi = counted_predict
report("start")
sys.argv = ["src/train_model3.py"] + rest.split()
runpy.run_path("src/train_model3.py", run_name="__main__")
report("end")

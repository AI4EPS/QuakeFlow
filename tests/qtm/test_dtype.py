# %%
"""
Does the normalized correlation need float64?

`CCTorch/cctorch/model.py` upcast unconditionally to float64 whenever --normalize was set,
which is every template-matching run. That is not free: the depthwise conv measures
1.32 TFLOP/s in float64 against a 1.29 TFLOP/s FP64 roofline on an RTX 4090, i.e. it is
already at the hardware limit and cannot be tuned further, while the same conv in float32
runs 4x faster. So the precision is the single largest lever on QTM runtime, and the
question is only what it costs in accuracy.

Two normalizations happen, and only one of them can be responsible:

  template  (model.py:124-126)  data2 is demeaned and divided by its own std over the mask.
                                Scale-invariant -- the amplitude template.dat was cut at
                                cancels exactly, so normalizing at cut time changes nothing.
  cc range  (model.py:127-128,  data1 is demeaned/unit-std over the whole trace, then a
             142-144)           running std per window rescales cc into [-1, 1]. The running
                                variance is sum2/n - (sum1/n)^2, which cancels.

Mixed-precision runs (see decompose()) separate them: the loss is in the xcor accumulation,
not the variance. A 400-tap dot product in float32 carries ~sqrt(nt2)*eps ~ 2e-6 of relative
error, which is what comes out.

    python tests/qtm/test_dtype.py --region synthetic
    python tests/qtm/test_dtype.py --region demo
    python tests/qtm/test_dtype.py --region redwood --skip_pipeline   # score what is on disk

Exit code is 0 if float32 tracks float64 to --tolerance on the detections both find.
"""

import argparse
import shutil
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from paths import DEFAULT_ROOT, SCRIPTS

DTYPES = ("float64", "float32")
KEY = ["idx_eve", "idx_sta", "phase_type"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--region", type=str, default="synthetic")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="largest |cc(float32) - cc(float64)| accepted on a shared detection")
    parser.add_argument("--skip_pipeline", action="store_true",
                        help="do not re-run; score the ccpairs_{dtype} directories already saved")
    return parser.parse_args()


def run_qtm(args, dtype):
    """Run the scan at this precision and park the output where compare() can find it."""
    base = Path(args.root_path) / args.region / "qtm"
    shutil.rmtree(base / "ccpairs", ignore_errors=True)
    t0 = time.time()
    out = subprocess.run([sys.executable, "run_qtm.py", args.root_path, args.region, "--dtype", dtype],
                         capture_output=True, text=True, cwd=SCRIPTS)
    elapsed = time.time() - t0
    if out.returncode != 0:
        print(out.stdout[-2000:], out.stderr[-2000:])
        raise SystemExit(f"run_qtm.py --dtype {dtype} failed with {out.returncode}")
    saved = base / f"ccpairs_{dtype}"
    shutil.rmtree(saved, ignore_errors=True)
    shutil.copytree(base / "ccpairs", saved)
    return elapsed


def load(args, dtype):
    files = sorted(glob(f"{args.root_path}/{args.region}/qtm/ccpairs_{dtype}/TM_???_???.csv"))
    if not files:
        raise SystemExit(f"no ccpairs_{dtype} for {args.region}; run without --skip_pipeline")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ## one row per template phase -- a template may peak more than once in the day, and the
    ## two precisions need not rank near-equal peaks the same way
    return df.loc[df.groupby(KEY)["cc"].idxmax()].set_index(KEY)


def compare(args, elapsed):
    a, b = (load(args, d) for d in DTYPES)
    shared = a.index.intersection(b.index)
    d = (a.loc[shared, "cc"] - b.loc[shared, "cc"]).abs()

    print(f"\n  {'':22s} {DTYPES[0]:>14s} {DTYPES[1]:>14s}")
    print(f"  {'detections':22s} {len(a):>14,} {len(b):>14,}")
    print(f"  {'max cc':22s} {a['cc'].max():>14.6f} {b['cc'].max():>14.6f}")
    print(f"  {'cc > 0.99':22s} {(a['cc'] > 0.99).sum():>14,} {(b['cc'] > 0.99).sum():>14,}")
    if all(elapsed.get(d) for d in DTYPES):
        print(f"  {'wall time (min)':22s}" + "".join(f"{elapsed[d] / 60:>15.1f}" for d in DTYPES))
        print(f"  {'speedup':22s} {'':>14s} {elapsed[DTYPES[0]] / elapsed[DTYPES[1]]:>14.2f}x")
    print(f"\n  shared detections      {len(shared):,} of {len(a):,} / {len(b):,}"
          f"   ({len(shared) / max(len(a), 1):.2%} / {len(shared) / max(len(b), 1):.2%})")
    if len(shared):
        print(f"  |cc32 - cc64|          max {d.max():.3e}   median {d.median():.3e}"
              f"   above {args.tolerance:g}: {(d > args.tolerance).sum():,}")
        worst = d.nlargest(5)
        print("  largest disagreements:")
        for k, v in worst.items():
            print(f"    idx_eve {k[0]:>6} idx_sta {k[1]:>4} {k[2]}  "
                  f"cc64 {a.loc[k, 'cc']:.6f}  cc32 {b.loc[k, 'cc']:.6f}  d {v:.2e}")

    ## detections one precision found and the other did not: these sit on the threshold, so a
    ## few are expected and only a large asymmetry would mean something
    for x, y, dx, dy in ((a, b, *DTYPES), (b, a, *DTYPES[::-1])):
        only = x.index.difference(y.index)
        if len(only):
            print(f"  {dx} only: {len(only):,}  (cc {x.loc[only, 'cc'].min():.4f}"
                  f" - {x.loc[only, 'cc'].max():.4f}, median {x.loc[only, 'cc'].median():.4f})")

    ok = len(shared) > 0 and d.max() <= args.tolerance
    print(f"\n  {'PASS' if ok else 'FAIL'}: float32 tracks float64 to {args.tolerance:g}"
          if ok else f"\n  FAIL: {(d > args.tolerance).sum()} detections differ by more than {args.tolerance:g}")
    return ok


# %%
if __name__ == "__main__":
    args = parse_args()
    elapsed = {}
    if not args.skip_pipeline:
        for dtype in DTYPES:
            print(f"=== {args.region} --dtype {dtype} ===", flush=True)
            elapsed[dtype] = run_qtm(args, dtype)
            print(f"  {elapsed[dtype] / 60:.1f} min", flush=True)
    sys.exit(0 if compare(args, elapsed) else 1)

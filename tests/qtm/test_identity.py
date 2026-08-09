# %%
"""
Regression test: a template correlated against the waveform it was cut from must return
cc = 1, at every station, for both phases, at any amplitude.

The template block of the benchmark is injected from the same samples the templates are
later cut out of, so that correlation is a waveform against itself.  Its cc does not depend
on SNR -- scaling the injection scales the template with it, and the normalised correlation
is unchanged.  So it must hold at every amplitude alike, and any drop is a processing
difference between the template side and the continuous side, never a detection limit.
(Test-block recovery *is* SNR-dependent; it is reported but not asserted.)

    python tests/qtm/test_identity.py --region synthetic
    python tests/qtm/test_identity.py --region synthetic --template_ratio 10 1 0.1
    python tests/qtm/test_identity.py --region synthetic --skip_pipeline   # check what is on disk

Exit code is 0 if every amplitude passes, 1 otherwise.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path
import sys
from glob import glob

import numpy as np
import pandas as pd
from paths import DEFAULT_ROOT, SCRIPTS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--region", type=str, default="synthetic")
    parser.add_argument("--template_ratio", type=float, nargs="+", default=[10.0, 1.0, 0.1],
                        help="template-block SNRs to test; the identity check must pass at all of them. "
                             "Ordered low to high so the region is left at the default when it finishes")
    parser.add_argument("--test_ratio", type=float, nargs="+", default=[3.0],
                        help="passed straight through to make_benchmark.py for every template_ratio")
    parser.add_argument("--tolerance", type=float, default=2.0, help="origin-time tolerance for a match (s)")
    parser.add_argument("--min_cc", type=float, default=0.999, help="cc a self-match must reach")
    parser.add_argument("--skip_pipeline", action="store_true",
                        help="do not rebuild or re-run; check the output already on disk")
    return parser.parse_args()


def run(cmd, cwd=None):
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if out.returncode != 0:
        print(out.stdout[-2000:])
        print(out.stderr[-2000:])
        raise SystemExit(f"command failed with {out.returncode}: {' '.join(cmd)}")
    return out


def build_and_run(args, template_ratio):
    """Rebuild the benchmark at this amplitude and push it through the pipeline."""
    base = f"{args.root_path}/{args.region}"
    ## the workflow scripts live in scripts/ and resolve their own relative paths against
    ## it, so they are run from there; this file sits in tests/ and is invoked from anywhere
    here = Path(__file__).resolve().parent
    run([sys.executable, str(here / "make_benchmark.py"), "--root_path", args.root_path,
         "--region", args.region, "--template_ratio", str(template_ratio),
         "--test_ratio", *[str(r) for r in args.test_ratio]])
    run([sys.executable, "cut_templates_cc.py", "--root_path", args.root_path,
         "--region", args.region, "--mode", "TM"], cwd=SCRIPTS)

    ## run_qtm.py shells out to torchrun and returns 0 even when that dies, so clear the
    ## output first and insist it comes back -- otherwise a crashed run is scored as a pass
    ## against whatever the previous run left behind
    shutil.rmtree(f"{base}/qtm/ccpairs", ignore_errors=True)
    run([sys.executable, "run_qtm.py", args.root_path, args.region], cwd=SCRIPTS)
    produced = glob(f"{base}/qtm/ccpairs/TM_???_???.csv")
    if not produced or not any(os.path.getsize(f) > 0 for f in produced):
        raise SystemExit(f"run_qtm.py produced no correlations in {base}/qtm/ccpairs")


# %%
def check(args):
    """Every injected template-block phase, matched against its own template."""
    base = f"{args.root_path}/{args.region}"
    gt = pd.read_csv(f"{base}/benchmark/ground_truth_events.csv")
    gt["time"] = pd.to_datetime(gt["time"], utc=True, format="ISO8601")
    gp = pd.read_csv(f"{base}/benchmark/ground_truth_picks.csv")
    stations = pd.read_csv(f"{base}/qtm/template_stations.csv")
    cc_events = pd.read_csv(f"{base}/qtm/template_events.csv")

    tm = pd.concat([pd.read_csv(f, parse_dates=["origin_time"])
                    for f in sorted(glob(f"{base}/qtm/ccpairs/TM_???_???.csv"))], ignore_index=True)
    tm["origin_time"] = pd.to_datetime(tm["origin_time"], utc=True)
    tm = tm.merge(stations[["idx_sta", "station_id"]], on="idx_sta", how="left")
    tmpl2src = gt.dropna(subset=["template_event_index"]).drop_duplicates("template_event_index")
    tmpl2src = tmpl2src.set_index("template_event_index")["idx_event"]
    tm["src"] = tm["idx_eve"].map(cc_events.set_index("idx_eve")["event_index"]).map(tmpl2src)

    ## best cc from the matching template, near the true origin time
    hits = []
    for _, g in gt.iterrows():
        near = tm[(tm["src"] == g["idx_event"])
                  & ((tm["origin_time"] - g["time"]).abs() < pd.Timedelta(seconds=args.tolerance))]
        hits.append(near.assign(inj_id=g["inj_id"]))
    hits = pd.concat(hits, ignore_index=True)
    hits = hits.loc[hits.groupby(["inj_id", "station_id", "phase_type"])["cc"].idxmax()]

    ## only phases that actually had signal put into the record
    expected = gp[gp["injected"]]
    got = expected.merge(hits[["inj_id", "station_id", "phase_type", "cc"]],
                         on=["inj_id", "station_id", "phase_type"], how="left")

    res = {}
    for blk in ("template", "test"):
        b = got[got["block"] == blk]
        found = b[b["cc"].notna()]
        res[blk] = {
            "expected": len(b),
            "returned": len(found),
            "cc_min": float(found["cc"].min()) if len(found) else np.nan,
            "cc_median": float(found["cc"].median()) if len(found) else np.nan,
            "snr": float(gt.loc[gt["block"] == blk, "snr_z_median"].median()),
        }

    ## only the template block is asserted -- it is the waveform against itself, so it
    ## cannot depend on SNR. The test block is the same waveform in different noise, so its
    ## recovery is a measurement, not a requirement.
    ident = got[got["block"] == "template"]
    missing = ident[ident["cc"].isna()]
    low = ident[ident["cc"].notna() & (ident["cc"] < args.min_cc)]
    res["ok"] = len(missing) == 0 and len(low) == 0

    for blk in ("template", "test"):
        r = res[blk]
        print(f"  {blk:8s} block  SNR {r['snr']:<8.4g} {r['returned']}/{r['expected']} returned"
              f"  ({r['returned'] / max(r['expected'], 1):.1%})"
              + (f", cc min {r['cc_min']:.6f}, median {r['cc_median']:.6f}" if r["returned"] else "")
              + ("   [asserted]" if blk == "template" else "   [reported]"))
    if len(missing):
        print(f"  FAIL {len(missing)} template phases missing:")
        print(missing[["inj_id", "idx_event", "station_id", "phase_type", "snr"]].head(20).to_string(index=False))
    if len(low):
        print(f"  FAIL {len(low)} template phases below {args.min_cc}:")
        print(low[["inj_id", "idx_event", "station_id", "phase_type", "snr", "cc"]]
              .sort_values("cc").head(20).to_string(index=False))
    return res


# %%
if __name__ == "__main__":
    args = parse_args()
    results = []
    for template_ratio in args.template_ratio:
        print(f"\n=== template_ratio = {template_ratio:g} ===", flush=True)
        if not args.skip_pipeline:
            build_and_run(args, template_ratio)
        res = check(args)
        res["template_ratio"] = template_ratio
        results.append(res)
        print(f"  {'PASS' if res['ok'] else 'FAIL'}")

    print("\n" + "=" * 92)
    print("The two blocks are not comparable at equal SNR, by design:")
    print("  template = the waveform the template was cut from, so it correlates with itself.")
    print("            Noise and all, cc is 1 at ANY amplitude. This is the assertion.")
    print("  test     = the same event in DIFFERENT noise, so only the event part correlates")
    print("            and cc falls off as ~1/sqrt(1 + 1/SNR^2). This is the measurement.")
    print("A handful returned at test SNR << 1 are chance peaks inside the matching window,")
    print("not recoveries.")
    print("=" * 92)
    print(f"{'':>10}  {'template block (asserted)':^34}  {'test block (reported)':^34}")
    print(f"{'amplitude':>10}  {'SNR':>7} {'returned':>12} {'min cc':>11}  "
          f"{'SNR':>7} {'returned':>12} {'median cc':>11}  {'result':>6}")
    for r in results:
        t, x = r["template"], r["test"]
        t_ret = f"{t['returned']}/{t['expected']}"
        x_ret = f"{x['returned']}/{x['expected']}"
        print(f"{r['template_ratio']:>10g}  "
              f"{t['snr']:>7.4g} {t_ret:>12} {t['cc_min']:>11.6f}  "
              f"{x['snr']:>7.4g} {x_ret:>12} {x['cc_median']:>11.6f}  "
              f"{'PASS' if r['ok'] else 'FAIL':>6}")
    failed = [r["template_ratio"] for r in results if not r["ok"]]
    if failed:
        print(f"\nFAILED at template_ratio {failed}")
        sys.exit(1)
    print("\nall amplitudes pass: a template still correlates with its own waveform at 1.0")

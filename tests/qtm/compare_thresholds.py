# %%
"""
Compare two detection thresholds cell by cell, in the same station x injection layout as
eval_recovery.png.

Run the pipeline once per threshold, keeping the scored picks each time:

    python cut_templates_cc.py --region synthetic --mode TM --nmad 0
    ... run_qtm.py / run_qtm_association.py / evaluate_qtm.py ...
    cp local/synthetic/benchmark/eval_picks.csv local/synthetic/benchmark/eval_picks_abs.csv

    python cut_templates_cc.py --region synthetic --mode TM --nmad 9
    ... rerun ...
    cp local/synthetic/benchmark/eval_picks.csv local/synthetic/benchmark/eval_picks_mad9.csv

    python tests/qtm/compare_thresholds.py --region synthetic --runs abs mad9

Writes {region}/benchmark/figures/eval_threshold_compare.png.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import DEFAULT_ROOT

MISSED, FOUND, WRONG, DEAD = 0, 1, 2, 3
CMAP = matplotlib.colors.ListedColormap(["#f0f0f0", "#1a9850", "#f46d43", "#ffffff"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--region", type=str, default="synthetic")
    parser.add_argument("--runs", type=str, nargs="+", default=["abs", "mad9"],
                        help="suffixes of eval_picks_<run>.csv, in the order to display")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="display names, one per run (default: the suffixes)")
    return parser.parse_args()


def grids(gp, labelled, order, row, col):
    """status per (station, injection) for each phase"""
    tp = labelled[labelled["label"] == "true_positive"]
    wt = labelled[labelled["label"] == "wrong_template"]
    out = {}
    for ph in ["P", "S"]:
        m = np.full((len(row), len(col)), DEAD, dtype=int)
        g = gp[(gp["phase_type"] == ph) & gp["injected"]]
        m[[row[s] for s in g["station_id"]], [col[i] for i in g["inj_id"]]] = MISSED
        w = wt[wt["phase_type"] == ph]
        if len(w):
            m[[row[s] for s in w["station_id"]], [col[i] for i in w["inj_id"]]] = WRONG
        t = tp[tp["phase_type"] == ph]
        m[[row[s] for s in t["station_id"]], [col[i] for i in t["inj_id"]]] = FOUND
        out[ph] = m
    return out


# %%
def main():
    args = parse_args()
    base = f"{args.root_path}/{args.region}"
    figdir = f"{base}/benchmark/figures"
    labels = args.labels or args.runs
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must have one entry per --runs")

    gt = pd.read_csv(f"{base}/benchmark/ground_truth_events.csv")
    gt["time"] = pd.to_datetime(gt["time"], utc=True, format="ISO8601")
    gt = gt.sort_values("time").reset_index(drop=True)
    gp = pd.read_csv(f"{base}/benchmark/ground_truth_picks.csv")

    runs = {}
    for r in args.runs:
        f = f"{base}/benchmark/eval_picks_{r}.csv"
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found -- see the header of this script for how to produce it")
        runs[r] = pd.read_csv(f)

    order = list(gt["inj_id"])
    col = {v: i for i, v in enumerate(order)}
    stations = sorted(gp["station_id"].unique())

    ## one station ordering for every panel, from the first run, so cells line up across
    ## columns and a difference can be read by eye
    ref = runs[args.runs[0]]
    tp0 = ref[ref["label"] == "true_positive"]
    score = tp0.groupby("station_id").size().reindex(stations).fillna(0).values
    stations = [s for _, s in sorted(zip(-score, stations))]
    row = {s: i for i, s in enumerate(stations)}

    mats = {r: grids(gp, runs[r], order, row, col) for r in args.runs}

    n = len(args.runs)
    fig = plt.figure(figsize=((9.5 if n <= 2 else 7.6) * n, 13))
    gs = fig.add_gridspec(3, n, height_ratios=[0.85, 3, 3], hspace=0.14, wspace=0.06)

    for k, r in enumerate(args.runs):
        d = runs[r]
        cnt = d["label"].value_counts()
        tp = int(cnt.get("true_positive", 0))
        fp = int(cnt.get("wrong_template", 0) + cnt.get("wrong_time", 0) + cnt.get("dead_channel", 0))
        inj = int(gp["injected"].sum())

        ## summary bars
        ax = fig.add_subplot(gs[0, k])
        keys = ["true_positive", "wrong_template", "wrong_time"]
        cols = ["#1a9850", "#f46d43", "#d73027"]
        vals = [int(cnt.get(x, 0)) for x in keys]
        b = ax.barh(range(len(keys)), vals, color=cols)
        ax.bar_label(b, fmt="%d", fontsize=9, padding=3)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(["true positive", "wrong template", "wrong time"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(int(runs[x]["label"].value_counts().max()) for x in args.runs) * 1.18)
        ax.set_title(f"{labels[k]}\nprecision {tp/max(tp+fp,1):.1%}   recall {tp}/{inj} = {tp/max(inj,1):.1%}",
                     fontsize=12)
        ax.grid(alpha=0.3, axis="x")

        for j, ph in enumerate(["P", "S"]):
            a = fig.add_subplot(gs[j + 1, k])
            m = mats[r][ph]
            a.imshow(m, aspect="auto", cmap=CMAP, vmin=-0.5, vmax=3.5, interpolation="nearest")
            a.set_xlim(-0.5, len(order) - 0.5)
            if k == 0:
                a.set_yticks(range(len(stations)))
                a.set_yticklabels(stations, fontsize=5.5)
                a.set_ylabel(f"{ph} phase   (station, ordered by picks recovered)", fontsize=9)
            else:
                a.set_yticks([])
            n_f, n_i = int((m == FOUND).sum()), int((m != DEAD).sum())
            n_w = int((m == WRONG).sum())
            a.text(0.005, 1.012, f"{ph}: {n_f}/{n_i} recovered, {n_w} wrong-template cells",
                   transform=a.transAxes, fontsize=9)
            for sid, g in gt.groupby("set_id"):
                a.axvline(col[g["inj_id"].iloc[-1]] + 0.5, color="k", lw=1.4, alpha=0.8)
            if j == 1:
                a.set_xlabel("injection (time order; vertical lines separate SNR sets)", fontsize=9)
                a.set_xticks(range(0, len(order), 5))
                a.tick_params(labelsize=7)
            else:
                a.set_xticks([])
                ## inside the panel, not above it -- above collides with the summary bars
                tr = a.get_xaxis_transform()
                for sid, g in gt.groupby("set_id"):
                    lo, hi = col[g["inj_id"].iloc[0]], col[g["inj_id"].iloc[-1]]
                    a.text((lo + hi) / 2, 0.985, f"SNR {g['amplitude'].iloc[0]:g}", transform=tr,
                           ha="center", va="top", fontsize=8.5,
                           bbox=dict(fc="w", ec="0.6", lw=0.4, alpha=0.85, pad=1.2))

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.5", lw=0.4) for c in
               ["#1a9850", "#f0f0f0", "#f46d43", "#ffffff"]]
    fig.legend(handles, ["pick recovered", "pick missed", "wrong template fired", "not injected"],
               loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.001))
    fig.suptitle("detection threshold: absolute cc vs Shelly-style N x MAD", fontsize=14, y=0.997)
    fig.subplots_adjust(top=0.9, bottom=0.065, left=0.075, right=0.99)
    out = f"{figdir}/eval_threshold_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    ## the numbers behind the picture
    print(f"\n{'run':>10}{'picks':>9}{'TP':>8}{'wrong_tmpl':>12}{'wrong_time':>12}{'precision':>11}{'recall':>9}")
    for r, lab in zip(args.runs, labels):
        c = runs[r]["label"].value_counts()
        tp = int(c.get("true_positive", 0))
        wt, wtime = int(c.get("wrong_template", 0)), int(c.get("wrong_time", 0))
        fp = wt + wtime + int(c.get("dead_channel", 0))
        print(f"{lab:>10}{len(runs[r]):>9}{tp:>8}{wt:>12}{wtime:>12}"
              f"{tp/max(tp+fp,1):>10.1%}{tp/max(int(gp['injected'].sum()),1):>9.1%}")


if __name__ == "__main__":
    main()

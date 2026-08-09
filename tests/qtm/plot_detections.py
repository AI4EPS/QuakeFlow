# %%
"""
Visualise the QTM detection results against the benchmark ground truth.

Two figures, each answering one question:

  detect_01_timeline    What did it find, and what is every candidate made of?
                        Ground truth over time, the station count behind each
                        candidate detection, and a raster of every correlation pick.
                        A true detection is a full column of stations at cc ~ 1; a
                        false one is a handful of stations. The picture shows what
                        separates them.

  detect_02_threshold   Where should the threshold go?
                        Distributions of station count and cc for real detections
                        against false alarms, and recall/precision as the station
                        threshold moves.

    python tests/qtm/plot_detections.py --region synthetic

Figures go to {region}/benchmark/figures/.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import DEFAULT_ROOT

## categorical slots 1-3 of the reference palette (validated for all-pairs use);
## identity is never carried by colour alone here -- markers and labels differ too
C_TRUE = "#2a78d6"
C_FALSE = "#eb6834"
C_MISS = "#1baf7a"
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--region", type=str, default="synthetic")
    parser.add_argument("--tolerance", type=float, default=2.0, help="origin-time tolerance for a match (s)")
    return parser.parse_args()


# %%
def load(args):
    base = f"{args.root_path}/{args.region}"
    gt = pd.read_csv(f"{base}/benchmark/ground_truth_events.csv")
    gt["time"] = pd.to_datetime(gt["time"], utc=True, format="ISO8601")
    gp = pd.read_csv(f"{base}/benchmark/ground_truth_picks.csv")
    ev = pd.read_csv(f"{base}/qtm/qtm_events.csv", parse_dates=["time"])
    ev["time"] = pd.to_datetime(ev["time"], utc=True)
    pk = pd.read_csv(f"{base}/qtm/qtm_picks.csv", parse_dates=["origin_time"])
    pk["origin_time"] = pd.to_datetime(pk["origin_time"], utc=True)
    stations = pd.read_csv(f"{base}/qtm/template_stations.csv")

    ## what each candidate is made of
    agg = pk[pk["event_index"] != -1].groupby("event_index").agg(
        n_station=("idx_sta", "nunique"), n_pick=("idx_sta", "size"), cc_median=("phase_score", "median")
    )
    ev = ev.join(agg, on="event_index")

    ## greedy nearest-in-time match, each candidate used once
    gt = gt.sort_values("time").reset_index(drop=True)
    ev = ev.sort_values("time").reset_index(drop=True)
    used, det = set(), []
    for _, g in gt.iterrows():
        d = (ev["time"] - g["time"]).dt.total_seconds().abs()
        d[list(used)] = np.inf
        k = int(d.idxmin())
        if d[k] <= args.tolerance:
            used.add(k)
            det.append(True)
        else:
            det.append(False)
    gt["detected"] = det
    ev["matched"] = [i in used for i in range(len(ev))]

    ## stations ordered by how far they sit from the sources, so moveout reads top to bottom
    order = gp.groupby("station_id")["dist_km"].median().sort_values()
    order = order[order.index.isin(stations["station_id"])]
    yorder = {sid: i for i, sid in enumerate(order.index)}
    stations["y"] = stations["station_id"].map(yorder)
    pk = pk.merge(stations[["idx_sta", "station_id", "y"]], on="idx_sta", how="left")
    return gt, ev, pk, stations, order


# %%
def plot_timeline(args, gt, ev, pk, order):
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    os.makedirs(figdir, exist_ok=True)
    t0 = min(gt["time"].min(), pk["origin_time"].min())
    mins = lambda t: (t - t0).dt.total_seconds() / 60

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, height_ratios=[1, 1.1, 2.2])

    ## ---- what was injected, and did we get it
    ax = axes[0]
    for flag, color, marker, label in [(True, C_TRUE, "o", "recovered"), (False, C_MISS, "X", "missed")]:
        m = gt["detected"] == flag
        if m.any():
            ax.scatter(mins(gt.loc[m, "time"]), gt.loc[m, "idx_event"], s=46, marker=marker,
                       facecolor=color, edgecolor="white", linewidth=0.8, label=f"{label} ({m.sum()})", zorder=3)
    ax.set_ylabel("source event", color=INK_MUTED, fontsize=9)
    ax.set_title(f"ground truth: {len(gt)} injections, {int(gt['detected'].sum())} recovered",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)

    ## ---- how much of the network is behind each candidate
    ax = axes[1]
    for flag, color, marker, label in [(True, C_TRUE, "o", "real detection"), (False, C_FALSE, "^", "false alarm")]:
        m = ev["matched"] == flag
        if m.any():
            ax.scatter(mins(ev.loc[m, "time"]), ev.loc[m, "n_station"], s=40, marker=marker,
                       facecolor=color, edgecolor="white", linewidth=0.8,
                       label=f"{label} ({m.sum()})", zorder=3)
    if ev["matched"].any() and (~ev["matched"]).any():
        lo = int(ev.loc[~ev["matched"], "n_station"].max())
        hi = int(ev.loc[ev["matched"], "n_station"].min())
        ax.axhspan(lo, hi, color=INK_MUTED, alpha=0.10, zorder=0)
        ax.annotate(f"nothing lands between {lo} and {hi} stations", (0.005, (lo + hi) / 2),
                    xycoords=("axes fraction", "data"), fontsize=8, color=INK_MUTED, va="center")
    ax.set_ylabel("stations in the\ncandidate", color=INK_MUTED, fontsize=9)
    ax.set_title(f"candidate detections: {len(ev)} entries in qtm_events.csv", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)

    ## ---- every correlation pick, so the shape of a true detection is visible
    ax = axes[2]
    for t in gt["time"]:
        ax.axvline((t - t0).total_seconds() / 60, color=C_TRUE, linewidth=0.7, alpha=0.28, zorder=1)
    sc = ax.scatter(mins(pk["origin_time"]), pk["y"], c=pk["phase_score"], cmap="Blues",
                    vmin=0.5, vmax=1.0, s=5, linewidth=0, zorder=2)
    cax = ax.inset_axes([1.006, 0.0, 0.010, 1.0])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("cc", fontsize=9, color=INK_MUTED)
    cb.outline.set_visible(False)
    ax.set_yticks(range(0, len(order), 3))
    ax.set_yticklabels([f"{s}  {d:.0f} km" for s, d in list(order.items())[::3]], fontsize=6, color=INK_MUTED)
    ax.set_ylabel("station, nearest at the bottom", color=INK_MUTED, fontsize=9)
    ax.set_xlabel(f"minutes into the record (starts {t0.strftime('%Y-%m-%d %H:%M:%S')})",
                  color=INK_MUTED, fontsize=9)
    ax.set_title("every correlation pick: a real detection is a full column, a false alarm is a few stations",
                 fontsize=10, color=INK, loc="left")
    ax.grid(alpha=0.12, linewidth=0.5)
    ax.set_axisbelow(True)

    for a in axes:
        a.tick_params(labelsize=8, colors=INK_MUTED)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{figdir}/detect_01_timeline.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/detect_01_timeline.png")


# %%
def plot_threshold(args, gt, ev):
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    real, false = ev[ev["matched"]], ev[~ev["matched"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    bins = np.arange(0, ev["n_station"].max() + 3) - 0.5
    ax.hist([false["n_station"], real["n_station"]], bins=bins, stacked=False,
            color=[C_FALSE, C_TRUE], label=[f"false alarm ({len(false)})", f"real ({len(real)})"],
            edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("distinct stations in the candidate", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("candidates", color=INK_MUTED, fontsize=9)
    ax.set_title(f"station count: {false['n_station'].min():.0f}-{false['n_station'].max():.0f} against "
                 f"{real['n_station'].min():.0f}-{real['n_station'].max():.0f}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    bins = np.linspace(0.5, 1.0, 26)
    ax.hist([false["cc_median"], real["cc_median"]], bins=bins,
            color=[C_FALSE, C_TRUE], label=["false alarm", "real"], edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("median cc of the candidate's picks", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("candidates", color=INK_MUTED, fontsize=9)
    ax.set_title(f"median cc: {false['cc_median'].median():.2f} against {real['cc_median'].median():.2f}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)

    ## one axis: recall and precision are both fractions
    ax = axes[2]
    thr = np.arange(1, int(ev["n_station"].max()) + 1)
    keep = [ev[ev["n_station"] >= t] for t in thr]
    recall = [k["matched"].sum() / len(gt) for k in keep]
    precision = [(k["matched"].sum() / len(k)) if len(k) else np.nan for k in keep]
    ax.plot(thr, recall, color=C_TRUE, linewidth=2, label="recall")
    ax.plot(thr, precision, color=C_FALSE, linewidth=2, linestyle="--", label="precision")
    best = int(thr[np.nanargmax(np.array(recall) + np.array(precision))])
    ax.axvline(best, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.annotate(f"{best} stations", (best, 0.03), fontsize=8, color=INK_MUTED, rotation=90,
                ha="right", va="bottom")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("minimum distinct stations required", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("fraction", color=INK_MUTED, fontsize=9)
    ax.set_title("either measure would do -- the association uses neither",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    for a in axes:
        a.tick_params(labelsize=8, colors=INK_MUTED)
        a.grid(alpha=0.18, linewidth=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{figdir}/detect_02_threshold.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/detect_02_threshold.png")

    ## the table view, so nothing depends on reading colour off the figure
    out = f"{args.root_path}/{args.region}/benchmark/detection_threshold.csv"
    pd.DataFrame({"min_stations": thr, "recall": recall, "precision": precision,
                  "n_candidates": [len(k) for k in keep]}).to_csv(out, index=False)
    print(f"  {out}")
    return best


# %%
if __name__ == "__main__":
    args = parse_args()
    gt, ev, pk, stations, order = load(args)
    print(f"{len(gt)} injections, {len(ev)} candidates, {len(pk)} correlation picks")
    plot_timeline(args, gt, ev, pk, order)
    best = plot_threshold(args, gt, ev)
    real, false = ev[ev["matched"]], ev[~ev["matched"]]
    print(f"\nreal detections   : {len(real)}, stations {real['n_station'].min()}-{real['n_station'].max()}")
    print(f"false alarms      : {len(false)}, stations {false['n_station'].min()}-{false['n_station'].max()}")
    print(f"requiring >= {best} distinct stations: "
          f"recall {ev[(ev['n_station'] >= best) & ev['matched']].shape[0]}/{len(gt)}, "
          f"false alarms {ev[(ev['n_station'] >= best) & ~ev['matched']].shape[0]}")

# %%
"""
Check the benchmark by reading it back the way the pipeline does.

Cuts each source event out of the **continuous synthetic record** (`waveforms/`, the same
files cut_templates_cc.py and run_qtm.py read) and draws one record section per event:
vertical component, sorted by ground-truth P travel time, with the ground-truth P and S
marked.  Only the window that was actually spliced into the record is drawn.

    python tests/qtm/plot_events.py --region synthetic
    python tests/qtm/plot_events.py --region synthetic --clean   # injected signal only

Figures go to {region}/benchmark/figures/.
"""

import argparse
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from paths import DEFAULT_ROOT
from mseed import read_trace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--region", type=str, default="synthetic")
    parser.add_argument("--component", type=str, default="Z")
    parser.add_argument("--block", type=str, default="all", choices=["all", "template", "test"],
                        help="which injections to draw a record section for")
    parser.add_argument("--gallery_station", type=str, default=None,
                        help="station for the all-injections gallery (default: the closest one)")
    parser.add_argument("--instrument", type=str, default=None, help="restrict to one of BH/HH/HN/EH")
    parser.add_argument("--max_sections", type=int, default=1,
                        help="per-injection record sections to write, per block (-1 for all). "
                             "record_gallery.png already shows every injection.")
    parser.add_argument("--freqmin", type=float, default=2.0)
    parser.add_argument("--freqmax", type=float, default=12.0)
    parser.add_argument("--no_filter", action="store_true", help="plot raw broadband instead of the 2-12 Hz band")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="plot waveforms_clean/ (injected segments only) instead of the continuous waveforms/",
    )
    return parser.parse_args()


def bandpass(x, sr, fmin, fmax):
    sos = butter(4, [fmin, min(fmax, 0.45 * sr)], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


# %%
def main():
    args = parse_args()
    base = f"{args.root_path}/{args.region}"
    wavdir = f"{base}/waveforms_clean" if args.clean else f"{base}/waveforms"
    figdir = f"{base}/benchmark/figures"
    os.makedirs(figdir, exist_ok=True)
    if not os.path.exists(wavdir):
        raise SystemExit(f"{wavdir} not found -- rerun make_benchmark.py with --noise_free")
    print(f"Reading {wavdir}")

    gt_ev = pd.read_csv(f"{base}/benchmark/ground_truth_events.csv")
    gt_pk = pd.read_csv(f"{base}/benchmark/ground_truth_picks.csv")
    gt_ev["time"] = pd.to_datetime(gt_ev["time"], utc=True, format="ISO8601")
    gt_pk["phase_time"] = pd.to_datetime(gt_pk["phase_time"], utc=True, format="ISO8601")
    stations = pd.read_csv(f"{base}/adloc/adloc_stations.csv")

    files = defaultdict(dict)  # station_id -> component -> path
    for y in sorted(os.listdir(wavdir)):
        for j in sorted(os.listdir(f"{wavdir}/{y}")):
            for f in sorted(os.listdir(f"{wavdir}/{y}/{j}")):
                if not f.endswith(".mseed"):
                    continue
                net, sta, loc, cha = f.replace(".mseed", "").split(".")
                files[f"{net}.{sta}.{loc}.{cha[:2]}"][cha[-1]] = f"{wavdir}/{y}/{j}/{f}"

    sel = gt_ev if args.block == "all" else gt_ev[gt_ev["block"] == args.block]
    sel = sel.sort_values("inj_id")
    tmpl = gt_ev[gt_ev["block"] == "template"].sort_values("idx_event")

    ## One 45-station section per injection is ~1 MB and they all look alike; record_gallery
    ## already shows every injection side by side. Keep a couple as spot checks by default.
    if args.max_sections >= 0:
        keep = pd.concat([g.head(args.max_sections) for _, g in sel.groupby("block", sort=False)])
        if len(keep) < len(sel):
            print(f"Plotting {len(keep)} of {len(sel)} record sections "
                  f"(--max_sections {args.max_sections}; use -1 for all)")
        sel = keep.sort_values("inj_id")
    print(f"Plotting {len(sel)} injections ({args.block})")

    # ------------------------------------------------------------------ record sections
    for _, ev in sel.iterrows():
        pk = gt_pk[(gt_pk["inj_id"] == ev["inj_id"]) & (gt_pk["injected"])]
        p = pk[pk["phase_type"] == "P"].set_index("station_id")
        s = pk[pk["phase_type"] == "S"].set_index("station_id")
        sids = sorted(set(p.index) & set(s.index) & set(files), key=lambda x: p.loc[x, "tp_tt"])
        if args.instrument:
            sids = [x for x in sids if x.split(".")[3] == args.instrument]
        if not sids:
            continue

        t0 = -float(p.loc[sids, "win_pre"].max())
        t1 = float(p.loc[sids, "win_post"].max())

        fig, ax = plt.subplots(figsize=(11, max(6, 0.3 * len(sids))))
        for k, sid in enumerate(sids):
            comp = args.component if args.component in files[sid] else sorted(files[sid])[-1]
            tr = read_trace(files[sid][comp])
            sr = tr.stats.sampling_rate
            tp = obspy.UTCDateTime(p.loc[sid, "phase_time"].to_pydatetime())
            pre, post = float(p.loc[sid, "win_pre"]), float(p.loc[sid, "win_post"])
            # exactly the window that was spliced into the record, nothing else
            i0 = max(int(round((tp - pre - tr.stats.starttime) * sr)), 0)
            i1 = min(int(round((tp + post - tr.stats.starttime) * sr)), tr.stats.npts)
            d = tr.data[i0:i1].astype(np.float64)
            if len(d) < 10:
                continue
            if not args.no_filter:
                d = bandpass(d, sr, args.freqmin, args.freqmax)
            amp = np.max(np.abs(d))
            if amp == 0:
                continue
            ax.plot(-pre + np.arange(len(d)) / sr, 0.45 * d / amp + k, lw=0.4, color="k")

            ts_rel = float((s.loc[sid, "phase_time"] - p.loc[sid, "phase_time"]).total_seconds())
            ax.plot([0, 0], [k - 0.45, k + 0.45], color="tab:red", lw=1.2)
            ax.plot([ts_rel, ts_rel], [k - 0.45, k + 0.45], color="tab:blue", lw=1.2)
            ax.plot([-pre, post], [k, k], color="0.7", lw=0.3, zorder=0)
            for edge in (-pre, post):
                ax.plot([edge, edge], [k - 0.3, k + 0.3], color="0.6", lw=0.8)
            snr = p.loc[sid, "snr"] if "snr" in p.columns else np.nan
            ax.text(
                t0 + 0.05,
                k + 0.12,
                f"{sid}  tP={p.loc[sid,'tp_tt']:.1f}s  S-P={ts_rel:.1f}s"
                + (f"  SNR={snr:.1f}" if np.isfinite(snr) else ""),
                fontsize=5.5,
                va="bottom",
            )

        ax.axvline(0, color="tab:red", lw=0.6, alpha=0.4)
        ax.set_xlim(t0, t1)
        ax.set_ylim(-1, len(sids))
        ax.set_yticks([])
        ax.set_xlabel("time relative to P (s)")
        band = "raw" if args.no_filter else f"{args.freqmin:g}-{args.freqmax:g} Hz"
        ax.set_title(
            f"injection {int(ev['inj_id'])} [{ev['block']}]  source {int(ev['idx_event'])} "
            f"(adloc #{int(ev['src_event_index'])})  M{ev['magnitude']:.2f}  SNR {ev['snr_z_median']:.3g}  |  "
            f"{band}, {args.component} comp, {'injected signal only' if args.clean else 'continuous record'}  |  "
            f"trace-normalised; red=P blue=S, grey ticks = window ends",
            fontsize=9,
        )
        fig.tight_layout()
        out = f"{figdir}/record_inj_{int(ev['inj_id']):02d}_{ev['block']}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"  {out}  ({len(sids)} stations)")

    # ------------------------------------------------------------------ all injections, one station
    ref = args.gallery_station or gt_pk[gt_pk["injected"]].groupby("station_id")["dist_km"].median().idxmin()
    comp = args.component if args.component in files.get(ref, {}) else sorted(files.get(ref, {"Z": None}))[-1]
    fig, axes = plt.subplots(len(gt_ev), 2, figsize=(13, 0.62 * len(gt_ev)), sharex=True)
    fig.subplots_adjust(left=0.09)
    traces, labels = [], []
    for _, ev in gt_ev.sort_values("inj_id").iterrows():
        pk = gt_pk[(gt_pk["inj_id"] == ev["inj_id"]) & (gt_pk["station_id"] == ref)
                   & (gt_pk["phase_type"] == "P") & gt_pk["injected"]]
        if not len(pk) or ref not in files:
            traces.append(None)
            labels.append(ev)
            continue
        row = pk.iloc[0]
        tr = read_trace(files[ref][comp])
        sr = tr.stats.sampling_rate
        tp = obspy.UTCDateTime(row["phase_time"].to_pydatetime())
        pre, post = float(row["win_pre"]), float(row["win_post"])
        i0 = max(int(round((tp - pre - tr.stats.starttime) * sr)), 0)
        i1 = min(int(round((tp + post - tr.stats.starttime) * sr)), tr.stats.npts)
        d = np.asarray(tr.data[i0:i1], dtype=np.float64)
        if not args.no_filter:
            d = bandpass(d, sr, args.freqmin, args.freqmax)
        traces.append((-pre + np.arange(len(d)) / sr, d, row))
        labels.append(ev)

    lim = max((np.max(np.abs(t[1])) for t in traces if t is not None), default=1.0)
    for k, (t, ev) in enumerate(zip(traces, labels)):
        color = "tab:green" if ev["block"] == "template" else "tab:red"
        for col in (0, 1):
            ax = axes[k, col]
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            if t is None:
                ax.text(0.5, 0.5, "not injected (dead channel)", transform=ax.transAxes,
                        fontsize=6, ha="center", va="center", color="0.5")
                continue
            x, d, row = t
            amp = lim if col == 0 else max(np.max(np.abs(d)), 1e-30)
            ax.plot(x, d / amp, lw=0.4, color="k")
            ax.axvline(0, color="tab:red", lw=0.8)
            ax.axvline(row["ts_tt"] - row["tp_tt"], color="tab:blue", lw=0.8)
            ax.set_ylim(-1.1, 1.1)
        axes[k, 0].text(-0.012, 0.5, f"{int(ev['inj_id']):02d} {ev['block'][:4]} src{int(ev['idx_event'])}\n"
                                     f"SNR {ev['snr_z_median']:.3g}",
                        transform=axes[k, 0].transAxes, fontsize=5.5, va="center", ha="right",
                        color=color, clip_on=False)
    axes[-1, 0].set_xlabel("time relative to P (s)")
    axes[-1, 1].set_xlabel("time relative to P (s)")
    axes[0, 0].set_title(f"all {len(gt_ev)} injections at {ref}{comp} -- common amplitude scale", fontsize=9)
    axes[0, 1].set_title("same traces, each normalised to itself", fontsize=9)
    fig.tight_layout(rect=(0.07, 0, 1, 1))
    fig.savefig(f"{figdir}/record_gallery.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/record_gallery.png")

    # ------------------------------------------------------------------ overview
    fig = plt.figure(figsize=(13, 5))

    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(stations["longitude"], stations["latitude"], marker="v", s=28, c="tab:blue", label="station")
    ax.scatter(tmpl["longitude"], tmpl["latitude"], s=18 * tmpl["magnitude"] ** 2, c="tab:red", alpha=0.7,
               edgecolor="k", lw=0.4, label="benchmark event")
    for _, e in tmpl.iterrows():
        ax.annotate(int(e["idx_event"]), (e["longitude"], e["latitude"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("geometry (ground-truth locations)")
    ax.legend(fontsize=7)
    ax.set_aspect(1 / np.cos(np.deg2rad(stations["latitude"].mean())))

    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(tmpl["idx_event"], tmpl["magnitude"], c="tab:red")
    for _, e in tmpl.iterrows():
        ax.annotate(f"#{int(e['src_event_index'])}", (e["idx_event"], e["magnitude"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("benchmark event index")
    ax.set_ylabel("magnitude")
    ax.set_title("magnitudes")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 3, 3)
    t0 = gt_ev["time"].min()
    sc = ax.scatter((gt_ev["time"] - t0).dt.total_seconds() / 60, gt_ev["idx_event"],
                    c=gt_ev["snr_z_median"] if "snr_z_median" in gt_ev else "tab:orange",
                    norm=matplotlib.colors.LogNorm() if "snr_z_median" in gt_ev else None,
                    cmap="viridis", s=22, edgecolor="k", lw=0.3)
    if "snr_z_median" in gt_ev:
        plt.colorbar(sc, ax=ax, label="median-Z SNR")
    ax.set_xlabel("minutes into the record")
    ax.set_ylabel("source event index")
    ax.set_title("injection schedule")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{figdir}/record_overview.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/record_overview.png")

    # ------------------------------------------------------------------ 3-component check
    ev = tmpl.iloc[0]
    pk = gt_pk[(gt_pk["inj_id"] == ev["inj_id"]) & (gt_pk["injected"])]
    p = pk[pk["phase_type"] == "P"].set_index("station_id")
    s = pk[pk["phase_type"] == "S"].set_index("station_id")
    sids = sorted(set(p.index) & set(s.index) & set(files), key=lambda x: p.loc[x, "tp_tt"])[:4]
    fig, axes = plt.subplots(len(sids), 3, figsize=(12, 2.1 * len(sids)), sharex=True)
    axes = np.atleast_2d(axes)
    for r, sid in enumerate(sids):
        ts_rel = float((s.loc[sid, "phase_time"] - p.loc[sid, "phase_time"]).total_seconds())
        pre, post = float(p.loc[sid, "win_pre"]), float(p.loc[sid, "win_post"])
        for c, comp in enumerate(["E", "N", "Z"]):
            ax = axes[r, c]
            if comp not in files[sid]:
                ax.set_axis_off()
                continue
            tr = read_trace(files[sid][comp])
            sr = tr.stats.sampling_rate
            tp = obspy.UTCDateTime(p.loc[sid, "phase_time"].to_pydatetime())
            i0 = max(int(round((tp - pre - tr.stats.starttime) * sr)), 0)
            i1 = min(int(round((tp + post - tr.stats.starttime) * sr)), tr.stats.npts)
            d = tr.data[i0:i1].astype(np.float64)
            if not args.no_filter:
                d = bandpass(d, sr, args.freqmin, args.freqmax)
            ax.plot(-pre + np.arange(len(d)) / sr, d, lw=0.4, color="k")
            ax.axvline(0, color="tab:red", lw=1)
            ax.axvline(ts_rel, color="tab:blue", lw=1)
            ax.set_title(f"{sid}{comp}  tP={p.loc[sid,'tp_tt']:.1f}s  {sr:.0f} Hz", fontsize=7)
            ax.tick_params(labelsize=6)
    axes[-1, 1].set_xlabel("time relative to P (s)")
    fig.suptitle(f"event {int(ev['idx_event'])}: 3-component, closest stations", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{figdir}/record_event_{int(ev['idx_event']):02d}_3c.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/record_event_{int(ev['idx_event']):02d}_3c.png")


if __name__ == "__main__":
    main()

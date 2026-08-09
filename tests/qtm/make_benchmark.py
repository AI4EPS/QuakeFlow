# %%
"""
Step 3 of the QTM benchmark: inject the cut events into the noise and write a region the
existing pipeline can be pointed at without modification.

Three knobs, and nothing else:

    --template_ratio    SNR of the template set  -> how clean the templates are
    --test_ratio        SNR of each test set, one value per set -> how hard the detections are
    --spacing_sec       seconds between injections -> inter-event spacing

The record is normalised (--normalize station) so these ratios ARE the SNR rather than a
multiplier on instrument counts.  Passing several --test_ratio values injects several test
sets in one record, which is how the detection-vs-SNR curve is measured without re-running
the pipeline per point.

The record is laid out as one set after another, every set a full pass over every source:

    template set     set 0, one injection per source, at --template_ratio.
                     These are the events written to adloc_events.csv / adloc_picks.csv,
                     so cut_templates_cc.py builds its templates out of them.
    test sets        one per --test_ratio, each a full pass over every source at that SNR,
                     laid out after the template set.  These are what QTM has to find.

Because every injection is a copy of the same waveform, a correct pipeline must recover
cc = 1 at the ground-truth origin time for the matching template.

    python tests/qtm/make_benchmark.py --region synthetic --test_ratio 3 1 0.3 0.1

Writes {region}/waveforms/, {region}/adloc/, {region}/config.json, the ground-truth tables
in {region}/benchmark/, and figures to {region}/benchmark/figures/.
"""

import argparse
import json
import os
import shutil
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

FREQMIN, FREQMAX = 2.0, 12.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--src_region", type=str, default="demo", help="region the config and stations come from")
    parser.add_argument("--region", type=str, default="synthetic")
    parser.add_argument("--template_ratio", type=float, default=10.0,
                        help="SNR of the one template set. With --normalize station this IS the SNR")
    parser.add_argument("--test_ratio", type=float, nargs="+", default=[3.0],
                        help="SNR of each test set, one value per set: '--test_ratio 3 1 0.3 0.1' injects "
                             "four test sets, each a full pass over every source at that SNR, so one run "
                             "gives the whole detection-vs-SNR curve")
    parser.add_argument("--spacing_sec", type=float, default=None,
                        help="seconds between injections (default: spread them evenly over the record)")
    parser.add_argument("--normalize", type=str, default="station", choices=["station", "network", "none"],
                        help="scale noise and events to unit rms before injecting, so the ratios above ARE the "
                             "SNR. 'station' equalises every station (all its components together); "
                             "'network' keeps the amplitude falloff across the network; 'none' injects raw counts")
    parser.add_argument("--keep_dead_stations", action="store_true",
                        help="keep stations whose cut is flat in some source event (marked injected=False) "
                             "instead of dropping them from the benchmark entirely")
    parser.add_argument("--guard_sec", type=float, default=30.0, help="quiet margin around the blocks")
    parser.add_argument("--noise_free", action="store_true", help="also write waveforms_clean/ (injections only)")
    return parser.parse_args()


def bandpass(x, sr, fmin=FREQMIN, fmax=FREQMAX):
    sos = butter(4, [fmin, min(fmax, 0.45 * sr)], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


# %%
def schedule(args, events, win, noise):
    """One template set plus one test set per --test_ratio, uniformly spaced.

    Every set is a full pass over all sources, so source counts are identical across sets
    and recall can be compared set to set without reweighting.  Every injection, including
    the template ones, is a detection the pipeline must make.
    """
    t0 = noise["starttime"].min()
    duration = (noise["npts"] / noise["sampling_rate"]).max()
    t1 = t0 + pd.Timedelta(seconds=duration)
    n_src = len(events)
    n_set = 1 + len(args.test_ratio)  # one template set, then one per --test_ratio
    n_inj = n_src * n_set

    ## an injection occupies [origin + min(tp-pre), origin + max(tp+post)] across all stations
    win = win.assign(span_start=win["tp_tt"] - win["win_pre"], span_end=win["tp_tt"] + win["win_post"])
    lead = float(win["span_start"].min())
    span = float(win["span_end"].max()) - lead

    ## keep the first and last injection fully inside the record, then spread the rest evenly
    first = t0 + pd.Timedelta(seconds=args.guard_sec - min(0.0, lead))
    last = t1 - pd.Timedelta(seconds=args.guard_sec + max(0.0, float(win["span_end"].max())))
    room = (last - first).total_seconds()
    if room <= 0:
        raise SystemExit(f"noise record ({duration:.0f} s) is shorter than one event ({span:.0f} s); "
                         f"re-run extract_noise.py with a longer --duration")
    spacing = args.spacing_sec if args.spacing_sec else room / max(n_inj - 1, 1)
    if spacing * (n_inj - 1) > room:
        raise SystemExit(
            f"{n_inj} injections ({n_set} sets x {n_src} sources) at {spacing:.0f} s need "
            f"{spacing * (n_inj - 1):.0f} s but the record only has {room:.0f} s; "
            f"drop a --test_ratio, lower --spacing_sec, or lengthen the noise window")
    times = [first + pd.Timedelta(seconds=i * spacing) for i in range(n_inj)]

    ## Set 0 is the template set -- one injection per source, and the templates the pipeline
    ## gets are cut from these.  Sets 1..N are test sets, each another full pass over every
    ## source at its own --test_ratio.  Sources cycle within a set, so set k occupies the
    ## k-th block of the record and "template" also means "early"; that confound was
    ## measured (2026-08-06) at 0.22% in network-median noise between record halves, so it
    ## is left alone.  Single stations do drift (CI.WMF +51%/hr, CI.SLA -55%/hr), so
    ## per-station test statistics carry it -- see README.
    ratios = [args.template_ratio] + list(args.test_ratio)
    rows = []
    for i, t in enumerate(times):
        i_set, i_src = divmod(i, n_src)
        rows.append({
            "inj_id": i,
            "block": "template" if i_set == 0 else "test",
            "set_id": i_set,
            "idx_event": int(events.iloc[i_src]["idx_event"]),
            "time": t,
            "amplitude": ratios[i_set],
        })
    inj = pd.DataFrame(rows)

    print(f"record          {t0} -> {t1}  ({duration:.0f} s)")
    print(f"injections      {n_inj} = {n_set} sets x {n_src} sources, {spacing:.1f} s apart, "
          f"{times[0].strftime('%H:%M:%S')} -> {times[-1].strftime('%H:%M:%S')}")
    for i_set, r in enumerate(ratios):
        g = inj[inj["set_id"] == i_set]
        lo = (g["time"].min() - t0).total_seconds() / 60
        hi = (g["time"].max() - t0).total_seconds() / 60
        kind = "template (the pipeline gets these)" if i_set == 0 else "test"
        print(f"  set {i_set}  SNR {r:<8g} {len(g)} injections, {lo:5.1f}-{hi:5.1f} min   {kind}")
    print(f"event duration  up to {span:.0f} s across the network"
          + (f"  -- WARNING: longer than the {spacing:.0f} s spacing, injections overlap" if span > spacing else ""))
    return inj, spacing, t0, t1


# %%
def normalization(args, meta, noise):
    """Scale factors that put the noise and the events on a common footing.

    Instrument gains across this network span three orders of magnitude, so a single
    multiplier on raw counts means a different SNR at every station. Dividing the noise by
    its own rms and each event by its own signal rms makes the ratio the SNR itself.

    Both are computed over all of a station's components together, never per component, so
    the relative amplitudes between E, N and Z -- the polarisation the correlation keys on
    -- are untouched.
    """
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))

    if args.normalize == "none":
        return defaultdict(lambda: 1.0), defaultdict(lambda: 1.0)

    if args.normalize == "station":
        noise_scale = defaultdict(lambda: 1.0, noise.groupby("station_id")["rms_band"].apply(rms).to_dict())
        signal_scale = defaultdict(
            lambda: 1.0,
            meta.groupby(["idx_event", "station_id"])["signal_rms"].apply(rms).to_dict(),
        )
    else:  ## network: one scale for the noise, one per event, so the falloff with distance stays
        one = rms(noise["rms_band"])
        noise_scale = defaultdict(lambda: one)
        per_event = meta.groupby("idx_event")["signal_rms"].apply(rms).to_dict()
        signal_scale = defaultdict(lambda: 1.0,
                                   {(e, s): per_event[e] for e, s in meta.groupby(["idx_event", "station_id"]).groups})

    bad = [k for k, v in signal_scale.items() if v <= 0]
    print(f"normalize={args.normalize}: noise scaled by station rms, events by their own signal rms"
          + (f" ({len(bad)} dead event-stations left as is)" if bad else ""))
    return noise_scale, signal_scale


# %%
def inject(args, events, win, noise, inj, noise_scale, signal_scale):
    """One pass over the noise channels, adding every injection that touches each channel."""
    base = f"{args.root_path}/{args.region}"
    year_jday = obspy.UTCDateTime(noise["starttime"].min().to_pydatetime())
    wavdir = f"{base}/waveforms/{year_jday.year}/{year_jday.julday:03d}"
    ## rewrite the directory from scratch: a rebuild with fewer stations must not leave the
    ## previous build's channels behind, or the pipeline reads mseed the ground truth has
    ## no rows for
    for d in ([wavdir] + ([f"{base}/waveforms_clean/{year_jday.year}/{year_jday.julday:03d}"] if args.noise_free else [])):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    cuts = {int(i): np.load(f"{base}/benchmark/events/event_{int(i):02d}.npz") for i in events["idx_event"]}
    winx = win.set_index(["idx_event", "station_id"])

    ## sample grid per station (all components of a station share it)
    grid = noise.groupby("station_id").agg({"starttime": "first", "sampling_rate": "first"})

    ## sample-exact placement, decided once per (injection, station)
    place = {}
    gt_picks = []
    for _, r in inj.iterrows():
        for station_id, g in grid.iterrows():
            if (r["idx_event"], station_id) not in winx.index:
                continue
            w = winx.loc[(r["idx_event"], station_id)]
            sr = g["sampling_rate"]
            i0 = int(round((r["time"] + pd.Timedelta(seconds=w["tp_tt"] - w["win_pre"]) - g["starttime"])
                           .total_seconds() * sr))
            place[(r["inj_id"], station_id)] = i0
            tp = g["starttime"] + pd.Timedelta(seconds=(i0 + round(w["win_pre"] * sr)) / sr)
            for phase, tt in [("P", w["tp_tt"]), ("S", w["ts_tt"])]:
                gt_picks.append(
                    {
                        "inj_id": r["inj_id"],
                        "idx_event": r["idx_event"],
                        "block": r["block"],
                        "station_id": station_id,
                        "phase_type": phase,
                        "phase_time": tp + pd.Timedelta(seconds=tt - w["tp_tt"]),
                        "dist_km": w["dist_km"],
                        "tp_tt": w["tp_tt"],
                        "ts_tt": w["ts_tt"],
                        "win_pre": w["win_pre"],
                        "win_post": w["win_post"],
                        "sampling_rate": sr,
                    }
                )
    gt_picks = pd.DataFrame(gt_picks)
    gt_picks["injected"] = False

    ok = defaultdict(bool)
    amp_used = {}
    for _, row in noise.sort_values("file_name").iterrows():
        name = f"{row['station_id']}{row['component']}"
        trace = read_trace(row["file_name"])
        sr = trace.stats.sampling_rate
        data = np.asarray(trace.data, dtype=np.float64) / max(noise_scale[row["station_id"]], 1e-30)
        clean = np.zeros_like(data)

        for _, r in inj.iterrows():
            if name not in cuts[r["idx_event"]]:
                continue
            i0 = place.get((r["inj_id"], row["station_id"]))
            seg = np.asarray(cuts[r["idx_event"]][name], dtype=np.float64)
            if i0 is None or i0 < 0 or i0 + len(seg) > len(data):
                continue
            scale = signal_scale[(r["idx_event"], row["station_id"])]
            clean[i0 : i0 + len(seg)] += r["amplitude"] * seg / max(scale, 1e-30)
            ## a cut taken from a dead or flat-lined stretch of the source record carries no
            ## signal; marking it injected would make the ground truth claim a detection
            ## that nothing was ever put there to produce
            if np.any(seg):
                ok[(r["inj_id"], row["station_id"])] = True

        trace.data = (data + clean).astype(np.float32)
        trace.write(f"{wavdir}/{name}.mseed", format="MSEED")
        if args.noise_free:
            trace.data = clean.astype(np.float32)
            trace.write(f"{base}/waveforms_clean/{year_jday.year}/{year_jday.julday:03d}/{name}.mseed",
                        format="MSEED")
        amp_used[name] = float(np.max(np.abs(clean)))
        print(f"  {name:20s} {sr:5.0f} Hz  injected {sum(1 for k in ok if k[1] == row['station_id'])} events")

    gt_picks["injected"] = [ok[(a, b)] for a, b in zip(gt_picks["inj_id"], gt_picks["station_id"])]
    print(f"\nWrote {len(noise)} channels -> {wavdir}")
    return gt_picks


# %%
def measure_snr(args, gt_picks, meta, noise, inj, noise_scale, signal_scale):
    """SNR each injection ends up with, measured from what was actually written.

    With --normalize station this comes out equal to the injection's amplitude at every
    station, which is the point of normalising; it is measured rather than assumed so the
    ground truth reports what is in the record.
    """
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    sig = meta.groupby(["idx_event", "station_id"])["signal_rms"].apply(rms).rename("sig").reset_index()
    noi = noise.groupby("station_id")["rms_band"].apply(rms).rename("noi").reset_index()

    m = sig.merge(noi, on="station_id")
    m["sig"] /= [max(signal_scale[(e, s)], 1e-30) for e, s in zip(m["idx_event"], m["station_id"])]
    m["noi"] /= [max(noise_scale[s], 1e-30) for s in m["station_id"]]
    m["snr_unit"] = m["sig"] / (m["noi"] + 1e-30)

    gt_picks = gt_picks.merge(m[["idx_event", "station_id", "snr_unit"]], on=["idx_event", "station_id"], how="left")
    gt_picks = gt_picks.merge(inj[["inj_id", "amplitude"]], on="inj_id", how="left")
    gt_picks["snr"] = gt_picks["amplitude"] * gt_picks["snr_unit"]
    gt_picks.drop(columns=["snr_unit"], inplace=True)
    return gt_picks


# %%
def write_region(args, events, gt_events, gt_picks):
    """Everything the pipeline reads, in the format it expects."""
    base = f"{args.root_path}/{args.region}"
    src = f"{args.root_path}/{args.src_region}"
    os.makedirs(f"{base}/adloc", exist_ok=True)
    os.makedirs(f"{base}/obspy", exist_ok=True)

    with open(f"{src}/config.json", "r") as fp:
        config = json.load(fp)
    config["starttime"] = gt_events["time"].min().strftime("%Y-%m-%dT%H:%M:%S")
    config["endtime"] = gt_events["time"].max().strftime("%Y-%m-%dT%H:%M:%S")
    config["benchmark"] = {"template_ratio": args.template_ratio, "test_ratio": list(args.test_ratio),
                           "spacing_sec": args.spacing_sec, "normalize": args.normalize}
    with open(f"{base}/config.json", "w") as fp:
        json.dump(config, fp, indent=4)
    for f in ["stations.csv", "stations.json"]:
        if os.path.exists(f"{src}/obspy/{f}"):
            shutil.copy(f"{src}/obspy/{f}", f"{base}/obspy/{f}")

    ## stations: no station terms, so the pipeline's predicted times are the ground truth
    stations = pd.read_csv(f"{src}/adloc/adloc_stations.csv")
    stations = stations[stations["station_id"].isin(gt_picks["station_id"].unique())].copy()
    stations["station_term_time_p"] = 0.0
    stations["station_term_time_s"] = 0.0
    stations["station_term_amplitude"] = 0.0
    stations.to_csv(f"{base}/adloc/adloc_stations.csv", index=False)

    ## only the template block is "known" to the pipeline -- the test block is what it must find
    tmpl_ev = gt_events[gt_events["block"] == "template"].copy()
    tmpl_pk = gt_picks[(gt_picks["block"] == "template") & (gt_picks["injected"])].copy()
    tmpl_ev["event_index"] = np.arange(len(tmpl_ev))
    tmpl_pk = tmpl_pk.merge(tmpl_ev[["inj_id", "event_index"]], on="inj_id")
    ## every injection records which template index the pipeline will know it by
    gt_events["template_event_index"] = gt_events["idx_event"].map(
        tmpl_ev.set_index("idx_event")["event_index"]
    )

    adloc_events = pd.DataFrame(
        {
            "longitude": tmpl_ev["longitude"].values,
            "latitude": tmpl_ev["latitude"].values,
            "depth_km": tmpl_ev["depth_km"].values,
            "time": tmpl_ev["time"].dt.tz_localize(None).astype(str).values,
            "adloc_score": 1.0,
            "adloc_residual_time": 0.0,
            "num_picks": tmpl_pk.groupby("event_index").size().reindex(tmpl_ev["event_index"]).values,
            "magnitude": tmpl_ev["magnitude"].values,
            "adloc_residual_amplitude": 0.0,
            "event_index": tmpl_ev["event_index"].values,
        }
    )
    adloc_events.to_csv(f"{base}/adloc/adloc_events.csv", index=False)

    adloc_picks = pd.DataFrame(
        {
            "station_id": tmpl_pk["station_id"].values,
            "phase_time": tmpl_pk["phase_time"].dt.tz_localize(None).astype(str).values,
            "phase_score": 1.0,
            "phase_type": tmpl_pk["phase_type"].values,
            "phase_amplitude": 1.0,
            "gamma_score": 1.0,
            "adloc_mask": 1,
            "adloc_residual_time": 0.0,
            "adloc_residual_amplitude": 0.0,
            "event_index": tmpl_pk["event_index"].values,
        }
    )
    adloc_picks.to_csv(f"{base}/adloc/adloc_picks.csv", index=False)

    gt_events.to_csv(f"{base}/benchmark/ground_truth_events.csv", index=False)
    gt_picks.to_csv(f"{base}/benchmark/ground_truth_picks.csv", index=False)

    print(f"\nWrote {base}/config.json")
    print(f"Wrote {base}/adloc/adloc_stations.csv ({len(stations)} stations)")
    print(f"Wrote {base}/adloc/adloc_events.csv ({len(adloc_events)} template events)")
    print(f"Wrote {base}/adloc/adloc_picks.csv ({len(adloc_picks)} picks)")
    print(f"Wrote {base}/benchmark/ground_truth_events.csv ({len(gt_events)} injections)")
    print(f"Wrote {base}/benchmark/ground_truth_picks.csv ({len(gt_picks)} picks)")
    print(f"\nNow run:\n"
          f"  python cut_templates_cc.py --region {args.region}\n"
          f"  python run_qtm.py local {args.region}\n"
          f"  python run_qtm_association.py --region {args.region}\n"
          f"  python tests/qtm/evaluate_qtm.py --region {args.region}")


# %%
def plot_schedule(args, gt_events, gt_picks, spacing, t0, t1):
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), width_ratios=[2, 1, 1])

    ax = axes[0]
    mins = (gt_events["time"] - t0).dt.total_seconds() / 60
    ## one norm for both blocks, or the colorbar ends up describing whichever was drawn last
    snr = gt_events["snr_z_median"].replace(0, np.nan)
    norm = matplotlib.colors.LogNorm(vmin=snr.min(), vmax=max(snr.max(), snr.min() * 10))
    for block, marker in [("template", "s"), ("test", "o")]:
        m = gt_events["block"] == block
        sc = ax.scatter(mins[m], gt_events.loc[m, "idx_event"], c=gt_events.loc[m, "snr_z_median"],
                        norm=norm, cmap="viridis", marker=marker, s=40,
                        edgecolor="k", lw=0.4, label=block)
    plt.colorbar(sc, ax=ax, label="median vertical SNR")
    ax.set_xlabel(f"minutes into the record (starts {t0.strftime('%Y-%m-%d %H:%M:%S')})")
    ax.set_ylabel("source event")
    ax.set_title(f"injection schedule: template SNR {args.template_ratio:g}, test SNR "
                 + "/".join(f"{r:g}" for r in args.test_ratio)
                 + f", {len(gt_events)} injections {spacing:.0f} s apart", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ## the spacing is uniform to within float noise, so pin the range or matplotlib zooms
    ## into nanosecond jitter and the axis reads as if the spacing varied
    dt = gt_events["time"].diff().dt.total_seconds().dropna()
    pad = max(0.5, 0.05 * spacing)
    lo, hi = min(float(dt.min()), spacing) - pad, max(float(dt.max()), spacing) + pad
    ax.hist(dt, bins=np.linspace(lo, hi, 21), color="tab:blue", edgecolor="k", lw=0.4)
    ax.axvline(spacing, color="tab:red", lw=1.5)
    ax.set_xlabel("time to the previous injection (s)")
    ax.set_ylabel("injections")
    ax.set_title("inter-event spacing (--spacing_sec)", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    z = gt_picks[(gt_picks["phase_type"] == "P") & gt_picks["injected"]]
    ax.hist(np.log10(z["snr"].dropna() + 1e-6), bins=30, color="tab:orange", edgecolor="k", lw=0.4)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("log10 SNR of each injected station")
    ax.set_ylabel("station-injections")
    ax.set_title("achieved SNR, one bar per set (--template_ratio / --test_ratio)", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{figdir}/bench_01_schedule.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/bench_01_schedule.png")


def plot_record(args, gt_events, noise, t0):
    """The whole record for a few stations, with the injections marked."""
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    base = f"{args.root_path}/{args.region}"
    year_jday = obspy.UTCDateTime(t0.to_pydatetime())
    z = noise[noise["component"].isin(["Z", "3"])].sort_values("rms_band")
    picks = list(z["station_id"])[:4]

    fig, axes = plt.subplots(len(picks), 1, figsize=(14, 2.3 * len(picks)), sharex=True)
    for ax, station_id in zip(np.atleast_1d(axes), picks):
        comp = z[z["station_id"] == station_id]["component"].iloc[0]
        f = f"{base}/waveforms/{year_jday.year}/{year_jday.julday:03d}/{station_id}{comp}.mseed"
        tr = read_trace(f)
        sr = tr.stats.sampling_rate
        d = bandpass(np.asarray(tr.data, dtype=np.float64), sr)
        step = int(sr)  # 1 s envelope so the whole record fits
        n = len(d) // step * step
        env = np.abs(d[:n]).reshape(-1, step).max(axis=1)
        ax.semilogy(np.arange(len(env)) / 60, env + 1e-6, lw=0.5, color="k")
        for _, e in gt_events.iterrows():
            ax.axvline((e["time"] - t0).total_seconds() / 60,
                       color="tab:red" if e["block"] == "test" else "tab:green", lw=0.6, alpha=0.7)
        ax.set_ylabel(f"{station_id}{comp}", fontsize=8)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[-1].set_xlabel("minutes into the record")
    np.atleast_1d(axes)[0].set_title(
        f"benchmark record, {FREQMIN:g}-{FREQMAX:g} Hz 1 s max envelope  |  "
        f"green = template injections (SNR {args.template_ratio:g}), "
        + "red = test injections (SNR " + "/".join(f"{r:g}" for r in args.test_ratio) + ")",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(f"{figdir}/bench_02_record.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/bench_02_record.png")


def plot_zoom(args, gt_events, gt_picks, noise, t0):
    """One test injection, side by side with the noise it was added to."""
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    base = f"{args.root_path}/{args.region}"
    year_jday = obspy.UTCDateTime(t0.to_pydatetime())
    test = gt_events[gt_events["block"] == "test"]
    ev = test.iloc[len(test) // 2]
    pk = gt_picks[(gt_picks["inj_id"] == ev["inj_id"]) & gt_picks["injected"]]
    p = pk[pk["phase_type"] == "P"].set_index("station_id")
    s = pk[pk["phase_type"] == "S"].set_index("station_id")
    zcomp = noise[noise["component"].isin(["Z", "3"])].set_index("station_id")["component"]
    sids = [x for x in p.sort_values("dist_km").index if x in zcomp.index][:8]

    fig, axes = plt.subplots(len(sids), 1, figsize=(12, 1.5 * len(sids)), sharex=True)
    for ax, sid in zip(axes, sids):
        comp = zcomp[sid]
        tp = obspy.UTCDateTime(p.loc[sid, "phase_time"].to_pydatetime())
        pre, post = float(p.loc[sid, "win_pre"]), float(p.loc[sid, "win_post"])
        for path, color, label in [
            (f"{base}/benchmark/noise/{sid}{comp}.mseed", "0.6", "noise only"),
            (f"{base}/waveforms/{year_jday.year}/{year_jday.julday:03d}/{sid}{comp}.mseed", "k", "benchmark"),
        ]:
            tr = read_trace(path)
            sr = tr.stats.sampling_rate
            i0 = max(int(round((tp - pre - 2 - tr.stats.starttime) * sr)), 0)
            i1 = min(int(round((tp + post + 2 - tr.stats.starttime) * sr)), tr.stats.npts)
            d = bandpass(np.asarray(tr.data[i0:i1], dtype=np.float64), sr)
            ax.plot(-pre - 2 + np.arange(len(d)) / sr, d, lw=0.5, color=color, label=label)
        ax.axvline(0, color="tab:red", lw=1)
        ax.axvline(float((s.loc[sid, "phase_time"] - p.loc[sid, "phase_time"]).total_seconds()), color="tab:blue", lw=1)
        ax.set_ylabel(f"{sid}{comp}\nSNR={p.loc[sid,'snr']:.1f}", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("time relative to the ground-truth P (s)")
    axes[0].set_title(
        f"test injection {int(ev['inj_id'])} (source event {int(ev['idx_event'])}, M{ev['magnitude']:.2f}) "
        f"at {ev['time'].strftime('%H:%M:%S')}  |  {FREQMIN:g}-{FREQMAX:g} Hz, red=P blue=S",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(f"{figdir}/bench_03_zoom.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/bench_03_zoom.png")


def plot_snr(args, gt_picks):
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    z = gt_picks[(gt_picks["phase_type"] == "P") & gt_picks["injected"] & gt_picks["snr"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for idx, g in z.groupby("idx_event"):
        axes[0].semilogy(g["dist_km"], g["snr"], ".", ms=4, label=f"{int(idx)}")
    axes[0].axhline(1, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("distance (km)")
    axes[0].set_ylabel("SNR in the benchmark record")
    axes[0].set_title(f"achieved SNR: template {args.template_ratio:g}, test "
                      + "/".join(f"{r:g}" for r in args.test_ratio), fontsize=10)
    axes[0].legend(fontsize=6, ncol=2, title="source event")
    axes[0].grid(alpha=0.3, which="both")

    med = z.groupby("idx_event")["snr"].median()
    n_ok = z[z["snr"] >= 3].groupby("idx_event").size().reindex(med.index).fillna(0)
    ax = axes[1]
    ax.bar(med.index - 0.2, med.values, width=0.4, color="tab:blue", label="median SNR")
    ax.set_xlabel("source event")
    ax.set_ylabel("median SNR", color="tab:blue")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.bar(med.index + 0.2, n_ok.values, width=0.4, color="tab:orange")
    ax2.set_ylabel("stations with SNR >= 3", color="tab:orange")
    ax.set_title("per source event", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(f"{figdir}/bench_04_snr.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/bench_04_snr.png")


# %%
if __name__ == "__main__":
    args = parse_args()
    base = f"{args.root_path}/{args.region}"

    events = pd.read_csv(f"{base}/benchmark/source_events.csv", parse_dates=["time"])
    events["time"] = pd.to_datetime(events["time"], utc=True)
    win = pd.read_csv(f"{base}/benchmark/event_windows.csv")
    meta = win.copy()
    ## tp/ts are per station, not per component -- one row each is enough to place the cuts
    win = win.drop_duplicates(subset=["idx_event", "station_id"])
    noise = pd.read_csv(f"{base}/benchmark/noise_stats.csv")
    noise["starttime"] = pd.to_datetime(noise["starttime"], utc=True, format="ISO8601")
    noise = noise[noise["station_id"].isin(win["station_id"].unique())]

    ## A station whose cut is flat for any source event carries no signal there -- obspy
    ## fills gaps with the last sample, which demeans to exactly zero.  Such a station can
    ## only ever show up as a hole in the ground truth, so drop it outright: keeping it
    ## means every recovery number is quietly averaged over stations that had nothing to
    ## find.  --keep_dead_stations restores the old behaviour of marking them injected=False.
    dead = sorted(meta.loc[meta["signal_rms"] <= 0, "station_id"].unique())
    if dead and not args.keep_dead_stations:
        before = win["station_id"].nunique()
        win = win[~win["station_id"].isin(dead)]
        meta = meta[~meta["station_id"].isin(dead)]
        noise = noise[~noise["station_id"].isin(dead)]
        print(f"dropped {len(dead)} station(s) flat in at least one source event "
              f"({', '.join(dead)}): {before} -> {win['station_id'].nunique()} stations")
    elif dead:
        print(f"keeping {len(dead)} station(s) with a flat cut: {', '.join(dead)}")

    inj, spacing, t0, t1 = schedule(args, events, win, noise)
    noise_scale, signal_scale = normalization(args, meta, noise)
    gt_picks = inject(args, events, win, noise, inj, noise_scale, signal_scale)
    gt_picks = measure_snr(args, gt_picks, meta, noise, inj, noise_scale, signal_scale)

    gt_events = inj.merge(
        events[["idx_event", "src_event_index", "event_id", "longitude", "latitude", "depth_km", "magnitude"]],
        on="idx_event",
    )
    snr_z = gt_picks[(gt_picks["phase_type"] == "P") & gt_picks["injected"]].groupby("inj_id")["snr"].median()
    gt_events["snr_z_median"] = gt_events["inj_id"].map(snr_z)
    gt_events["num_stations"] = gt_events["inj_id"].map(
        gt_picks[(gt_picks["phase_type"] == "P") & gt_picks["injected"]].groupby("inj_id").size()
    )
    gt_events["template_ratio"] = args.template_ratio
    gt_events["normalize"] = args.normalize
    gt_events = gt_events.sort_values("time").reset_index(drop=True)

    write_region(args, events, gt_events, gt_picks)

    print("\nPlotting:")
    plot_schedule(args, gt_events, gt_picks, spacing, t0, t1)
    plot_record(args, gt_events, noise, t0)
    plot_zoom(args, gt_events, gt_picks, noise, t0)
    plot_snr(args, gt_picks)

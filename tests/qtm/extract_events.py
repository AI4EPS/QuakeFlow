# %%
"""
Step 2 of the QTM benchmark: cut 10 test events out of the real waveforms.

Events are taken from the AdLoc catalogue, preferring well-recorded ones that are isolated
in time (so the cut is not contaminated by a neighbour) and spread over magnitude.  The
cut window is defined by *theoretical* arrivals, not by the picks:

    vp = 6 km/s, vs = vp / 1.73, straight ray from the hypocentre to the station
    window = [tp - win_lead_sec,  tp + win_sp_factor * (ts - tp) + win_tail_sec]

Those same theoretical times become the ground-truth picks of the benchmark, so whatever
is written into the record is exactly where the pipeline is told to look.

Everything comes from AdLoc: the catalogue says which events exist, how well recorded they
are, and where they are.  --hypocenter fit (refit each event from its own AdLoc picks under
this vp/vs) was the default while the local/demo catalogue was 21 km off; that is fixed, the
refit now moves events only ~0.4 km, so the catalogue value is used directly.  SCEDC is read
only as an independent check.  Note the AdLoc depths are still ~6 km deep -- see the
velocity-model note in README.md; this shifts the theoretical arrivals but not the benchmark,
since the same times position the template and every injection.

    python tests/qtm/extract_events.py --src_region demo --region synthetic

Writes {region}/benchmark/events/event_XX.npz (one array per channel), source_events.csv,
event_windows.csv, and figures to {region}/benchmark/figures/.
"""

import argparse
import json
import os
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from pyproj import Proj
from scipy.signal import butter, sosfiltfilt
from paths import DEFAULT_ROOT
from mseed import read_trace

FREQMIN, FREQMAX = 2.0, 12.0
TAPER_SEC = 0.25  # cosine ramp on each end of the cut, so splicing does not make a step


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--src_region", type=str, default="demo")
    parser.add_argument("--region", type=str, default="synthetic")
    parser.add_argument("--num_events", type=int, default=10, help="number of source events to cut")
    parser.add_argument("--min_gap", type=float, default=20.0, help="min seconds to the nearest catalogue event")
    parser.add_argument("--min_picks", type=int, default=40, help="min number of AdLoc picks")
    parser.add_argument("--mag_min", type=float, default=1.0)
    parser.add_argument("--mag_max", type=float, default=4.0)
    parser.add_argument("--vp", type=float, default=6.0)
    parser.add_argument("--vp_vs_ratio", type=float, default=1.73)
    parser.add_argument("--win_lead_sec", type=float, default=3.0,
                        help="cut starts this many seconds before the theoretical P")
    parser.add_argument("--win_sp_factor", type=float, default=4.0,
                        help="cut ends this many S-P times after P, so the window scales with distance")
    parser.add_argument("--win_tail_sec", type=float, default=0.0,
                        help="extra seconds added to the end of the cut")
    parser.add_argument("--hypocenter", type=str, default="adloc", choices=["adloc", "fit", "scedc"],
                        help="ground-truth hypocentre: the AdLoc catalogue value (default), a refit from the "
                             "AdLoc picks under this vp/vs, or the SCEDC value. 'fit' and 'scedc' are kept "
                             "for cross-checks")
    parser.add_argument("--min_separation_km", type=float, default=5.0,
                        help="warn if two selected sources end up closer than this")
    parser.add_argument("--min_snr", type=float, default=8.0,
                        help="min SNR against the local background, measured on the probe stations")
    parser.add_argument("--probe_stations", type=str, default="CI.CLC..HHZ,CI.SLA..HHZ,CI.WRC2..HHZ,CI.WMF..HHZ",
                        help="channels used to screen candidates for contamination by neighbouring events")
    parser.add_argument("--max_event_figures", type=int, default=2,
                        help="per-source cut figures to write (-1 for all); cut_summary.png covers the rest")
    parser.add_argument("--no_time_shift", action="store_true",
                        help="do not align the origin time to the median pick residual")
    return parser.parse_args()


def bandpass(x, sr, fmin=FREQMIN, fmax=FREQMAX):
    sos = butter(4, [fmin, min(fmax, 0.45 * sr)], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


# %%
def select_events(args, stations):
    """Well-recorded, temporally isolated events, spread out in space."""
    events = pd.read_csv(f"{args.root_path}/{args.src_region}/adloc/adloc_events.csv", parse_dates=["time"])
    events["time"] = pd.to_datetime(events["time"], utc=True)
    events = events.sort_values("time").reset_index(drop=True)

    dt_prev = events["time"].diff().dt.total_seconds().fillna(np.inf)
    dt_next = (-events["time"].diff(-1).dt.total_seconds()).fillna(np.inf)
    events["gap_sec"] = np.minimum(dt_prev, dt_next)

    ## attach the SCEDC hypocentre as an independent check on the AdLoc one
    catalog = pd.read_csv(f"{args.root_path}/{args.src_region}/obspy/catalog.csv")
    catalog["time"] = pd.to_datetime(catalog["time"], utc=True)
    catalog = catalog[["time", "event_id", "longitude", "latitude", "depth_km", "magnitude"]].sort_values("time")
    events = pd.merge_asof(
        events, catalog, on="time", direction="nearest",
        tolerance=pd.Timedelta("3s"), suffixes=("_adloc", "_scedc"),
    )
    matched = events["event_id"].notna()
    dx = (events.loc[matched, "longitude_adloc"] - events.loc[matched, "longitude_scedc"]).median()
    dy = (events.loc[matched, "latitude_adloc"] - events.loc[matched, "latitude_scedc"]).median()
    dz = (events.loc[matched, "depth_km_adloc"] - events.loc[matched, "depth_km_scedc"]).median()
    km = np.hypot(dx * 111.19 * np.cos(np.deg2rad(events["latitude_adloc"].median())), dy * 111.19)
    print(f"{matched.sum()}/{len(events)} AdLoc events matched to SCEDC within 3 s")
    print(f"median AdLoc-SCEDC offset: {dx:+.4f} lon, {dy:+.4f} lat ({km:.1f} km horizontal), {dz:+.2f} km depth")
    if km > 2:
        print(f"  *** the AdLoc catalogue is mislocated by {km:.0f} km -- using --hypocenter={args.hypocenter}")

    cand = events[
        matched
        & (events["gap_sec"] >= args.min_gap)
        & (events["num_picks"] >= args.min_picks)
        & (events["magnitude_adloc"] >= args.mag_min)
        & (events["magnitude_adloc"] <= args.mag_max)
    ].copy()
    print(f"{len(cand)} candidate events (gap>={args.min_gap:g}s, picks>={args.min_picks}, "
          f"{args.mag_min:g}<=M<={args.mag_max:g}, SCEDC match)")

    ## an event sitting in the coda of a bigger one would carry that coda into the benchmark
    cand["probe_snr"] = probe_snr(args, cand)
    cand = cand[cand["probe_snr"] >= args.min_snr]
    print(f"{len(cand)} left after requiring SNR >= {args.min_snr:g} against the local background")
    if len(cand) < args.num_events:
        raise SystemExit(f"only {len(cand)} candidates, need {args.num_events} -- relax the selection")

    src = "scedc" if args.hypocenter == "scedc" else "adloc"
    for col in ["longitude", "latitude", "depth_km"]:
        cand[col] = cand[f"{col}_{src}"]
    cand["magnitude"] = cand["magnitude_scedc"].fillna(cand["magnitude_adloc"])

    proj = make_proj(args)
    cand = cand.reset_index(drop=True)
    if args.hypocenter == "fit":
        cand = relocate(args, cand, stations, proj)
    else:
        cand["x_km"], cand["y_km"] = proj(cand["longitude"].values, cand["latitude"].values)
        cand["z_km"] = cand["depth_km"]

    ## Spread the sources out in space: nearby events have similar waveforms, and a template
    ## matching benchmark cannot tell them apart. Farthest-point sampling from the strongest
    ## candidate maximises the smallest separation in the set.
    xyz = cand[["x_km", "y_km", "z_km"]].values

    chosen = [int(cand["probe_snr"].idxmax())]
    dmin = np.linalg.norm(xyz - xyz[chosen[0]], axis=-1)
    while len(chosen) < args.num_events:
        k = int(np.argmax(dmin))
        chosen.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(xyz - xyz[k], axis=-1))

    sel = cand.loc[chosen].sort_values("time").reset_index(drop=True)
    sel["idx_event"] = np.arange(len(sel))
    sel.rename(columns={"event_index": "src_event_index"}, inplace=True)

    sep = np.linalg.norm(
        sel[["x_km", "y_km", "z_km"]].values[:, None, :] - sel[["x_km", "y_km", "z_km"]].values[None, :, :], axis=-1
    )
    np.fill_diagonal(sep, np.inf)
    sel["min_sep_km"] = sep.min(axis=1)
    print(f"source separation: min {sep.min():.1f} km, median nearest-neighbour {np.median(sel['min_sep_km']):.1f} km")
    if sep.min() < args.min_separation_km:
        print(f"  WARNING: closest pair is {sep.min():.1f} km apart (< --min_separation_km "
              f"{args.min_separation_km:g}); their waveforms may be hard to tell apart")
    print(sel[["idx_event", "src_event_index", "event_id", "time", "magnitude", "num_picks", "gap_sec",
               "min_sep_km"]].to_string(index=False))
    return sel, events


def make_proj(args):
    """Same projection as cut_templates_cc.py, so x/y/z agree with the pipeline."""
    with open(f"{args.root_path}/{args.src_region}/config.json", "r") as fp:
        config = json.load(fp)
    lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
    lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
    return Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0}  +units=km")


def load_stations(args, proj=None):
    """AdLoc stations that actually have waveforms, projected."""
    stations = pd.read_csv(f"{args.root_path}/{args.src_region}/adloc/adloc_stations.csv")
    files = sorted(glob(f"{args.root_path}/{args.src_region}/waveforms/????/???/*.mseed"))
    have = {os.path.basename(f).replace(".mseed", "")[:-1] for f in files}
    stations = stations[stations["station_id"].isin(have)].reset_index(drop=True)
    proj = proj if proj is not None else make_proj(args)
    stations["x_km"], stations["y_km"] = proj(stations["longitude"].values, stations["latitude"].values)
    stations["z_km"] = -stations["elevation_m"] / 1e3
    return stations


def load_picks(args):
    picks = pd.read_csv(f"{args.root_path}/{args.src_region}/adloc/adloc_picks.csv", parse_dates=["phase_time"])
    picks = picks[picks["adloc_mask"] == 1].copy()
    picks["phase_time"] = pd.to_datetime(picks["phase_time"], utc=True)
    return picks


# %%
def relocate(args, cand, stations, proj):
    """Refit hypocentre and origin time from the AdLoc picks, under the same straight-ray
    vp/vs the cut windows use.

    This keeps the benchmark self-contained -- the ground truth comes from AdLoc's own picks
    and nothing else -- and makes the theoretical arrivals consistent with the model that
    places them, which a catalogue location from a different velocity model would not be.
    """
    picks = load_picks(args)
    picks = picks[picks["event_index"].isin(cand["event_index"])]
    picks = picks.merge(stations[["station_id", "x_km", "y_km", "z_km"]], on="station_id")
    vs = args.vp / args.vp_vs_ratio
    origin = cand.set_index("event_index")["time"].to_dict()

    def best(grid, sta, v, obs):
        d = np.linalg.norm(grid[:, None, :] - sta[None, :, :], axis=-1)
        r = obs[None, :] - d / v
        r = r - np.median(r, axis=1, keepdims=True)
        rms = np.sqrt((r**2).mean(axis=1))
        return int(np.argmin(rms)), rms

    def axes(c, half, step):
        return [np.arange(c[i] - half, c[i] + half + step / 2, step) for i in range(3)]

    out = []
    for event_index, g in picks.groupby("event_index"):
        sta = g[["x_km", "y_km", "z_km"]].values
        v = np.where(g["phase_type"].values == "P", args.vp, vs)
        obs = (g["phase_time"] - origin[event_index]).dt.total_seconds().values
        c = np.array([0.0, 0.0, 8.0])
        for half, step in [(60.0, 3.0), (4.0, 0.5), (0.6, 0.1)]:
            ax, ay, az = axes(c, half, step)
            az = az[(az >= 0) & (az <= 30)] if len(az[(az >= 0) & (az <= 30)]) else np.array([max(c[2], 0.0)])
            gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
            grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
            k, rms = best(grid, sta, v, obs)
            c = grid[k]
        d = np.linalg.norm(c[None, :] - sta, axis=-1)
        dt = float(np.median(obs - d / v))
        out.append({"event_index": event_index, "x_km": c[0], "y_km": c[1], "z_km": c[2],
                    "fit_rms": float(rms[k]), "fit_dt": dt})

    fit = pd.DataFrame(out)
    fit["longitude"], fit["latitude"] = proj(fit["x_km"].values, fit["y_km"].values, inverse=True)
    fit["depth_km"] = fit["z_km"]
    cand = cand.drop(columns=["longitude", "latitude", "depth_km"]).merge(fit, on="event_index")
    cand["time"] = cand["time"] + pd.to_timedelta(cand["fit_dt"], unit="s")

    moved = np.linalg.norm(
        cand[["x_km", "y_km"]].values - np.stack(proj(cand["longitude_adloc"].values,
                                                      cand["latitude_adloc"].values), axis=1), axis=-1
    )
    ref = np.stack(proj(cand["longitude_scedc"].values, cand["latitude_scedc"].values), axis=1)
    to_scedc = np.linalg.norm(cand[["x_km", "y_km"]].values - ref, axis=-1)
    print(f"refit {len(cand)} events from their AdLoc picks with vp={args.vp:g}, vs={vs:.2f}:")
    print(f"  travel-time rms   {cand['fit_rms'].median():.2f} s (median)")
    print(f"  moved from AdLoc  {np.median(moved):.1f} km (median)")
    print(f"  vs SCEDC          {np.median(to_scedc):.1f} km (median) -- independent check, not used")
    return cand


# %%
def probe_snr(args, cand):
    """Peak amplitude of each candidate over the background just before it, on a few channels.

    The aftershock sequence is dense enough that a small event can sit inside the coda of a
    bigger one; such a cut would inject that coda along with the event.
    """
    snr = []
    for name in args.probe_stations.split(","):
        f = glob(f"{args.root_path}/{args.src_region}/waveforms/????/???/{name}.mseed")
        if not f:
            print(f"  probe channel {name} not found, skipped")
            continue
        trace = read_trace(f[0])
        sr = trace.stats.sampling_rate
        d = np.asarray(trace.data, dtype=np.float64)
        if np.ma.isMaskedArray(d):
            d = d.filled(0.0)
        env = np.abs(bandpass(d - d.mean(), sr))
        t0 = pd.Timestamp(trace.stats.starttime.datetime, tz="UTC")
        i = ((cand["time"] - t0).dt.total_seconds() * sr).values

        def window(lo, hi):
            a = np.clip((i + lo * sr).astype(int), 0, len(env) - 1)
            b = np.clip((i + hi * sr).astype(int), 1, len(env))
            return a, b

        a, b = window(-60, -20)
        back = np.array([np.median(env[x:y]) if y > x else np.nan for x, y in zip(a, b)])
        a, b = window(0, 40)
        peak = np.array([np.max(env[x:y]) if y > x else np.nan for x, y in zip(a, b)])
        snr.append(peak / (back + 1e-30))
        print(f"  probe {name}: median SNR {np.nanmedian(snr[-1]):.1f}")
    return np.nanmedian(np.array(snr), axis=0)


# %%
def theoretical_windows(args, events, stations):
    """Straight-ray P/S times and the cut window for every event-station pair."""

    vs = args.vp / args.vp_vs_ratio
    rows = []
    for _, ev in events.iterrows():
        d = np.linalg.norm(
            stations[["x_km", "y_km", "z_km"]].values - np.array([ev.x_km, ev.y_km, ev.z_km]), axis=-1
        )
        for (_, sta), dist in zip(stations.iterrows(), d):
            tp, ts = dist / args.vp, dist / vs
            rows.append(
                {
                    "idx_event": int(ev["idx_event"]),
                    "src_event_index": int(ev["src_event_index"]),
                    "station_id": sta["station_id"],
                    "sampling_rate": np.nan,  # filled while cutting
                    "dist_km": dist,
                    "tp_tt": tp,
                    "ts_tt": ts,
                    "win_pre": args.win_lead_sec,
                    "win_post": args.win_sp_factor * (ts - tp) + args.win_tail_sec,
                }
            )
    win = pd.DataFrame(rows)
    win["win_sec"] = win["win_pre"] + win["win_post"]
    print(f"{len(win)} event-station windows, "
          f"length {win['win_sec'].min():.1f}-{win['win_sec'].max():.1f} s (median {win['win_sec'].median():.1f} s)")

    resid = pick_residuals(args, events, win)
    if not args.no_time_shift:
        shift = resid.groupby("idx_event")["resid"].median()
        events["time_shift_sec"] = events["idx_event"].map(shift).fillna(0.0)
        events["time"] = events["time"] + pd.to_timedelta(events["time_shift_sec"], unit="s")
        resid["resid"] -= resid["idx_event"].map(shift)
        print("origin times nudged so the theoretical arrivals sit on the AdLoc picks:")
        print(events[["idx_event", "time_shift_sec"]].round(3).to_string(index=False))
    else:
        events["time_shift_sec"] = 0.0
    print(f"pick residual after alignment: mean {resid['resid'].mean():+.2f} s, std {resid['resid'].std():.2f} s")
    return win, stations, events, resid


def pick_residuals(args, events, win):
    """Observed AdLoc pick minus theoretical arrival, for the selected events only."""
    picks = load_picks(args).merge(
        events[["idx_event", "src_event_index", "time"]], left_on="event_index", right_on="src_event_index"
    )
    picks = picks.merge(win[["idx_event", "station_id", "dist_km", "tp_tt", "ts_tt"]], on=["idx_event", "station_id"])
    picks["pred_tt"] = np.where(picks["phase_type"] == "P", picks["tp_tt"], picks["ts_tt"])
    picks["obs_tt"] = (picks["phase_time"] - picks["time"]).dt.total_seconds()
    picks["resid"] = picks["obs_tt"] - picks["pred_tt"]
    return picks[["idx_event", "station_id", "phase_type", "dist_km", "obs_tt", "pred_tt", "resid"]]


# %%
def cut(args, events, win):
    """One pass over the day files; every event is cut out of each channel as it is read."""
    out = f"{args.root_path}/{args.region}/benchmark"
    os.makedirs(f"{out}/events", exist_ok=True)

    files = sorted(glob(f"{args.root_path}/{args.src_region}/waveforms/????/???/*.mseed"))
    origin = events.set_index("idx_event")["time"].to_dict()
    key = win.set_index(["idx_event", "station_id"])

    cuts = {int(i): {} for i in events["idx_event"]}
    meta = []
    for f in files:
        name = os.path.basename(f).replace(".mseed", "")
        station_id, component = name[:-1], name[-1]
        if (win["station_id"] == station_id).sum() == 0:
            continue
        trace = read_trace(f)
        sr = trace.stats.sampling_rate
        data = np.asarray(trace.data, dtype=np.float64)
        if np.ma.isMaskedArray(data):
            data = data.filled(0.0)
        t_start = trace.stats.starttime

        for idx_event in cuts:
            row = key.loc[(idx_event, station_id)]
            tp_abs = obspy.UTCDateTime(origin[idx_event].to_pydatetime()) + row["tp_tt"]
            i0 = int(round((tp_abs - row["win_pre"] - t_start) * sr))
            i1 = i0 + int(round(row["win_sec"] * sr))
            if i0 < 0 or i1 > len(data):
                continue
            seg = data[i0:i1] - data[i0:i1].mean()

            ntaper = max(2, int(round(TAPER_SEC * sr)))
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ntaper) / ntaper))
            seg[:ntaper] *= ramp
            seg[-ntaper:] *= ramp[::-1]

            ## SNR in the band the templates are built in: everything after P over the pre-P noise.
            ## This is the SNR the cut had in the source record; the SNR it ends up with in the
            ## benchmark depends on the set ratio and is computed in make_benchmark.py.
            filt = bandpass(seg, sr)
            n_pre = int(round((row["win_pre"] - 0.5) * sr))
            i_p = int(round(row["win_pre"] * sr))
            noise_rms = np.std(filt[:n_pre]) if n_pre > 10 else np.nan
            signal_rms = np.std(filt[i_p:])
            snr = np.max(np.abs(filt[i_p:])) / (noise_rms + 1e-30)

            cuts[idx_event][name] = seg.astype(np.float32)
            meta.append(
                {
                    "idx_event": idx_event,
                    "station_id": station_id,
                    "channel": name.split(".")[3],
                    "component": component,
                    "sampling_rate": sr,
                    "npts": len(seg),
                    "dist_km": row["dist_km"],
                    "tp_tt": row["tp_tt"],
                    "ts_tt": row["ts_tt"],
                    "win_pre": row["win_pre"],
                    "win_post": row["win_post"],
                    "peak": float(np.max(np.abs(seg))),
                    "signal_rms": float(signal_rms),
                    "peak_band": float(np.max(np.abs(filt[i_p:]))),
                    "noise_rms": float(noise_rms),
                    "snr": float(snr),
                    "snr_rms": float(signal_rms / (noise_rms + 1e-30)),
                }
            )
        print(f"  cut {name:20s} {sr:5.0f} Hz for {sum(name in c for c in cuts.values())} events")

    for idx_event, arrays in cuts.items():
        np.savez_compressed(f"{out}/events/event_{idx_event:02d}.npz", **arrays)
    meta = pd.DataFrame(meta).sort_values(["idx_event", "station_id", "component"])
    meta.to_csv(f"{out}/event_windows.csv", index=False)
    events.to_csv(f"{out}/source_events.csv", index=False)
    print(f"\nWrote {len(cuts)} event files -> {out}/events/")
    print(f"Wrote {out}/event_windows.csv ({len(meta)} channel cuts)")
    print(f"Wrote {out}/source_events.csv")

    ## obspy fills gaps with the last sample, so a dead stretch of the source record reads as
    ## a constant run and demeans to exactly zero -- there is nothing there to inject
    dead = meta[meta["signal_rms"] == 0]
    if len(dead) > 0:
        print(f"\nNOTE: {len(dead)}/{len(meta)} cuts are flat in the source record (nothing to inject):")
        print(dead.groupby(["idx_event", "station_id"]).size().rename("channels").to_string())

    ## the templates cut_templates_cc.py builds run to ts + 3.7 s; flag windows that stop short
    z = meta[meta["component"].isin(["Z", "3"])]
    short = z[z["win_post"] < (z["ts_tt"] - z["tp_tt"]) + 3.7]
    if len(short) > 0:
        print(f"\nNOTE: {len(short)}/{len(z)} vertical cuts end before the S template window does "
              f"(needs ts-tp >= 1.24 s, i.e. dist >= ~10 km). Use --win_tail_sec to extend.")
        print(short.groupby("idx_event")["dist_km"].agg(["count", "max"]).to_string())
    return meta


# %%
def plot_events(args, events, meta):
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    os.makedirs(figdir, exist_ok=True)
    out = f"{args.root_path}/{args.region}/benchmark"

    ## cut_summary.png / cut_geometry.png carry the aggregate; the per-source sections are
    ## ~1 MB each and mostly look alike, so only a couple are written unless asked
    sel = events if args.max_event_figures < 0 else events.head(args.max_event_figures)
    if len(sel) < len(events):
        print(f"  ({len(sel)} of {len(events)} per-source figures; "
              f"--max_event_figures -1 for all)")

    for _, ev in sel.iterrows():
        idx = int(ev["idx_event"])
        arrays = np.load(f"{out}/events/event_{idx:02d}.npz")
        m = meta[(meta["idx_event"] == idx) & (meta["component"].isin(["Z", "3"]))].sort_values("dist_km")
        if len(m) == 0:
            continue
        fig, ax = plt.subplots(figsize=(11, max(6, 0.3 * len(m))))
        t1 = float(m["win_post"].max())
        for k, row in enumerate(m.itertuples()):
            name = f"{row.station_id}{row.component}"
            if name not in arrays:
                continue
            sr = row.sampling_rate
            d = bandpass(np.asarray(arrays[name], dtype=np.float64), sr)
            amp = np.max(np.abs(d))
            if amp == 0:
                continue
            ax.plot(-row.win_pre + np.arange(len(d)) / sr, 0.45 * d / amp + k, lw=0.4, color="k")
            ts_rel = row.ts_tt - row.tp_tt
            ax.plot([0, 0], [k - 0.45, k + 0.45], color="tab:red", lw=1.2)
            ax.plot([ts_rel, ts_rel], [k - 0.45, k + 0.45], color="tab:blue", lw=1.2)
            for edge in (-row.win_pre, row.win_post):
                ax.plot([edge, edge], [k - 0.3, k + 0.3], color="0.6", lw=0.8)
            ax.text(-row.win_pre + 0.05, k + 0.12,
                    f"{name}  {row.dist_km:.1f} km  S-P={ts_rel:.1f}s  SNR={row.snr:.1f}", fontsize=5.5)
        ax.axvline(0, color="tab:red", lw=0.5, alpha=0.4)
        ax.set_xlim(-args.win_lead_sec - 0.5, t1 + 0.5)
        ax.set_ylim(-1, len(m))
        ax.set_yticks([])
        ax.set_xlabel("time relative to theoretical P (s)")
        ax.set_title(
            f"cut event {idx} (adloc #{int(ev['src_event_index'])})  M{ev['magnitude']:.2f}  {ev['time']}  |  "
            f"{FREQMIN:g}-{FREQMAX:g} Hz, vertical, sorted by distance  |  "
            f"red=P blue=S (vp={args.vp:g}), grey = window ends",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(f"{figdir}/cut_event_{idx:02d}.png", dpi=160)
        plt.close(fig)
        print(f"  {figdir}/cut_event_{idx:02d}.png")

    ## summary
    z = meta[meta["component"].isin(["Z", "3"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for idx, g in z.groupby("idx_event"):
        axes[0].semilogy(g["dist_km"], g["snr"], ".", ms=4, label=f"{idx}")
    axes[0].axhline(10, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("distance (km)")
    axes[0].set_ylabel(f"SNR ({FREQMIN:g}-{FREQMAX:g} Hz)")
    axes[0].set_title("peak SNR of the cuts in the source record")
    axes[0].legend(fontsize=6, ncol=2, title="event")
    axes[0].grid(alpha=0.3, which="both")

    med = z.groupby("idx_event")["snr"].median()
    axes[1].scatter(events["magnitude"], med.reindex(events["idx_event"]).values, c="tab:red")
    for _, e in events.iterrows():
        axes[1].annotate(int(e["idx_event"]), (e["magnitude"], med[int(e["idx_event"])]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("magnitude")
    axes[1].set_ylabel("median vertical SNR")
    axes[1].set_title("magnitude vs SNR of the 10 cuts")
    axes[1].grid(alpha=0.3, which="both")

    axes[2].scatter(z["dist_km"], z["win_pre"] + z["win_post"], s=6, c="tab:blue")
    axes[2].set_xlabel("distance (km)")
    axes[2].set_ylabel("window length (s)")
    axes[2].set_title(f"cut window = {args.win_lead_sec:g} s + {args.win_sp_factor:g}*(ts-tp)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{figdir}/cut_summary.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/cut_summary.png")


# %%
def plot_geometry(args, events, all_events, stations, resid):
    """Two things worth checking: are the hypocentres sane, and do vp=6 arrivals match the picks."""
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))

    ax = axes[0]
    ok = all_events["event_id"].notna()
    ax.scatter(all_events.loc[ok, "longitude_scedc"], all_events.loc[ok, "latitude_scedc"],
               s=4, c="tab:green", alpha=0.4, label="SCEDC catalogue")
    ax.scatter(all_events.loc[ok, "longitude_adloc"], all_events.loc[ok, "latitude_adloc"],
               s=4, c="tab:purple", alpha=0.4, label="AdLoc catalogue")
    ax.scatter(stations["longitude"], stations["latitude"], marker="v", s=30, c="tab:blue", label="station")
    ax.scatter(events["longitude"], events["latitude"], s=45, c="tab:red", edgecolor="k", lw=0.5,
               label="benchmark event", zorder=5)
    for _, e in events.iterrows():
        ax.annotate(int(e["idx_event"]), (e["longitude"], e["latitude"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points", zorder=6)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"hypocentres (benchmark uses --hypocenter={args.hypocenter})", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_aspect(1 / np.cos(np.deg2rad(stations["latitude"].mean())))

    ax = axes[1]
    for ph, color in [("P", "tab:red"), ("S", "tab:blue")]:
        r = resid[resid["phase_type"] == ph]
        ax.scatter(r["dist_km"], r["obs_tt"], s=5, alpha=0.5, color=color, label=f"{ph} pick")
        d = np.linspace(0, resid["dist_km"].max(), 2)
        v = args.vp if ph == "P" else args.vp / args.vp_vs_ratio
        ax.plot(d, d / v, color=color, lw=1.2, ls="--", label=f"{ph}: d/{v:.2f}")
    ax.set_xlabel("distance (km)")
    ax.set_ylabel("travel time (s)")
    ax.set_title("AdLoc picks vs the straight-ray model used for the cuts", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for ph, color in [("P", "tab:red"), ("S", "tab:blue")]:
        r = resid[resid["phase_type"] == ph]["resid"]
        ax.hist(r, bins=40, alpha=0.55, color=color, label=f"{ph}: {r.mean():+.2f} +/- {r.std():.2f} s")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(-args.win_lead_sec, color="k", ls="--", lw=1)
    ax.set_xlabel("observed pick - theoretical arrival (s)")
    ax.set_ylabel("picks")
    ax.set_title(f"residuals (dashed = start of the {args.win_lead_sec:g} s pre-P window)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    xyz = events[["x_km", "y_km", "z_km"]].values
    sep = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    im = ax.imshow(sep, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax, label="separation (km)")
    for i in range(len(sep)):
        for j in range(len(sep)):
            if i != j:
                ax.text(j, i, f"{sep[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="w" if sep[i, j] < sep.max() * 0.6 else "k")
    ax.set_xticks(range(len(sep)))
    ax.set_yticks(range(len(sep)))
    ax.set_xlabel("source event")
    ax.set_ylabel("source event")
    off = sep + np.diag(np.full(len(sep), np.inf))
    ax.set_title(f"source separation (min {off.min():.1f} km)", fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{figdir}/cut_geometry.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/cut_geometry.png")


# %%
if __name__ == "__main__":
    args = parse_args()
    stations = load_stations(args)
    print(f"{len(stations)} stations with waveforms")
    events, all_events = select_events(args, stations)
    win, stations, events, resid = theoretical_windows(args, events, stations)
    meta = cut(args, events, win)
    print("\nPlotting:")
    plot_geometry(args, events, all_events, stations, resid)
    plot_events(args, events, meta)

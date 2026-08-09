# %%
"""
Step 1 of the QTM benchmark: extract a noise-only sequence.

The Ridgecrest M6.4 foreshock is at 2019-07-04T17:33:49, and the AdLoc catalogue lists
only 34 events in the 17 hours before it.  Between 11:52 and 15:21 there is no catalogued
event at all, so a window inside that gap is pure background noise for every station.

Writes one mseed per channel to {region}/benchmark/noise/ (data replaced, headers kept),
a per-channel noise table, and figures for eyeballing the result.

    python tests/qtm/extract_noise.py --src_region demo --region synthetic

Figures go to {region}/benchmark/figures/.
"""

import argparse
import os
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from paths import DEFAULT_ROOT

FREQMIN, FREQMAX = 2.0, 12.0  # the band cut_templates_cc.py uses for the templates


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--src_region", type=str, default="demo", help="region the real waveforms come from")
    parser.add_argument("--region", type=str, default="synthetic", help="region the benchmark is written to")
    parser.add_argument("--start", type=str, default="2019-07-04T12:00:00", help="start of the noise window (UTC)")
    parser.add_argument("--duration", type=float, default=3600.0, help="length of the noise window in seconds")
    parser.add_argument("--ref_station", type=str, default="CI.CLC..HHZ", help="station for the day-long overview")
    return parser.parse_args()


def bandpass(x, sr, fmin=FREQMIN, fmax=FREQMAX):
    sos = butter(4, [fmin, min(fmax, 0.45 * sr)], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


# %%
def extract(args):
    src = f"{args.root_path}/{args.src_region}"
    out = f"{args.root_path}/{args.region}/benchmark"
    os.makedirs(f"{out}/noise", exist_ok=True)
    os.makedirs(f"{out}/figures", exist_ok=True)

    t0 = obspy.UTCDateTime(args.start)
    t1 = t0 + args.duration

    events = pd.read_csv(f"{src}/adloc/adloc_events.csv", parse_dates=["time"])
    events["time"] = pd.to_datetime(events["time"], utc=True)
    inside = events[
        (events["time"] >= pd.Timestamp(t0.datetime, tz="UTC") - pd.Timedelta(seconds=60))
        & (events["time"] <= pd.Timestamp(t1.datetime, tz="UTC") + pd.Timedelta(seconds=60))
    ]
    print(f"Noise window {t0} -- {t1} ({args.duration:.0f} s)")
    if len(inside) > 0:
        print(f"WARNING: {len(inside)} catalogued event(s) fall inside the window:")
        print(inside[["time", "magnitude", "num_picks"]].to_string())
    else:
        print("No catalogued event inside the window (+/- 60 s margin)")

    files = sorted(glob(f"{src}/waveforms/????/???/*.mseed"))
    print(f"Found {len(files)} channel files in {src}/waveforms")

    rows = []
    for f in files:
        name = os.path.basename(f).replace(".mseed", "")
        stream = obspy.read(f, format="MSEED")
        stream.merge(fill_value=0)
        trace = stream[0]
        sr = trace.stats.sampling_rate
        npts_expected = int(round(args.duration * sr))

        trace.trim(t0, t1, pad=True, fill_value=0, nearest_sample=True)
        data = np.asarray(trace.data, dtype=np.float64)
        if np.ma.isMaskedArray(data):
            data = data.filled(0.0)
        data = data[:npts_expected]
        if len(data) < npts_expected:  # should not happen for a full-day file
            data = np.pad(data, (0, npts_expected - len(data)))

        frac_zero = float(np.mean(data == 0))
        data = data - data.mean()
        trace.data = data.astype(np.float32)
        trace.stats.npts = len(data)
        trace.write(f"{out}/noise/{name}.mseed", format="MSEED")

        filt = bandpass(data, sr)
        rows.append(
            {
                "file_name": f"{out}/noise/{name}.mseed",
                "station_id": f"{name[:-1]}",
                "network": name.split(".")[0],
                "station": name.split(".")[1],
                "location": name.split(".")[2],
                "channel": name.split(".")[3],
                "component": name[-1],
                "instrument": name.split(".")[3][:2],
                "sampling_rate": sr,
                "npts": len(data),
                "starttime": trace.stats.starttime.isoformat(),
                "rms": float(np.std(data)),
                "rms_band": float(np.std(filt)),
                "peak_band": float(np.max(np.abs(filt))),
                "crest_band": float(np.max(np.abs(filt)) / (np.std(filt) + 1e-30)),
                "frac_zero": frac_zero,
            }
        )
        print(f"  {name:20s} {sr:5.0f} Hz  rms={rows[-1]['rms']:.3g}  rms({FREQMIN:g}-{FREQMAX:g}Hz)"
              f"={rows[-1]['rms_band']:.3g}  crest={rows[-1]['crest_band']:.1f}  zeros={frac_zero:.1%}")

    noise = pd.DataFrame(rows).sort_values(["station_id", "component"])
    noise.to_csv(f"{out}/noise_stats.csv", index=False)
    print(f"\nWrote {len(noise)} channels -> {out}/noise/")
    print(f"Wrote {out}/noise_stats.csv")

    bad = noise[(noise["frac_zero"] > 0.01) | (noise["crest_band"] > 15)]
    if len(bad) > 0:
        print("\nChannels with gaps or transients (kept, but worth a look):")
        print(bad[["file_name", "frac_zero", "crest_band"]].to_string(index=False))

    return noise, events, t0, t1


# %%
def plot_day_overview(args, events, t0, t1):
    """Where the noise window sits in the day, and what the reference station looks like."""
    src = f"{args.root_path}/{args.src_region}"
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"

    ref = glob(f"{src}/waveforms/????/???/{args.ref_station}.mseed")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[1, 1.4])

    ax = axes[0]
    day0 = pd.Timestamp(t0.datetime, tz="UTC").normalize()
    ax.scatter((events["time"] - day0).dt.total_seconds() / 3600, events["magnitude"], s=6, c="tab:red", alpha=0.6)
    ax.set_ylabel("magnitude")
    ax.set_title(f"AdLoc catalogue for {day0.date()} ({len(events)} events) and the chosen noise window")
    ax.grid(alpha=0.3)

    if ref:
        tr = obspy.read(ref[0], format="MSEED")[0]
        tr.merge = None
        sr = tr.stats.sampling_rate
        d = np.asarray(tr.data, dtype=np.float64)
        d = bandpass(d - d.mean(), sr)
        # envelope on a 10 s grid so a full day is plottable
        step = int(10 * sr)
        n = len(d) // step * step
        env = np.abs(d[:n]).reshape(-1, step).max(axis=1)
        th = (tr.stats.starttime - obspy.UTCDateTime(day0.to_pydatetime())) / 3600 + np.arange(len(env)) * 10 / 3600
        axes[1].semilogy(th, env + 1e-12, lw=0.4, color="k")
        axes[1].set_ylabel(f"|{args.ref_station}| ({FREQMIN:g}-{FREQMAX:g} Hz)\n10 s max envelope")
    axes[1].set_xlabel(f"hours since {day0}")
    axes[1].grid(alpha=0.3)

    h0 = (pd.Timestamp(t0.datetime, tz="UTC") - day0).total_seconds() / 3600
    h1 = (pd.Timestamp(t1.datetime, tz="UTC") - day0).total_seconds() / 3600
    for ax in axes:
        ax.axvspan(h0, h1, color="tab:green", alpha=0.25, zorder=0)
        ax.axvline(h0, color="tab:green", lw=1)
        ax.axvline(h1, color="tab:green", lw=1)
    axes[0].annotate("noise window", (h0, ax.get_ylim()[1]), color="tab:green", fontsize=9)
    axes[0].set_xlim(0, 24)

    fig.tight_layout()
    fig.savefig(f"{figdir}/noise_01_day_overview.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/noise_01_day_overview.png")


def plot_noise_gallery(args, noise):
    """All vertical channels over a short slice, so gaps/spikes/dead channels are obvious."""
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    z = noise[noise["component"].isin(["Z", "3"])].sort_values("station_id").reset_index(drop=True)
    snippet = 120.0

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.28 * len(z))), width_ratios=[2.2, 1])
    ax = axes[0]
    for k, row in z.iterrows():
        tr = obspy.read(row["file_name"], format="MSEED")[0]
        sr = tr.stats.sampling_rate
        d = np.asarray(tr.data, dtype=np.float64)[: int(snippet * sr)]
        d = bandpass(d, sr)
        amp = np.max(np.abs(d))
        if amp == 0:
            ax.text(0, k, f"{row['station_id']}  DEAD", fontsize=6, color="tab:red", va="center")
            continue
        ax.plot(np.arange(len(d)) / sr, 0.45 * d / amp + k, lw=0.3, color="k")
        ax.text(0.2, k + 0.15, f"{row['station_id']}{row['component']}  rms={row['rms_band']:.2g}", fontsize=6)
    ax.set_xlim(0, snippet)
    ax.set_ylim(-1, len(z))
    ax.set_yticks([])
    ax.set_xlabel("seconds into the noise window")
    ax.set_title(f"noise, vertical channels, {FREQMIN:g}-{FREQMAX:g} Hz, trace-normalised (first {snippet:g} s)")

    ax = axes[1]
    y = np.arange(len(z))
    ax.barh(y, z["rms_band"], color="tab:blue", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.station_id}{r.component}" for r in z.itertuples()], fontsize=6)
    ax.set_xscale("log")
    ax.set_ylim(-1, len(z))
    ax.invert_yaxis()
    ax.set_xlabel(f"noise rms in {FREQMIN:g}-{FREQMAX:g} Hz (counts)")
    ax.set_title("noise level per channel")
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(f"{figdir}/noise_02_gallery.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/noise_02_gallery.png")


def plot_noise_spectra(args, noise):
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    z = noise[noise["component"].isin(["Z", "3"])].sort_values("station_id")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for _, row in z.iterrows():
        tr = obspy.read(row["file_name"], format="MSEED")[0]
        sr = tr.stats.sampling_rate
        d = np.asarray(tr.data, dtype=np.float64)
        f, p = welch(d, fs=sr, nperseg=int(60 * sr))
        color = {"BH": "tab:blue", "HH": "tab:orange", "HN": "tab:green", "EH": "tab:red"}.get(row["instrument"], "k")
        axes[0].loglog(f[1:], p[1:], lw=0.6, alpha=0.7, color=color)
    for inst, color in [("BH", "tab:blue"), ("HH", "tab:orange"), ("HN", "tab:green"), ("EH", "tab:red")]:
        axes[0].plot([], [], color=color, label=inst)
    axes[0].axvspan(FREQMIN, FREQMAX, color="0.85", zorder=0)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("PSD (counts^2/Hz)")
    axes[0].set_title(f"noise spectra, vertical channels (grey = {FREQMIN:g}-{FREQMAX:g} Hz template band)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].hist(np.log10(noise["rms_band"] + 1e-30), bins=25, color="tab:blue", edgecolor="k", lw=0.4)
    axes[1].set_xlabel(f"log10 rms in {FREQMIN:g}-{FREQMAX:g} Hz")
    axes[1].set_ylabel("channels")
    axes[1].set_title("spread of noise levels across all channels")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{figdir}/noise_03_spectra.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/noise_03_spectra.png")


# %%
if __name__ == "__main__":
    args = parse_args()
    noise, events, t0, t1 = extract(args)
    print("\nPlotting:")
    plot_day_overview(args, events, t0, t1)
    plot_noise_gallery(args, noise)
    plot_noise_spectra(args, noise)

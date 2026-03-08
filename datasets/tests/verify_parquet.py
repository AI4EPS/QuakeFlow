# %%
"""
Verify parquet files by plotting waveforms with phase picks.

Similar to plot_ceed.py - shows real data samples from GCS parquet files
with waveform, P/S picks, polarity, and metadata.
"""
import argparse
import os
import random

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
BUCKET = "quakeflow_dataset"
SAMPLING_RATE = 100.0
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def get_gcs_fs():
    return fsspec.filesystem("gs", token=GCS_CREDENTIALS_PATH)


def load_parquet_from_gcs(region, year, jday):
    """Load a parquet file from GCS."""
    path = f"{BUCKET}/{region}EDC/waveform_parquet/{year:04d}/{jday:03d}.parquet"
    fs = get_gcs_fs()
    with fs.open(path, "rb") as f:
        table = pq.read_table(f)
    return table.to_pandas()


def normalize_trace(trace):
    trace = trace - np.mean(trace)
    max_val = np.max(np.abs(trace))
    return trace / max_val if max_val > 0 else trace


def plot_single_record(record, ax_wave, ax_zoom_p, ax_zoom_s, ax_info):
    """Plot a single waveform record in 4 panels."""
    w = record["waveform"]
    waveform = np.stack([np.asarray(ch, dtype=np.float32) for ch in w])

    nt = waveform.shape[1]
    time = np.arange(nt) / SAMPLING_RATE
    components = ["E", "N", "Z"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    p_idx = record.get("p_phase_index")
    s_idx = record.get("s_phase_index")
    p_pol = record.get("p_phase_polarity", "")

    # Full waveform
    for i, (comp, color) in enumerate(zip(components, colors)):
        ax_wave.plot(time, normalize_trace(waveform[i]) + i, color=color, lw=0.5, label=comp)
    if p_idx is not None and not (isinstance(p_idx, float) and np.isnan(p_idx)):
        ax_wave.axvline(p_idx / SAMPLING_RATE, color="red", ls="--", lw=1.5, label=f"P ({p_pol})")
    if s_idx is not None and not (isinstance(s_idx, float) and np.isnan(s_idx)):
        ax_wave.axvline(s_idx / SAMPLING_RATE, color="blue", ls="--", lw=1.5, label="S")
    ax_wave.set_yticks([0, 1, 2])
    ax_wave.set_yticklabels(components)
    ax_wave.legend(loc="upper right", fontsize=7)
    ax_wave.set_xlim(0, time[-1])
    station_label = f"{record.get('network','')}.{record.get('station','')}"
    ax_wave.set_title(f"{record.get('event_id','')}, {station_label}, M{record.get('event_magnitude','?')}", fontsize=9)

    # Zoom P
    if p_idx is not None and not (isinstance(p_idx, float) and np.isnan(p_idx)):
        p_time = p_idx / SAMPLING_RATE
        t0 = max(0, p_time - 1)
        t1 = min(time[-1], p_time + 3)
        i0, i1 = int(t0 * SAMPLING_RATE), int(t1 * SAMPLING_RATE)
        for i, (comp, color) in enumerate(zip(components, colors)):
            ax_zoom_p.plot(time[i0:i1], normalize_trace(waveform[i, i0:i1]) + i, color=color, lw=0.8)
        ax_zoom_p.axvline(p_time, color="red", ls="--", lw=1)
        ax_zoom_p.set_xlim(t0, t1)
        ax_zoom_p.set_yticks([0, 1, 2])
        ax_zoom_p.set_yticklabels(components)
    else:
        ax_zoom_p.text(0.5, 0.5, "No P", ha="center", va="center", transform=ax_zoom_p.transAxes)
    ax_zoom_p.set_title(f"P zoom ({p_pol})", fontsize=9)

    # Zoom S
    if s_idx is not None and not (isinstance(s_idx, float) and np.isnan(s_idx)):
        s_time = s_idx / SAMPLING_RATE
        t0 = max(0, s_time - 2)
        t1 = min(time[-1], s_time + 5)
        i0, i1 = int(t0 * SAMPLING_RATE), int(t1 * SAMPLING_RATE)
        for i, (comp, color) in enumerate(zip(components, colors)):
            ax_zoom_s.plot(time[i0:i1], normalize_trace(waveform[i, i0:i1]) + i, color=color, lw=0.8)
        ax_zoom_s.axvline(s_time, color="blue", ls="--", lw=1)
        ax_zoom_s.set_xlim(t0, t1)
        ax_zoom_s.set_yticks([0, 1, 2])
        ax_zoom_s.set_yticklabels(components)
    else:
        ax_zoom_s.text(0.5, 0.5, "No S", ha="center", va="center", transform=ax_zoom_s.transAxes)
    ax_zoom_s.set_title("S zoom", fontsize=9)

    # Info panel
    ax_info.axis("off")
    info_lines = [
        f"event_id: {record.get('event_id', '')}",
        f"event_time: {record.get('event_time', '')}",
        f"lat/lon: {record.get('event_latitude', '')}, {record.get('event_longitude', '')}",
        f"depth: {record.get('event_depth_km', '')} km",
        f"magnitude: {record.get('event_magnitude', '')} ({record.get('event_magnitude_type', '')})",
        "",
        f"network.station: {record.get('network','')}.{record.get('station','')}",
        f"location: {record.get('location','')}",
        f"instrument: {record.get('instrument','')}",
        f"component: {record.get('component','')}",
        f"sta lat/lon: {record.get('station_latitude','')}, {record.get('station_longitude','')}",
        f"distance: {record.get('distance_km', ''):.1f} km" if isinstance(record.get('distance_km'), (int, float)) else f"distance: {record.get('distance_km', '')}",
        f"azimuth: {record.get('azimuth', '')}",
        "",
        f"p_index: {p_idx}",
        f"s_index: {s_idx}",
        f"p_polarity: {p_pol}",
        f"p_score: {record.get('p_phase_score', '')}",
        f"s_score: {record.get('s_phase_score', '')}",
        f"snr: {record.get('snr', ''):.1f}" if isinstance(record.get('snr'), (int, float)) else f"snr: {record.get('snr', '')}",
        "",
        f"strike/dip/rake: {record.get('strike','')}/{record.get('dip','')}/{record.get('rake','')}",
        f"waveform shape: {waveform.shape}",
    ]
    ax_info.text(0.05, 0.95, "\n".join(info_lines), transform=ax_info.transAxes,
                 fontsize=7, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


def plot_parquet_samples(region, year, jday, n_samples=6, seed=42, output_prefix=None):
    """Plot N random samples from a parquet file."""
    print(f"Loading {region}EDC {year:04d}/{jday:03d}...")
    df = load_parquet_from_gcs(region, year, jday)
    print(f"  {len(df)} records loaded")

    # Filter to records with valid P and S
    mask = df["p_phase_index"].notna() & df["s_phase_index"].notna()
    df_valid = df[mask]
    print(f"  {len(df_valid)} records with P+S picks")

    if len(df_valid) == 0:
        print("  No valid records, skipping")
        return

    random.seed(seed)
    indices = random.sample(range(len(df_valid)), min(n_samples, len(df_valid)))

    fig = plt.figure(figsize=(20, 5 * len(indices)))
    gs = fig.add_gridspec(len(indices), 4, width_ratios=[2, 1, 1, 1])

    for row, idx in enumerate(indices):
        record = df_valid.iloc[idx].to_dict()
        ax_wave = fig.add_subplot(gs[row, 0])
        ax_zoom_p = fig.add_subplot(gs[row, 1])
        ax_zoom_s = fig.add_subplot(gs[row, 2])
        ax_info = fig.add_subplot(gs[row, 3])
        plot_single_record(record, ax_wave, ax_zoom_p, ax_zoom_s, ax_info)

    plt.tight_layout()
    prefix = output_prefix or f"parquet_{region}_{year:04d}_{jday:03d}"
    out_path = os.path.join(FIGURES_DIR, f"{prefix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_summary_stats(region, year, jday, output_prefix=None):
    """Plot summary statistics for a parquet file."""
    print(f"Loading stats for {region}EDC {year:04d}/{jday:03d}...")
    df = load_parquet_from_gcs(region, year, jday)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{region}EDC {year:04d}/{jday:03d} — {len(df)} records", fontsize=14)

    # Magnitude distribution
    ax = axes[0, 0]
    mag = df["event_magnitude"].dropna()
    ax.hist(mag, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Count")
    ax.set_title(f"Magnitude Distribution (n={len(mag)})")

    # Distance distribution
    ax = axes[0, 1]
    dist = df["distance_km"].dropna()
    ax.hist(dist, bins=30, edgecolor="black", alpha=0.7, color="orange")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distance Distribution (n={len(dist)})")

    # SNR distribution
    ax = axes[0, 2]
    snr = df["snr"].dropna()
    snr_clipped = snr[snr < snr.quantile(0.99)]  # clip outliers
    ax.hist(snr_clipped, bins=30, edgecolor="black", alpha=0.7, color="green")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Count")
    ax.set_title(f"SNR Distribution (n={len(snr)})")

    # Polarity distribution
    ax = axes[1, 0]
    pol = df["p_phase_polarity"].fillna("none")
    pol_counts = pol.value_counts()
    ax.bar(pol_counts.index, pol_counts.values, edgecolor="black", alpha=0.7, color="purple")
    ax.set_xlabel("P Polarity")
    ax.set_ylabel("Count")
    ax.set_title("P Phase Polarity")

    # Event map
    ax = axes[1, 1]
    ev = df.drop_duplicates("event_id")
    ax.scatter(ev["event_longitude"], ev["event_latitude"], s=3, alpha=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Event Locations ({len(ev)} events)")
    ax.set_aspect("equal")

    # Station map
    ax = axes[1, 2]
    sta = df.drop_duplicates(["network", "station"])
    ax.scatter(sta["station_longitude"], sta["station_latitude"], s=5, alpha=0.7, c="red", marker="^")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Station Locations ({len(sta)} stations)")
    ax.set_aspect("equal")

    plt.tight_layout()
    prefix = output_prefix or f"parquet_stats_{region}_{year:04d}_{jday:03d}"
    out_path = os.path.join(FIGURES_DIR, f"{prefix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify parquet files by plotting")
    parser.add_argument("--region", type=str, default="SC", help="Region: SC or NC")
    parser.add_argument("--year", type=int, default=2024, help="Year")
    parser.add_argument("--jday", type=int, default=1, help="Julian day")
    parser.add_argument("--n_samples", type=int, default=6, help="Number of waveform samples to plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stats_only", action="store_true", help="Only plot summary stats")
    parser.add_argument("--waveforms_only", action="store_true", help="Only plot waveforms")
    args = parser.parse_args()

    if not args.waveforms_only:
        plot_summary_stats(args.region, args.year, args.jday)
    if not args.stats_only:
        plot_parquet_samples(args.region, args.year, args.jday, n_samples=args.n_samples, seed=args.seed)

# %%

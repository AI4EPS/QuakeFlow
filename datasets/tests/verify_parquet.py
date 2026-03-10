# %%
"""
Verify parquet files by plotting waveforms with phase picks.

Uses ceed.py's records_to_sample, generate_labels, and plot_overview
for event-based multi-station visualization.
"""
import argparse
import os
import sys
import random

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

# Add EQNet to path for ceed imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "EQNet", "eqnet", "data"))
from ceed import records_to_sample, generate_labels, plot_overview, plot_trace, LabelConfig

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


def plot_parquet_events(region, year, jday, n_events=3, n_traces=3, output_prefix=None, no_traces=False):
    """Plot top N events (most stations) using ceed.py's plot_overview and plot_trace."""
    print(f"Loading {region}EDC {year:04d}/{jday:03d}...")
    df = load_parquet_from_gcs(region, year, jday)
    print(f"  {len(df)} records, {df['event_id'].nunique()} events")

    config = LabelConfig()

    # Group by event, pick events with most stations
    event_counts = df.groupby("event_id").size().sort_values(ascending=False)
    event_ids = event_counts.head(n_events).index.tolist()

    for event_id in event_ids:
        event_df = df[df["event_id"] == event_id]
        records = []
        for _, row in event_df.iterrows():
            d = row.to_dict()
            # Convert waveform from object array of arrays to proper 2D list
            w = d.get("waveform")
            if w is not None and hasattr(w, 'tolist'):
                d["waveform"] = np.array(w.tolist(), dtype=np.float32).tolist()
            # Convert NaN values to None for phase indices (ceed.py checks `is not None`)
            for key in ("p_phase_index", "s_phase_index"):
                if key in d and isinstance(d[key], float) and np.isnan(d[key]):
                    d[key] = None
            records.append(d)

        # Convert to ceed Sample
        sample = records_to_sample(records)
        mag = records[0].get("event_magnitude", "?")
        title = f"{event_id} | M{mag} | {sample.nx} stations"

        # Overview plot (2x2: Z wiggle, Z+picks, phase labels, event time)
        prefix = output_prefix or f"parquet_{region}_{year:04d}_{jday:03d}"
        overview_path = os.path.join(FIGURES_DIR, f"{prefix}_{event_id}.png")
        plot_overview(sample, config, title=title, save_path=overview_path)
        print(f"  Saved overview: {overview_path} ({sample.nx} stations)")

        if no_traces:
            continue

        # Trace plots for stations with picks (prefer both P+S, fall back to P-only)
        labeled_stations = set()
        for tgt in sample.targets:
            p_stas = {int(s) for s, _ in tgt.p_picks}
            s_stas = {int(s) for s, _ in tgt.s_picks}
            labeled_stations |= (p_stas & s_stas)
        if not labeled_stations:
            for tgt in sample.targets:
                labeled_stations |= {int(s) for s, _ in tgt.p_picks}
        labeled_stations = sorted(labeled_stations)

        if labeled_stations:
            labels = generate_labels(sample, config)
            n = min(n_traces, len(labeled_stations))
            stations = [labeled_stations[i] for i in np.linspace(0, len(labeled_stations) - 1, n, dtype=int)]
            for j, sta in enumerate(stations):
                trace_id = sample.trace_ids[sta] if sta < len(sample.trace_ids) else f"station_{sta}"
                trace_path = os.path.join(FIGURES_DIR, f"{prefix}_{event_id}_trace{j:02d}.png")
                plot_trace(sample, labels, sta=sta, title=f"{title} | {trace_id}", save_path=trace_path)
            print(f"  Saved {n} trace plots")


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
    parser.add_argument("--n_events", type=int, default=3, help="Number of events to plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stats_only", action="store_true", help="Only plot summary stats")
    parser.add_argument("--waveforms_only", action="store_true", help="Only plot waveforms")
    parser.add_argument("--no_traces", action="store_true", help="Skip single-trace plots, only overview")
    args = parser.parse_args()

    if not args.waveforms_only:
        plot_summary_stats(args.region, args.year, args.jday)
    if not args.stats_only:
        plot_parquet_events(args.region, args.year, args.jday, n_events=args.n_events, no_traces=args.no_traces)

# %%

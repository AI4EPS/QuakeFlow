# %%
import argparse
import json
import os

import fsspec
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Constants
GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
DEFAULT_BUCKET = "quakeflow_dataset"


def open_h5_file(h5_path, mode="r"):
    """Open an HDF5 file from local path or Google Cloud Storage.

    Args:
        h5_path: Local path or GCS path (gs://bucket/path/to/file.h5)
        mode: File mode ('r' for read)

    Returns:
        h5py.File object
    """
    if h5_path.startswith("gs://"):
        with open(GCS_CREDENTIALS_PATH, "r") as f:
            token = json.load(f)
        fs = fsspec.filesystem("gs", token=token)
        return h5py.File(fs.open(h5_path, "rb"), mode)
    else:
        return h5py.File(h5_path, mode)


def normalize_trace(trace):
    """Normalize trace to [-0.5, 0.5] range for plotting."""
    trace = trace - np.mean(trace)
    max_val = np.max(np.abs(trace))
    if max_val > 0:
        return trace / max_val
    return trace


def plot_waveform(h5_file, event_id=None, station_id=None, output_dir="figures"):
    """Plot waveforms from HDF5 file for quality checking.

    Args:
        h5_file: Path to the waveform.h5 file
        event_id: Specific event ID to plot (if None, plots first event)
        station_id: Specific station ID to plot (if None, plots first station)
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    with open_h5_file(h5_file, "r") as fp:
        if event_id is None:
            event_id = list(fp.keys())[0]

        if event_id not in fp:
            print(f"Event {event_id} not found in {h5_file}")
            return

        event_group = fp[event_id]
        event_attrs = dict(event_group.attrs)

        stations = [key for key in event_group.keys() if isinstance(event_group[key], h5py.Dataset)]
        if len(stations) == 0:
            print(f"No stations found for event {event_id}")
            return

        if station_id is None:
            station_id = stations[0]

        if station_id not in event_group:
            print(f"Station {station_id} not found for event {event_id}")
            return

        ds = event_group[station_id]
        waveform = ds[:]
        station_attrs = dict(ds.attrs)
        station_attrs["data_shape"] = waveform.shape

        sampling_rate = 100.0
        nt = waveform.shape[1]
        time = np.arange(nt) / sampling_rate

        p_index = station_attrs.get("p_phase_index", None)
        s_index = station_attrs.get("s_phase_index", None)
        p_polarity = station_attrs.get("p_phase_polarity", "")
        s_polarity = station_attrs.get("s_phase_polarity", "")

        fig = plt.figure(figsize=(16, 10))

        # Create grid: upper panel spans full width, lower has two panels + info
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.8])

        # Upper panel: full waveform
        ax_full = fig.add_subplot(gs[0, :2])
        components = ["E", "N", "Z"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, (comp, color) in enumerate(zip(components, colors)):
            ax_full.plot(time, normalize_trace(waveform[i]) + i, color=color, label=comp, lw=0.5)

        if p_index is not None:
            ax_full.axvline(p_index / sampling_rate, color="red", linestyle="--", lw=1.5, label=f"P ({p_polarity})")
        if s_index is not None:
            ax_full.axvline(s_index / sampling_rate, color="blue", linestyle="--", lw=1.5, label=f"S ({s_polarity})")

        ax_full.set_xlabel("Time (s)")
        ax_full.set_ylabel("Trace")
        ax_full.set_yticks([0, 1, 2])
        ax_full.set_yticklabels(components)
        ax_full.set_title(f"Event: {event_id}, Station: {station_id}")
        ax_full.legend(loc="upper right")
        ax_full.set_xlim(0, time[-1])

        # Lower left panel: zoom around P phase (1s before, 3s after)
        ax_p = fig.add_subplot(gs[1, 0])
        if p_index is not None:
            p_time = p_index / sampling_rate
            t_start = max(0, p_time - 1)
            t_end = min(time[-1], p_time + 3)
            idx_start = int(t_start * sampling_rate)
            idx_end = int(t_end * sampling_rate)

            for i, (comp, color) in enumerate(zip(components, colors)):
                trace_segment = waveform[i, idx_start:idx_end]
                ax_p.plot(time[idx_start:idx_end], normalize_trace(trace_segment) + i, color=color, label=comp, lw=0.8)

            ax_p.axvline(p_time, color="red", linestyle="--", lw=0.5, label=f"P ({p_polarity})")
            ax_p.set_xlim(t_start, t_end)
            ax_p.set_yticks([0, 1, 2])
            ax_p.set_yticklabels(components)
            ax_p.legend(loc="upper right")
        else:
            ax_p.text(0.5, 0.5, "No P-phase", ha="center", va="center", transform=ax_p.transAxes)

        ax_p.set_xlabel("Time (s)")
        ax_p.set_ylabel("Trace")


        # Lower right panel: zoom around S phase (2s before, 5s after)
        ax_s = fig.add_subplot(gs[1, 1])
        if s_index is not None:
            s_time = s_index / sampling_rate
            t_start = max(0, s_time - 2)
            t_end = min(time[-1], s_time + 5)
            idx_start = int(t_start * sampling_rate)
            idx_end = int(t_end * sampling_rate)

            for i, (comp, color) in enumerate(zip(components, colors)):
                trace_segment = waveform[i, idx_start:idx_end]
                ax_s.plot(time[idx_start:idx_end], normalize_trace(trace_segment) + i, color=color, label=comp, lw=0.8)

            ax_s.axvline(s_time, color="blue", linestyle="--", lw=0.5, label=f"S ({s_polarity})")
            ax_s.set_xlim(t_start, t_end)
            ax_s.set_yticks([0, 1, 2])
            ax_s.set_yticklabels(components)
            ax_s.legend(loc="upper right")
        else:
            ax_s.text(0.5, 0.5, "No S-phase", ha="center", va="center", transform=ax_s.transAxes)

        ax_s.set_xlabel("Time (s)")
        ax_s.set_ylabel("Trace")

        # Right side panel: event and station attributes
        ax_info = fig.add_subplot(gs[:, 2])
        ax_info.axis("off")

        info_text = "EVENT ATTRIBUTES\n" + "=" * 30 + "\n"
        for key, value in sorted(event_attrs.items()):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            info_text += f"{key}: {value}\n"

        info_text += "\n" + "STATION ATTRIBUTES\n" + "=" * 30 + "\n"
        for key, value in sorted(station_attrs.items()):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            # Truncate long values for display
            value_str = str(value)
            if len(value_str) > 40:
                value_str = value_str[:37] + "..."
            info_text += f"{key}: {value_str}\n"

        ax_info.text(
            0.05, 0.95, info_text, transform=ax_info.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        output_file = f"{output_dir}/{event_id}_{station_id}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_file}")
        plt.close()


def list_events(h5_file, max_events=10):
    """List events and their stations in the HDF5 file."""
    with open_h5_file(h5_file, "r") as fp:
        events = list(fp.keys())
        print(f"Total events: {len(events)}")
        print("-" * 50)

        for i, event_id in enumerate(events[:max_events]):
            event_group = fp[event_id]
            stations = [key for key in event_group.keys() if isinstance(event_group[key], h5py.Dataset)]
            attrs = dict(event_group.attrs)

            mag = attrs.get("magnitude", "N/A")
            lat = attrs.get("latitude", "N/A")
            lon = attrs.get("longitude", "N/A")
            has_fm = "strike" in attrs

            print(f"{event_id}: M{mag}, ({lat}, {lon}), {len(stations)} stations, FM: {has_fm}")

        if len(events) > max_events:
            print(f"... and {len(events) - max_events} more events")


def build_gcs_path(region, year, jday, bucket=DEFAULT_BUCKET):
    """Build GCS path for waveform.h5 file.

    Args:
        region: 'SC' or 'NC'
        year: Year (e.g., 2025)
        jday: Julian day (e.g., 1)
        bucket: GCS bucket name

    Returns:
        GCS path string (gs://bucket/path/waveform.h5)
    """
    return f"gs://{bucket}/{region}EDC/dataset/{year:04d}/{jday:03d}/waveform.h5"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize waveforms from HDF5 file")
    parser.add_argument("--h5_file", type=str, default=None, help="Path to the waveform.h5 file (local or gs://)")
    parser.add_argument("--region", type=str, default=None, help="Region code (SC or NC) for cloud path")
    parser.add_argument("--year", type=int, default=None, help="Year for cloud path")
    parser.add_argument("--jday", type=int, default=None, help="Julian day for cloud path")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET, help="GCS bucket name")
    parser.add_argument("--event_id", type=str, default=None, help="Specific event ID to plot")
    parser.add_argument("--station_id", type=str, default=None, help="Specific station ID to plot")
    parser.add_argument("--output_dir", type=str, default="figures", help="Output directory for figures")
    parser.add_argument("--list", action="store_true", help="List events in the file")
    parser.add_argument("--plot_all", action="store_true", help="Plot all events (first station each)")
    parser.add_argument("--max_events", type=int, default=10, help="Max events to list or plot")

    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = parse_args()

    # Determine h5_file path
    if args.h5_file:
        h5_file = args.h5_file
    elif args.region and args.year and args.jday:
        h5_file = build_gcs_path(args.region, args.year, args.jday, args.bucket)
        print(f"Using cloud path: {h5_file}")
    else:
        print("Error: Provide either --h5_file or (--region, --year, --jday)")
        exit(1)

    if args.list:
        list_events(h5_file, args.max_events)
    elif args.plot_all:
        with open_h5_file(h5_file, "r") as fp:
            events = list(fp.keys())[:args.max_events]

        for event_id in events:
            plot_waveform(h5_file, event_id=event_id, output_dir=args.output_dir)
    else:
        plot_waveform(h5_file, event_id=args.event_id, station_id=args.station_id, output_dir=args.output_dir)

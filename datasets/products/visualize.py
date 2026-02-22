# %%
"""
Visualize single event-station waveforms with polarity annotations.

Creates a 4-panel figure per event-station pair showing:
- Full waveform with P/S arrivals
- Zoom around P with predicted polarity amplitudes
- Zoom around S with predicted polarity amplitudes
- Text panel with polarity comparison and metadata

Usage:
    python visualize.py --h5_file ../NCEDC/dataset/2024/001.h5 \
        --polarity_file ../NCEDC/dataset/2024/001.polarities.csv \
        --plot_all --max_plots 10
"""
import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import SAMPLING_RATE, normalize_trace


def get_polarity_label(polarity_value):
    """Return polarity label (U/D) and color. Positive=Up(green), Negative=Down(red)."""
    if polarity_value > 0:
        return "U", "green"
    else:
        return "D", "red"


def select_stations(df, max_plots, s_only=True):
    """Select stations for plotting, preferring those with S picks.

    Args:
        df: DataFrame of polarity records (must have s_phase_index column)
        max_plots: Maximum total plots to return
        s_only: If True (default), only plot stations with S picks.
            If False, include P-only stations after S-pick stations.

    Returns:
        Filtered DataFrame
    """
    has_s = df["s_phase_index"].notna() if "s_phase_index" in df.columns else pd.Series(False, index=df.index)
    n_s = has_s.sum()
    print(f"  {n_s} stations with S picks, {len(df) - n_s} P-only")
    if s_only:
        df = df[has_s]
    else:
        df = df.copy()
        df["_has_s"] = has_s
        df = df.sort_values("_has_s", ascending=False).drop(columns=["_has_s"])
    return df.head(max_plots)


def plot_polarity_waveform(
    h5_file,
    polarity_row,
    output_dir="figures",
):
    """Plot waveforms with polarity labels for verification.

    Layout:
    - Upper panel: full waveform with all 3 components
    - Lower left: zoom around P phase
    - Lower middle: zoom around S phase
    - Right panel: attributes and polarity info
    """
    os.makedirs(output_dir, exist_ok=True)

    event_id = polarity_row["event_id"]
    network = polarity_row["network"]
    station = polarity_row["station"]
    location = polarity_row.get("location", "")
    if pd.isna(location):
        location = ""

    # Predicted polarity values from radiation pattern
    p_polarities = [polarity_row["p_polarity_e"], polarity_row["p_polarity_n"], polarity_row["p_polarity_z"]]
    s_polarities = [polarity_row["s_polarity_e"], polarity_row["s_polarity_n"], polarity_row["s_polarity_z"]]
    log_sp_ratio = polarity_row.get("log_sp_ratio", np.nan)

    with h5py.File(h5_file, "r") as fp:
        if event_id not in fp:
            print(f"Event {event_id} not found in {h5_file}")
            return False

        event_group = fp[event_id]
        event_attrs = dict(event_group.attrs)

        # Find station dataset (match network.station prefix)
        found_station = None
        for key in event_group.keys():
            if isinstance(event_group[key], h5py.Dataset):
                if key.startswith(f"{network}.{station}"):
                    found_station = key
                    break

        if found_station is None:
            print(f"Station {network}.{station} not found for event {event_id}")
            return False

        ds = event_group[found_station]
        waveform = ds[:]
        station_attrs = dict(ds.attrs)

        nt = waveform.shape[1]
        time = np.arange(nt) / SAMPLING_RATE

        p_index = station_attrs.get("p_phase_index", None)
        s_index = station_attrs.get("s_phase_index", None)

        # Catalog polarity from HDF5
        p_polarity = station_attrs.get("p_phase_polarity", "")
        s_polarity = station_attrs.get("s_phase_polarity", "")
        log_sp_ratio = station_attrs.get("log_sp_ratio", log_sp_ratio)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.8])

        # Upper panel: full waveform
        ax_full = fig.add_subplot(gs[0, :2])
        components = ["E", "N", "Z"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        n_comp = len(components)
        for i, (comp, color) in enumerate(zip(components, colors)):
            offset = n_comp - 1 - i
            ax_full.plot(time, normalize_trace(waveform[i]) + offset, color="k", label=comp, lw=0.5)

        if p_index is not None:
            ax_full.axvline(p_index / SAMPLING_RATE, color="red", linestyle="--", lw=1.5, label=f"P ({p_polarity})")
            ax_full.text(p_index / SAMPLING_RATE, 2.5, f"S/P={10**log_sp_ratio:.2f}", color="purple", fontsize=10, ha='right')
        if s_index is not None:
            ax_full.axvline(s_index / SAMPLING_RATE, color="blue", linestyle="--", lw=1.5, label=f"S ({s_polarity})")
            if p_index is None:
                ax_full.text(s_index / SAMPLING_RATE, 2.5, f"S/P={10**log_sp_ratio:.2f}", color="purple", fontsize=10, ha='right')

        ax_full.set_xlabel("Time (s)")
        ax_full.set_ylabel("Trace")
        ax_full.set_yticks([0, 1, 2])
        ax_full.set_yticklabels(components[::-1])
        ax_full.set_title(f"Event: {event_id}, Station: {found_station}")
        ax_full.legend(loc="upper right")
        ax_full.set_xlim(0, time[-1])

        # Lower left panel: zoom around P phase (1s before, 3s after)
        ax_p = fig.add_subplot(gs[1, 0])
        if p_index is not None:
            p_time = p_index / SAMPLING_RATE
            t_start = max(0, p_time - 1)
            t_end = min(time[-1], p_time + 3)
            idx_start = int(t_start * SAMPLING_RATE)
            idx_end = int(t_end * SAMPLING_RATE)

            for i, (comp, color) in enumerate(zip(components, colors)):
                offset = n_comp - 1 - i
                trace_segment = waveform[i, idx_start:idx_end]
                p_label, p_color = get_polarity_label(p_polarities[i])
                ax_p.plot(time[idx_start:idx_end], normalize_trace(trace_segment) + offset,
                         color="k", label=f"{comp}:{p_label}", lw=0.8)
                ax_p.text(t_start + 0.05, offset + 0.3, p_polarities[i], fontsize=12, fontweight='bold', color=p_color)

            ax_p.axvline(p_time, color="red", linestyle="--", lw=0.5, label=f"P ({p_polarity})")
            ax_p.set_xlim(t_start, t_end)
            ax_p.set_yticks([0, 1, 2])
            ax_p.set_yticklabels(components[::-1])
            ax_p.legend(loc="upper right")
        else:
            ax_p.text(0.5, 0.5, "No P-phase", ha="center", va="center", transform=ax_p.transAxes)

        ax_p.set_xlabel("Time (s)")
        ax_p.set_ylabel("Trace")
        ax_p.set_title("P-wave Polarity (E/N/Z)")

        # Lower middle panel: zoom around S phase (2s before, 5s after)
        ax_s = fig.add_subplot(gs[1, 1])
        if s_index is not None:
            s_time = s_index / SAMPLING_RATE
            t_start = max(0, s_time - 2)
            t_end = min(time[-1], s_time + 5)
            idx_start = int(t_start * SAMPLING_RATE)
            idx_end = int(t_end * SAMPLING_RATE)

            for i, (comp, color) in enumerate(zip(components, colors)):
                offset = n_comp - 1 - i
                trace_segment = waveform[i, idx_start:idx_end]
                s_label, s_color = get_polarity_label(s_polarities[i])
                ax_s.plot(time[idx_start:idx_end], normalize_trace(trace_segment) + offset,
                         color="k", label=f"{comp}:{s_label}", lw=0.8)
                ax_s.text(t_start + 0.05, offset + 0.3, s_polarities[i], fontsize=12, fontweight='bold', color=s_color)

            ax_s.axvline(s_time, color="blue", linestyle="--", lw=0.5, label=f"S ({s_polarity})")
            ax_s.set_xlim(t_start, t_end)
            ax_s.set_yticks([0, 1, 2])
            ax_s.set_yticklabels(components[::-1])
            ax_s.legend(loc="upper right")
        else:
            ax_s.text(0.5, 0.5, "No S-phase", ha="center", va="center", transform=ax_s.transAxes)

        ax_s.set_xlabel("Time (s)")
        ax_s.set_ylabel("Trace")
        ax_s.set_title("S-wave Polarity (E/N/Z)")

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
            value_str = str(value)
            if len(value_str) > 40:
                value_str = value_str[:37] + "..."
            info_text += f"{key}: {value_str}\n"

        ax_info.text(
            0.05, 1.0, info_text, transform=ax_info.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        output_file = f"{output_dir}/{event_id}_{network}.{station}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_file}")
        plt.close()
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize waveforms with polarity labels")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to the HDF5 dataset file")
    parser.add_argument("--polarity_file", type=str, required=True, help="Path to polarities CSV")
    parser.add_argument("--event_id", type=str, default=None, help="Specific event ID to plot")
    parser.add_argument("--station", type=str, default=None, help="Specific station to plot (format: NET.STA)")
    parser.add_argument("--output_dir", type=str, default="figures", help="Output directory for figures")
    parser.add_argument("--list", action="store_true", help="List entries in the polarity file")
    parser.add_argument("--plot_all", action="store_true", help="Plot all event-station pairs")
    parser.add_argument("--max_plots", type=int, default=10, help="Max plots to generate")
    parser.add_argument("--random", action="store_true", help="Randomly sample entries to plot")

    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = parse_args()

    if args.list:
        df = pd.read_csv(args.polarity_file)
        print(f"Total entries: {len(df)}")
        print(f"Unique events: {df['event_id'].nunique()}")
        print("-" * 60)
        print(df.head(args.max_plots).to_string(index=False))
        if len(df) > args.max_plots:
            print(f"... and {len(df) - args.max_plots} more entries")
    else:
        df = pd.read_csv(args.polarity_file)

        # Filter by event_id if specified
        if args.event_id is not None:
            df = df[df["event_id"] == args.event_id]

        # Only plot entries with catalog polarity labels
        df = df[df["p_phase_polarity"].isin(["U", "D"])]

        # Filter by station if specified
        if args.station is not None:
            net, sta = args.station.split(".")
            df = df[(df["network"] == net) & (df["station"] == sta)]

        if len(df) == 0:
            print("No matching entries found.")
        elif args.plot_all:
            if args.random:
                df = df.sample(n=min(args.max_plots, len(df)), random_state=42)
            else:
                # By default, only plot stations with S picks
                df = select_stations(df, args.max_plots)

            print(f"Plotting {len(df)} entries...")
            for _, row in df.iterrows():
                plot_polarity_waveform(args.h5_file, row, output_dir=args.output_dir)
        else:
            # Plot first match
            row = df.iloc[0]
            plot_polarity_waveform(args.h5_file, row, output_dir=args.output_dir)

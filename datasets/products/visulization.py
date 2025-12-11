# %%
import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_trace(trace):
    """Normalize trace to [-0.5, 0.5] range for plotting."""
    trace = trace - np.mean(trace)
    max_val = np.max(np.abs(trace))
    if max_val > 0:
        return trace / max_val
    return trace


def get_polarity_label(polarity_value):
    """Return polarity label (U/D) and color. Positive=Up(green), Negative=Down(red)."""
    if polarity_value > 0:
        return "U", "green"
    else:
        return "D", "red"


def plot_polarity_waveform(
    h5_file,
    polarity_row,
    output_dir="figures",
    sampling_rate=100.0,
):
    """Plot waveforms with polarity labels for verification.

    Same layout as datasets/visualization.py:
    - Upper panel: full waveform with all 3 components
    - Lower left: zoom around P phase
    - Lower middle: zoom around S phase
    - Right panel: attributes and polarity info

    Args:
        h5_file: Path to the waveform.h5 file
        polarity_row: A row from polarities.csv (Series or dict)
        output_dir: Directory to save figures
        sampling_rate: Sampling rate of waveforms
    """
    os.makedirs(output_dir, exist_ok=True)

    event_id = polarity_row["event_id"]
    network = polarity_row["network"]
    station = polarity_row["station"]
    location = polarity_row.get("location", "")
    if pd.isna(location):
        location = ""
    station_id = f"{network}.{station}.{location}"

    # Polarity values
    p_polarity_e = polarity_row["p_polarity_e"]
    p_polarity_n = polarity_row["p_polarity_n"]
    p_polarity_z = polarity_row["p_polarity_z"]
    s_polarity_e = polarity_row["s_polarity_e"]
    s_polarity_n = polarity_row["s_polarity_n"]
    s_polarity_z = polarity_row["s_polarity_z"]
    log_sp_ratio = polarity_row.get("log_sp_ratio", np.nan)

    p_polarities = [p_polarity_e, p_polarity_n, p_polarity_z]
    s_polarities = [s_polarity_e, s_polarity_n, s_polarity_z]

    with h5py.File(h5_file, "r") as fp:
        if event_id not in fp:
            print(f"Event {event_id} not found in {h5_file}")
            return False

        event_group = fp[event_id]
        event_attrs = dict(event_group.attrs)

        # Try to find station with different location codes
        found_station = None
        for key in event_group.keys():
            if isinstance(event_group[key], h5py.Dataset):
                if key.startswith(f"{network}.{station}"):
                    found_station = key
                    break

        if found_station is None:
            print(f"Station {station_id} not found for event {event_id}")
            return False

        ds = event_group[found_station]
        waveform = ds[:]
        station_attrs = dict(ds.attrs)

        nt = waveform.shape[1]
        time = np.arange(nt) / sampling_rate

        p_index = station_attrs.get("p_phase_index", None)
        s_index = station_attrs.get("s_phase_index", None)

        fig = plt.figure(figsize=(16, 10))

        # Create grid: upper panel spans full width, lower has two panels + info
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.8])

        # Upper panel: full waveform
        ax_full = fig.add_subplot(gs[0, :2])
        components = ["E", "N", "Z"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, (comp, color) in enumerate(zip(components, colors)):
            ax_full.plot(time, normalize_trace(waveform[i]) + i, color="k", label=comp, lw=0.5)

        # Get polarity labels from HDF5 file
        p_polarity = station_attrs.get("p_phase_polarity", "")
        s_polarity = station_attrs.get("s_phase_polarity", "")
        log_sp_ratio = station_attrs.get("log_sp_ratio", log_sp_ratio)

        if p_index is not None:
            ax_full.axvline(p_index / sampling_rate, color="red", linestyle="--", lw=1.5, label=f"P ({p_polarity})")
            ax_full.text(p_index / sampling_rate, 2.5, f"S/P={10**log_sp_ratio:.2f}", color="purple", fontsize=10, ha='right')
        if s_index is not None:
            ax_full.axvline(s_index / sampling_rate, color="blue", linestyle="--", lw=1.5, label=f"S ({s_polarity})")
            if p_index is None:
                ax_full.text(s_index / sampling_rate, 2.5, f"S/P={10**log_sp_ratio:.2f}", color="purple", fontsize=10, ha='right')


        ax_full.set_xlabel("Time (s)")
        ax_full.set_ylabel("Trace")
        ax_full.set_yticks([0, 1, 2])
        ax_full.set_yticklabels(components)
        ax_full.set_title(f"Event: {event_id}, Station: {found_station}")
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
                p_label, p_color = get_polarity_label(p_polarities[i])
                ax_p.plot(time[idx_start:idx_end], normalize_trace(trace_segment) + i,
                         color="k", label=f"{comp}:{p_label}", lw=0.8)
                # Add polarity text annotation
                ax_p.text(t_start + 0.05, i + 0.3, p_polarities[i], fontsize=12, fontweight='bold', color=p_color)

            ax_p.axvline(p_time, color="red", linestyle="--", lw=0.5, label=f"P ({p_polarity})")
            ax_p.set_xlim(t_start, t_end)
            ax_p.set_yticks([0, 1, 2])
            ax_p.set_yticklabels(components)
            ax_p.legend(loc="upper right")
        else:
            ax_p.text(0.5, 0.5, "No P-phase", ha="center", va="center", transform=ax_p.transAxes)

        ax_p.set_xlabel("Time (s)")
        ax_p.set_ylabel("Trace")
        ax_p.set_title("P-wave Polarity (E/N/Z)")

        # Lower middle panel: zoom around S phase (2s before, 5s after)
        ax_s = fig.add_subplot(gs[1, 1])
        if s_index is not None:
            s_time = s_index / sampling_rate
            t_start = max(0, s_time - 2)
            t_end = min(time[-1], s_time + 5)
            idx_start = int(t_start * sampling_rate)
            idx_end = int(t_end * sampling_rate)

            for i, (comp, color) in enumerate(zip(components, colors)):
                trace_segment = waveform[i, idx_start:idx_end]
                s_label, s_color = get_polarity_label(s_polarities[i])
                ax_s.plot(time[idx_start:idx_end], normalize_trace(trace_segment) + i,
                         color="k", label=f"{comp}:{s_label}", lw=0.8)
                # Add polarity text annotation
                ax_s.text(t_start + 0.05, i + 0.3, s_polarities[i], fontsize=12, fontweight='bold', color=s_color)

            ax_s.axvline(s_time, color="blue", linestyle="--", lw=0.5, label=f"S ({s_polarity})")
            ax_s.set_xlim(t_start, t_end)
            ax_s.set_yticks([0, 1, 2])
            ax_s.set_yticklabels(components)
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


def list_polarities(polarity_file, max_entries=10):
    """List events and stations in the polarities file."""
    df = pd.read_csv(polarity_file)
    print(f"Total entries: {len(df)}")
    print(f"Unique events: {df['event_id'].nunique()}")
    print("-" * 60)
    print(df.head(max_entries).to_string(index=False))
    if len(df) > max_entries:
        print(f"... and {len(df) - max_entries} more entries")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize waveforms with polarity labels")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to the waveform.h5 file")
    parser.add_argument("--polarity_file", type=str, required=True, help="Path to polarities.csv")
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
        list_polarities(args.polarity_file, args.max_plots)
    else:
        df = pd.read_csv(args.polarity_file)

        # Filter by event_id if specified
        if args.event_id is not None:
            df = df[df["event_id"] == args.event_id]

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
                df = df.head(args.max_plots)

            for _, row in df.iterrows():
                plot_polarity_waveform(args.h5_file, row, output_dir=args.output_dir)
        else:
            # Plot first match
            row = df.iloc[0]
            plot_polarity_waveform(args.h5_file, row, output_dir=args.output_dir)

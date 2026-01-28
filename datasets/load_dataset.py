# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "numpy",
#     "pandas",
#     "gcsfs",
#     "matplotlib",
# ]
# ///
"""
Load QuakeFlow earthquake dataset from GCS using HuggingFace datasets.

Dataset structure (per record = one event-station pair):
- Event info: event_id, event_time, event_latitude, event_longitude, event_depth_km, event_magnitude
- Station info: network, station, location, instrument, station_latitude, station_longitude
- Waveform: 3 channels (ENZ) × 12288 samples at 100Hz (~122.88s window)
- Phase info: p_phase_time, p_phase_index, s_phase_time, s_phase_index, etc.
"""

import json
import os

import numpy as np
from datasets import load_dataset

# GCS configuration
BUCKET = "gs://quakeflow_dataset"
GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")


def get_gcs_storage_options():
    """Load GCS credentials for authenticated access."""
    with open(GCS_CREDENTIALS_PATH, "r") as f:
        token = json.load(f)
    return {"token": token}


def load_quakeflow_dataset(region="SC", years=None, days=None, streaming=False):
    """
    Load QuakeFlow dataset from GCS.

    Args:
        region: "SC" (Southern California) or "NC" (Northern California)
        years: List of years to load, e.g., [2025, 2026]. None = all available.
        days: List of days to load, e.g., [1, 2, 3]. None = all days.
        streaming: If True, stream data without downloading everything first.

    Returns:
        HuggingFace Dataset object
    """
    storage_options = get_gcs_storage_options()

    # Build file patterns
    if years is None:
        pattern = f"{BUCKET}/{region}EDC/dataset/**/*.parquet"
    elif days is None:
        patterns = [f"{BUCKET}/{region}EDC/dataset/{year}/*.parquet" for year in years]
        pattern = patterns if len(patterns) > 1 else patterns[0]
    else:
        patterns = [
            f"{BUCKET}/{region}EDC/dataset/{year}/{day:03d}.parquet"
            for year in years for day in days
        ]
        pattern = patterns

    dataset = load_dataset(
        "parquet",
        data_files=pattern,
        streaming=streaming,
        storage_options=storage_options,
    )

    return dataset["train"]


# %% Iterate over phase picks
def iterate_phases():
    """Each record is one event-station pair with P and/or S phase."""
    dataset = load_quakeflow_dataset(region="SC", years=[2025], days=[9])

    for record in dataset:
        event_id = record["event_id"]
        station = f"{record['network']}.{record['station']}"
        waveform = np.array(record["waveform"])  # Shape: (3, 12288)

        print(f"Event: {event_id}, Station: {station}, Waveform shape: {waveform.shape}")

        # P phase info
        if record["p_phase_index"] is not None:
            print(f"  P arrival at sample {record['p_phase_index']}, time: {record['p_phase_time']}")

        # S phase info
        if record["s_phase_index"] is not None:
            print(f"  S arrival at sample {record['s_phase_index']}, time: {record['s_phase_time']}")

        break  # Remove to iterate all


# %% Group records by event
def iterate_events():
    """Group records by event_id to get all stations for each event."""
    dataset = load_quakeflow_dataset(region="SC", years=[2025], days=[9])
    df = dataset.to_pandas()

    for event_id, group in df.groupby("event_id"):
        row = group.iloc[0]
        print(f"Event {event_id}: M{row['event_magnitude']} at {row['event_time']}, {len(group)} stations")

        # Stack waveforms - each is an object array of shape (3,) containing arrays of len 12288
        waveforms = np.stack([np.stack(w) for w in group["waveform"]])  # (n_stations, 3, 12288)
        print(f"  Combined waveforms shape: {waveforms.shape}")

        # Get P and S indices
        p_indices = group["p_phase_index"].dropna().astype(int).tolist()
        s_indices = group["s_phase_index"].dropna().astype(int).tolist()
        print(f"  P picks: {len(p_indices)}, S picks: {len(s_indices)}")

        break  # Remove to iterate all


# %% Stream records
def iterate_streaming():
    """Stream data without loading everything into memory."""
    dataset = load_quakeflow_dataset(region="SC", years=[2025], days=[9], streaming=True)

    for i, record in enumerate(dataset):
        waveform = np.array(record["waveform"])
        print(f"Record {i}: {record['event_id']} - {record['network']}.{record['station']}")

        if i >= 5:
            break


# %% Stream by event
def iterate_events_streaming():
    """
    Stream events one at a time using itertools.groupby.

    Since dataset is sorted by event_id, records for each event are contiguous.
    This allows streaming without loading all data into memory.
    """
    from itertools import groupby

    dataset = load_quakeflow_dataset(region="SC", years=[2025], days=[9], streaming=True)

    for i, (event_id, records_iter) in enumerate(groupby(dataset, key=lambda x: x["event_id"])):
        records = list(records_iter)  # Materialize only this event's records

        event_time = records[0]["event_time"]
        magnitude = records[0]["event_magnitude"]
        n_stations = len(records)

        print(f"Event {event_id}: M{magnitude} at {event_time}, {n_stations} stations")

        # Get all waveforms for this event
        waveforms = np.stack([np.array(r["waveform"]) for r in records])  # Shape: (n_stations, 3, 12288)
        print(f"  Combined waveforms shape: {waveforms.shape}")

        # Get P and S indices for all stations
        p_indices = [r["p_phase_index"] for r in records if r["p_phase_index"] is not None]
        s_indices = [r["s_phase_index"] for r in records if r["s_phase_index"] is not None]
        print(f"  P picks: {len(p_indices)}, S picks: {len(s_indices)}")

        if i >= 2:
            break  # Remove to iterate all


# %% Filter by magnitude and SNR
def filter_dataset():
    """Filter dataset by criteria."""
    dataset = load_quakeflow_dataset(region="SC", years=[2025], days=[9])

    # Filter for M >= 2.0 events with SNR >= 5
    filtered = dataset.filter(
        lambda x: (x["event_magnitude"] is not None and x["event_magnitude"] >= 2.0)
                  and (x["snr"] is not None and x["snr"] >= 5.0)
    )

    print(f"Original: {len(dataset)}, Filtered: {len(filtered)}")

    for record in filtered:
        print(f"M{record['event_magnitude']:.1f} event, SNR={record['snr']:.1f}")
        break


# %% Visualize waveforms with phase picks
def visualize_waveforms(n_examples=3):
    """
    Plot waveforms with P and S phase arrivals marked.

    Creates a figure showing 3-component (E, N, Z) seismograms with
    vertical lines indicating P (blue) and S (red) arrival times.
    """
    import matplotlib.pyplot as plt

    dataset = load_quakeflow_dataset(region="SC", years=[2025], days=[9])

    # Filter for records with both P and S picks
    records_with_picks = [
        r for r in dataset
        if r["p_phase_index"] is not None and r["s_phase_index"] is not None
    ][:n_examples]

    if not records_with_picks:
        print("No records with both P and S picks found")
        return

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]

    sampling_rate = 100  # Hz
    channel_names = ["E", "N", "Z"]

    for ax, record in zip(axes, records_with_picks):
        waveform = np.array(record["waveform"])  # (3, 12288)
        n_samples = waveform.shape[1]
        time = np.arange(n_samples) / sampling_rate

        p_idx = record["p_phase_index"]
        s_idx = record["s_phase_index"]

        # Plot each channel with offset
        for i, (channel, name) in enumerate(zip(waveform, channel_names)):
            normalized = channel / (np.abs(channel).max() + 1e-10)
            ax.plot(time, normalized + i * 2.5, "k", linewidth=0.5, label=name if i == 0 else None)
            ax.text(-1, i * 2.5, name, fontsize=10, va="center", ha="right")

        # Mark P and S arrivals
        ax.axvline(p_idx / sampling_rate, color="blue", linewidth=1.5, linestyle="--", label=f"P ({p_idx / sampling_rate:.1f}s)")
        ax.axvline(s_idx / sampling_rate, color="red", linewidth=1.5, linestyle="--", label=f"S ({s_idx / sampling_rate:.1f}s)")

        # Labels
        station = f"{record['network']}.{record['station']}"
        mag = record["event_magnitude"]
        snr = record.get("snr")
        title = f"{record['event_id']} | {station} | M{mag:.1f}"
        if snr is not None:
            title += f" | SNR={snr:.1f}"
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, n_samples / sampling_rate)
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Waveforms with P and S Phase Arrivals", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("waveform_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to waveform_visualization.png")


if __name__ == "__main__":
    print("=" * 60)
    print("1. Iterate over phase picks")
    print("=" * 60)
    iterate_phases()

    print("\n" + "=" * 60)
    print("2. Group records by event")
    print("=" * 60)
    iterate_events()

    print("\n" + "=" * 60)
    print("3. Stream records")
    print("=" * 60)
    iterate_streaming()

    print("\n" + "=" * 60)
    print("4. Stream by event")
    print("=" * 60)
    iterate_events_streaming()

    print("\n" + "=" * 60)
    print("5. Filter by magnitude and SNR")
    print("=" * 60)
    filter_dataset()

    print("\n" + "=" * 60)
    print("6. Visualize waveforms with phase picks")
    print("=" * 60)
    visualize_waveforms(n_examples=3)

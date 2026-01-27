"""Compare Parquet output with existing HDF5 files on GCP (sampled comparison)."""

import json
import os

import fsspec
import h5py
import numpy as np
import pandas as pd

GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")


def load_parquet_sample(parquet_path: str, n_samples: int = 100) -> pd.DataFrame:
    """Load Parquet file and sample records."""
    df = pd.read_parquet(parquet_path)
    print(f"Parquet total: {len(df)} records")

    # Sample unique event-station pairs
    df["station_id"] = (
        df["network"] + "." + df["station"] + "." + df["location"] + "." + df["instrument"]
    )

    unique_pairs = df.groupby(["event_id", "station_id"]).first().reset_index()
    if len(unique_pairs) > n_samples:
        unique_pairs = unique_pairs.sample(n=n_samples, random_state=42)

    print(f"Parquet sampled: {len(unique_pairs)} event-station pairs")
    return df, unique_pairs


def load_hdf5_sample(hdf5_path: str, fs, sample_keys: list) -> dict:
    """Load specific records from HDF5 file."""
    records = {}

    with fs.open(hdf5_path, "rb") as f:
        with h5py.File(f, "r") as fp:
            for event_id, station_id in sample_keys:
                if event_id not in fp:
                    continue
                event_group = fp[event_id]
                if station_id not in event_group:
                    continue

                ds = event_group[station_id]
                waveform = ds[:]
                ds_attrs = dict(ds.attrs)
                event_attrs = dict(event_group.attrs)

                records[(event_id, station_id)] = {
                    "waveform": waveform,
                    "snr": ds_attrs.get("snr"),
                    "component": ds_attrs.get("component"),
                    "p_phase_index": ds_attrs.get("p_phase_index"),
                    "s_phase_index": ds_attrs.get("s_phase_index"),
                    "begin_time": event_attrs.get("begin_time"),
                }

    print(f"HDF5 loaded: {len(records)} matching records")
    return records


def compare_waveforms(parquet_df: pd.DataFrame, sampled_pairs: pd.DataFrame, hdf5_records: dict):
    """Compare waveforms between Parquet and HDF5."""
    print("\n" + "=" * 60)
    print("Waveform Comparison")
    print("=" * 60)

    matches = 0
    mismatches = 0
    missing = 0

    for _, row in sampled_pairs.iterrows():
        event_id = row["event_id"]
        station_id = row["station_id"]
        key = (event_id, station_id)

        if key not in hdf5_records:
            missing += 1
            continue

        hdf5_rec = hdf5_records[key]

        # Get parquet waveform (nested list -> numpy array)
        wf = row["waveform"]
        if isinstance(wf, (list, np.ndarray)):
            pq_waveform = np.array([np.array(ch, dtype=np.float32) for ch in wf])
        else:
            print(f"Unexpected waveform type: {type(wf)}")
            mismatches += 1
            continue
        hdf5_waveform = hdf5_rec["waveform"]

        # Compare shapes
        if pq_waveform.shape != hdf5_waveform.shape:
            print(f"Shape mismatch: {event_id}/{station_id}")
            print(f"  Parquet: {pq_waveform.shape}, HDF5: {hdf5_waveform.shape}")
            mismatches += 1
            continue

        # Compare values (allowing small numerical differences)
        if np.allclose(pq_waveform, hdf5_waveform, rtol=1e-4, atol=1e-4):
            matches += 1
        else:
            diff = np.abs(pq_waveform - hdf5_waveform)
            print(f"Value mismatch: {event_id}/{station_id}")
            print(f"  Max diff: {diff.max():.4f}, Mean diff: {diff.mean():.4f}")
            mismatches += 1

    print(f"\nResults: {matches} matches, {mismatches} mismatches, {missing} missing in HDF5")


def compare_metadata(sampled_pairs: pd.DataFrame, hdf5_records: dict):
    """Compare metadata between Parquet and HDF5."""
    print("\n" + "=" * 60)
    print("Metadata Comparison (first 10)")
    print("=" * 60)

    count = 0
    for _, row in sampled_pairs.iterrows():
        if count >= 10:
            break

        event_id = row["event_id"]
        station_id = row["station_id"]
        key = (event_id, station_id)

        if key not in hdf5_records:
            continue

        hdf5_rec = hdf5_records[key]

        print(f"\n{event_id} / {station_id}:")
        print(f"  SNR:       Parquet={row['snr']:.3f}, HDF5={hdf5_rec['snr']}")
        print(f"  Component: Parquet={row['component']}, HDF5={hdf5_rec['component']}")
        print(f"  Begin:     Parquet={row['begin_time']}")
        print(f"             HDF5={hdf5_rec['begin_time']}")

        # Phase indices (now using p_phase_index and s_phase_index columns)
        print(f"  P index:   Parquet={row.get('p_phase_index')}, HDF5={hdf5_rec.get('p_phase_index')}")
        print(f"  S index:   Parquet={row.get('s_phase_index')}, HDF5={hdf5_rec.get('s_phase_index')}")

        count += 1


def main():
    region = "NC"
    year = 2025
    day = 1
    n_samples = 100

    parquet_path = f"./{region}EDC/dataset/{year:04d}/{day:03d}.parquet"
    hdf5_path = f"quakeflow_dataset/{region}EDC/dataset/{year:04d}/{day:03d}/waveform.h5"

    print("=" * 60)
    print(f"Comparing Parquet vs HDF5 for {region}EDC {year}/{day:03d}")
    print(f"Sampling {n_samples} event-station pairs")
    print("=" * 60)

    if not os.path.exists(parquet_path):
        print(f"Parquet file not found: {parquet_path}")
        return

    # Load parquet sample
    parquet_df, sampled_pairs = load_parquet_sample(parquet_path, n_samples)

    # Get keys to load from HDF5
    sample_keys = list(zip(sampled_pairs["event_id"], sampled_pairs["station_id"]))

    # Load HDF5 sample
    with open(GCS_CREDENTIALS_PATH, "r") as f:
        token = json.load(f)
    fs = fsspec.filesystem("gs", token=token)

    try:
        hdf5_records = load_hdf5_sample(hdf5_path, fs, sample_keys)
    except FileNotFoundError:
        print(f"HDF5 file not found: {hdf5_path}")
        return

    # Compare
    compare_metadata(sampled_pairs, hdf5_records)
    compare_waveforms(parquet_df, sampled_pairs, hdf5_records)

    # Summary stats
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    has_p = parquet_df["p_phase_index"].notna().sum()
    has_s = parquet_df["s_phase_index"].notna().sum()
    has_both = (parquet_df["p_phase_index"].notna() & parquet_df["s_phase_index"].notna()).sum()
    print(f"Parquet records: {len(parquet_df)}")
    print(f"  With P: {has_p}, With S: {has_s}, With both P+S: {has_both}")
    print(f"Parquet SNR range: [{parquet_df['snr'].min():.2f}, {parquet_df['snr'].max():.2f}]")


if __name__ == "__main__":
    main()

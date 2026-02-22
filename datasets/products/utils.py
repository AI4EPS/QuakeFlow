# %%
"""Shared utilities for the polarity product pipeline."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

SAMPLING_RATE = 100.0


# %%
def detect_first_motion(waveform, phase_index, pre_window=3, post_window=10):
    """Detect polarity from waveform first motion on Z component.

    Finds the first local extremum after phase arrival relative to
    pre-arrival baseline. Trough -> "D", peak -> "U".

    Args:
        waveform: Array of shape (3, N) with channels [E, N, Z]
        phase_index: Sample index of the phase arrival
        pre_window: Samples before arrival for baseline (default: 3)
        post_window: Samples after arrival to search for extremum (default: 10)

    Returns:
        Dictionary with keys 'e', 'n', 'z' (amplitude diffs) and 'label' (U/D/""),
        or None if detection not possible.
    """
    if phase_index is None or (isinstance(phase_index, float) and np.isnan(phase_index)):
        return None
    phase_index = int(phase_index)
    if phase_index < pre_window or phase_index + post_window >= waveform.shape[1]:
        return None

    z_post = waveform[2, phase_index : phase_index + post_window]
    z_pre_mean = np.mean(waveform[2, phase_index - pre_window : phase_index])

    pol = ""
    for ii, s in enumerate(z_post):
        if s <= z_pre_mean:
            if (s < z_post[ii + 1]) | (ii == post_window - 2):
                if ii == 0:
                    break
                pol = "D"
                break
        else:
            if (s > z_post[ii + 1]) | (ii == post_window - 2):
                if ii == 0:
                    break
                pol = "U"
                break

    result = {}
    for i, comp in enumerate(["e", "n", "z"]):
        before = np.mean(waveform[i, phase_index - pre_window : phase_index])
        after = np.mean(waveform[i, phase_index : phase_index + post_window])
        result[comp] = float(after - before)
    result["label"] = pol
    return result


def normalize_trace(trace):
    """Normalize trace to [-1, 1] range for plotting."""
    trace = trace - np.mean(trace)
    max_val = np.max(np.abs(trace))
    if max_val > 0:
        return trace / max_val
    return trace


# %%
def load_dataset(path):
    """Load dataset from HDF5 or Parquet, returning a normalized DataFrame.

    Returns a DataFrame with columns:
        event_id, network, station, location, instrument,
        event_latitude, event_longitude, event_depth_km,
        station_latitude, station_longitude, station_depth_km,
        strike, dip, rake, p_phase_polarity, azimuth, takeoff_angle,
        p_phase_index, s_phase_index, snr, distance_km

    Waveforms are NOT loaded by default (use load_waveforms for that).
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return _load_parquet(path)
    elif path.suffix in (".h5", ".hdf5"):
        return _load_h5(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_waveforms(path, event_station_pairs=None):
    """Load waveforms from HDF5 or Parquet.

    Args:
        path: Path to the dataset file
        event_station_pairs: Optional list of (event_id, network, station) tuples
            to filter. If None, loads all.

    Returns:
        Dictionary mapping (event_id, network, station, location, instrument) -> waveform array (3, N)
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return _load_waveforms_parquet(path, event_station_pairs)
    elif path.suffix in (".h5", ".hdf5"):
        return _load_waveforms_h5(path, event_station_pairs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _load_parquet(path):
    """Load metadata from parquet (no waveforms)."""
    columns = [
        "event_id", "network", "station", "location", "instrument",
        "event_latitude", "event_longitude", "event_depth_km",
        "station_latitude", "station_longitude", "station_depth_km",
        "strike", "dip", "rake",
        "p_phase_polarity", "p_phase_index", "s_phase_index",
        "azimuth", "takeoff_angle", "snr", "distance_km",
    ]
    table = pq.read_table(path)
    available = [c for c in columns if c in table.schema.names]
    df = table.select(available).to_pandas()
    # Rename station_elevation_m -> station_depth_km if needed
    if "station_depth_km" not in df.columns and "station_elevation_m" in table.schema.names:
        df["station_depth_km"] = -pq.read_table(path, columns=["station_elevation_m"]).to_pandas()["station_elevation_m"] / 1000.0
    if "station_depth_km" not in df.columns:
        df["station_depth_km"] = 0.0
    return df


def _load_h5(path):
    """Load metadata from HDF5 (no waveforms)."""
    records = []
    with h5py.File(path, "r") as f:
        for event_id in tqdm(f.keys(), desc="Loading HDF5"):
            event_attrs = dict(f[event_id].attrs)
            for station_key in f[event_id].keys():
                ds = f[event_id][station_key]
                if not isinstance(ds, h5py.Dataset):
                    continue
                sa = dict(ds.attrs)
                if "station_latitude" not in sa:
                    continue
                records.append({
                    "event_id": event_id,
                    "network": str(sa.get("network", "")),
                    "station": str(sa.get("station", "")),
                    "location": str(sa.get("location", "")),
                    "instrument": str(sa.get("instrument", "")),
                    "event_latitude": float(event_attrs.get("event_latitude", np.nan)),
                    "event_longitude": float(event_attrs.get("event_longitude", np.nan)),
                    "event_depth_km": float(event_attrs.get("event_depth_km", np.nan)),
                    "station_latitude": float(sa.get("station_latitude", np.nan)),
                    "station_longitude": float(sa.get("station_longitude", np.nan)),
                    "station_depth_km": float(sa.get("station_depth_km", 0.0)),
                    "strike": float(event_attrs["strike"]) if "strike" in event_attrs else np.nan,
                    "dip": float(event_attrs["dip"]) if "dip" in event_attrs else np.nan,
                    "rake": float(event_attrs["rake"]) if "rake" in event_attrs else np.nan,
                    "p_phase_polarity": str(sa.get("p_phase_polarity", "")),
                    "p_phase_index": int(sa["p_phase_index"]) if "p_phase_index" in sa else None,
                    "s_phase_index": int(sa["s_phase_index"]) if "s_phase_index" in sa else None,
                    "azimuth": float(sa.get("azimuth", np.nan)),
                    "takeoff_angle": float(sa.get("takeoff_angle", np.nan)),
                    "snr": float(sa.get("snr", np.nan)),
                    "distance_km": float(sa.get("distance_km", np.nan)),
                })
    return pd.DataFrame(records)


def _load_waveforms_parquet(path, event_station_pairs=None):
    """Load waveforms from parquet."""
    columns = ["event_id", "network", "station", "location", "instrument", "waveform"]
    df = pq.read_table(path, columns=columns).to_pandas()
    waveforms = {}
    for _, row in df.iterrows():
        key = (row["event_id"], row["network"], row["station"], row["location"], row["instrument"])
        if event_station_pairs is not None and key[:3] not in event_station_pairs:
            continue
        waveforms[key] = np.stack(row["waveform"]).astype(np.float32)
    return waveforms


def _load_waveforms_h5(path, event_station_pairs=None):
    """Load waveforms from HDF5."""
    waveforms = {}
    with h5py.File(path, "r") as f:
        for event_id in f.keys():
            for station_key in f[event_id].keys():
                ds = f[event_id][station_key]
                if not isinstance(ds, h5py.Dataset):
                    continue
                sa = dict(ds.attrs)
                key = (
                    event_id,
                    str(sa.get("network", "")),
                    str(sa.get("station", "")),
                    str(sa.get("location", "")),
                    str(sa.get("instrument", "")),
                )
                if event_station_pairs is not None and key[:3] not in event_station_pairs:
                    continue
                waveforms[key] = ds[:]
    return waveforms

# %%
"""
Cut event waveforms and save to Parquet format.

Key logic:
1. For each event, use the first P phase across ALL stations as the reference time
2. All stations for the same event share the same time window (begin_time, end_time)
3. Cache mseed streams per station to avoid redundant reads
"""
import argparse
import gc
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import partial

import fsspec
import numpy as np
import obspy
from obspy.clients.fdsn import Client as FDSNClient
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

np.random.seed(42)

# Constants
GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
SAMPLING_RATE = 100.0
TIME_BEFORE = 40.96
TIME_AFTER = 40.96 * 2
NT = int(round((TIME_BEFORE + TIME_AFTER) * SAMPLING_RATE))  # 12288 samples
EARTH_RADIUS_KM = 6371.0
PARQUET_SCHEMA = pa.schema([
    # Event info
    ("event_id", pa.string()),
    ("event_time", pa.string()),
    ("event_latitude", pa.float32()),
    ("event_longitude", pa.float32()),
    ("event_depth_km", pa.float32()),
    ("event_magnitude", pa.float32()),
    ("event_magnitude_type", pa.string()),
    ("event_time_index", pa.int32()),
    # Station info
    ("network", pa.string()),
    ("station", pa.string()),
    ("location", pa.string()),
    ("instrument", pa.string()),
    ("station_latitude", pa.float32()),
    ("station_longitude", pa.float32()),
    ("station_elevation_m", pa.float32()),
    ("station_depth_km", pa.float32()),
    # Waveform
    ("waveform", pa.list_(pa.list_(pa.float32(), NT), 3)),
    ("component", pa.string()),
    ("snr", pa.float32()),
    ("begin_time", pa.string()),
    ("end_time", pa.string()),
    # P phase
    ("p_phase_time", pa.string()),
    ("p_phase_index", pa.int32()),
    ("p_phase_score", pa.float32()),
    ("p_phase_polarity", pa.string()),
    ("p_phase_remark", pa.string()),
    # S phase
    ("s_phase_time", pa.string()),
    ("s_phase_index", pa.int32()),
    ("s_phase_score", pa.float32()),
    ("s_phase_polarity", pa.string()),
    ("s_phase_remark", pa.string()),
    # Optional fields
    ("azimuth", pa.float32()),
    ("back_azimuth", pa.float32()),
    ("takeoff_angle", pa.float32()),
    ("distance_km", pa.float32()),
    # Focal mechanism
    ("strike", pa.float32()),
    ("dip", pa.float32()),
    ("rake", pa.float32()),
    ("num_first_motions", pa.float32()),
    ("first_motion_misfit", pa.float32()),
    ("num_sp_ratios", pa.float32()),
    ("sp_ratio_misfit", pa.float32()),
    ("plane1_uncertainty", pa.float32()),
    ("plane2_uncertainty", pa.float32()),
    ("fm_quality", pa.string()),
])


def calc_azimuth(lat1, lon1, lat2, lon2):
    """Calculate azimuth from point 1 to point 2 in degrees (vectorized)."""
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    az = np.degrees(np.arctan2(x, y))
    return np.round((az + 360) % 360, 2)


def calc_distance_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km (vectorized, Haversine)."""
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return np.round(EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)), 3)


def load_station_csv(region, gcs_fs):
    """Load station CSV from GCS and return a DataFrame with unique station coordinates.

    Loads primary region first, then supplements with the other region's stations
    to cover cross-network picks (e.g. CI stations in NCEDC phases).
    Aggregates to network+station level by taking the median coordinates.
    """
    primary = "SCEDC" if region == "SC" else "NCEDC"
    secondary = "NCEDC" if region == "SC" else "SCEDC"

    with gcs_fs.open(f"quakeflow_dataset/{primary}/stations.csv", "r") as f:
        stations = pd.read_csv(f)

    # Supplement with stations only found in the other region
    try:
        with gcs_fs.open(f"quakeflow_dataset/{secondary}/stations.csv", "r") as f:
            other = pd.read_csv(f)
        existing = set(zip(stations["network"], stations["station"]))
        mask = [
            (net, sta) not in existing
            for net, sta in zip(other["network"], other["station"])
        ]
        extra = other[mask]
        if len(extra) > 0:
            stations = pd.concat([stations, extra], ignore_index=True)
            print(f"Supplemented with {extra.groupby(['network', 'station']).ngroups} stations from {secondary}")
    except Exception as e:
        print(f"Warning: could not load {secondary} stations: {e}")

    # Aggregate to network+station level (median across channels/locations/time periods)
    station_coords = (
        stations.groupby(["network", "station"])
        .agg(
            station_latitude=("latitude", "median"),
            station_longitude=("longitude", "median"),
            station_elevation_m=("elevation_m", "median"),
            station_depth_km=("depth_km", "median"),
        )
        .reset_index()
    )

    return station_coords


def calc_snr(data, index0, noise_window=100, signal_window=100, gap_window=10):
    """
    Calculate signal-to-noise ratio for each channel.

    Window layout:
        |--noise_window--|--gap--|--signal_window--|
                                 ^
                               index0 (phase arrival)

    Args:
        data: Waveform array of shape (n_channels, n_samples)
        index0: Sample index of phase arrival
        noise_window: Samples for noise estimation (default: 100 = 1s at 100Hz)
        signal_window: Samples for signal estimation (default: 100 = 1s at 100Hz)
        gap_window: Samples between noise and signal windows (default: 10 = 0.1s)

    Returns:
        List of SNR values for each channel (0 if cannot compute)
    """
    n_channels, n_samples = data.shape

    # Phase arrival outside waveform
    if index0 < 0 or index0 >= n_samples:
        return [0.0] * n_channels

    # Define window boundaries
    noise_end = index0 - gap_window
    noise_start = noise_end - noise_window
    signal_start = index0
    signal_end = index0 + signal_window

    # Clamp to valid bounds
    noise_start = max(0, noise_start)
    noise_end = max(0, noise_end)
    signal_end = min(n_samples, signal_end)

    # Not enough samples for either window
    if noise_end <= noise_start or signal_end <= signal_start:
        return [0.0] * n_channels

    # Vectorized SNR computation across all channels
    noise_std = np.std(data[:, noise_start:noise_end], axis=1)
    signal_std = np.std(data[:, signal_start:signal_end], axis=1)

    # Avoid division by zero
    valid = (noise_std > 0) & (signal_std > 0)
    snr = np.zeros(n_channels, dtype=np.float64)
    if valid.any():
        snr[valid] = signal_std[valid] / noise_std[valid]

    return snr.tolist()


def prepare_picks(picks, events, config):
    """
    Prepare picks DataFrame with consistent time windows per event.

    For each event:
    - Find the first P phase time across ALL stations
    - Use that as the reference to calculate begin_time and end_time
    - All picks for the same event share the same time window
    """
    # Merge event info into picks
    picks = picks.merge(events, on="event_id", how="inner")

    # For each event, find the first P phase time across all stations
    # Exclude phase_score == 0 (placeholder phases with zero-weight, e.g. NCEDC weight code 4)
    p_picks = picks[(picks["phase_type"] == "P") & (picks["phase_score"] > 0)]
    first_p_per_event = p_picks.groupby("event_id")["phase_time"].min().reset_index()
    first_p_per_event.columns = ["event_id", "first_p_time"]

    # For events without usable P picks, use the first S pick (also excluding score == 0)
    events_with_p = set(first_p_per_event["event_id"])
    s_only_picks = picks[~picks["event_id"].isin(events_with_p) & (picks["phase_type"] == "S") & (picks["phase_score"] > 0)]
    first_s_per_event = s_only_picks.groupby("event_id")["phase_time"].min().reset_index()
    first_s_per_event.columns = ["event_id", "first_p_time"]  # Use same column name

    # Combine P and S-only events
    first_phase_per_event = pd.concat([first_p_per_event, first_s_per_event], ignore_index=True)

    # Merge first phase time back to picks
    picks = picks.merge(first_phase_per_event, on="event_id", how="inner")

    # Calculate time window (consistent across all stations for same event)
    picks["begin_time"] = picks["first_p_time"] - pd.Timedelta(seconds=config["time_before"])
    picks["end_time"] = picks["first_p_time"] + pd.Timedelta(seconds=config["time_after"])

    # Calculate phase_index relative to begin_time
    picks["phase_index"] = (
        (picks["phase_time"] - picks["begin_time"]).dt.total_seconds() * config["sampling_rate"]
    ).astype(int)

    return picks


def _download_inv_from_fdsn(network, station, region):
    """Download station inventory from FDSN web service (SCEDC/NCEDC first, then IRIS)."""
    primary = "SCEDC" if region == "SC" else "NCEDC"
    secondary = "NCEDC" if region == "SC" else "SCEDC"
    for provider in [primary, "IRIS", secondary]:
        try:
            client = FDSNClient(provider)
            inv = client.get_stations(network=network, station=station, level="response")
            print(f"Downloaded inventory for {network}.{station} from {provider}")
            return inv
        except Exception:
            continue
    raise Exception(f"No FDSN provider has inventory for {network}.{station}")


def _upload_inv_to_gcs(inv, network, station, region, gcs_fs):
    """Upload inventory XML to GCS, replacing any existing file."""
    folder = "SCEDC" if region == "SC" else "NCEDC"
    gcs_path = f"quakeflow_dataset/{folder}/FDSNstationXML/{network}/{network}.{station}.xml"
    with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
        inv.write(tmp.name, format="STATIONXML")
        gcs_fs.put(tmp.name, gcs_path)
    print(f"Uploaded FDSN inventory to gs://{gcs_path}")


def load_and_process_stream(mseed_3c, network, station, region, sampling_rate, gcs_fs, inv_cache=None):
    """Load mseed files and process (remove sensitivity, rotate to ZNE)."""
    mseed_files = mseed_3c.split("|")

    stream_3c = obspy.Stream()
    for c in mseed_files:
        try:
            with fsspec.open(c, "rb", anon=True) as f:
                stream = obspy.read(f)
                stream.merge(fill_value="latest")
                trace = stream[0]
                if trace.stats.sampling_rate != sampling_rate:
                    if trace.stats.sampling_rate % sampling_rate == 0:
                        try:
                            trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
                        except ArithmeticError:
                            print(f"Decimation factor too large for {c}, skipping")
                            continue
                    else:
                        trace.resample(sampling_rate)
                stream_3c.append(trace)
        except Exception as err:
            print(f"Error reading mseed: {err} on {c}")
            return None

    if len(stream_3c) == 0:
        return None

    try:
        cache_key = (network, station)
        if inv_cache is not None and cache_key in inv_cache:
            inv = inv_cache[cache_key]
            if inv is None:
                return None  # Previously failed, skip
        else:
            # Try GCS first, fall back to FDSN if missing or broken
            inv = None
            folder = "SCEDC" if region == "SC" else "NCEDC"
            gcs_path = f"quakeflow_dataset/{folder}/FDSNstationXML/{network}/{network}.{station}.xml"
            try:
                with gcs_fs.open(gcs_path, "r") as f:
                    inv = obspy.read_inventory(f)
            except Exception as gcs_err:
                print(f"GCS inventory missing for {network}.{station}: {gcs_err}, trying FDSN...")
                inv = None

            if inv is None:
                try:
                    inv = _download_inv_from_fdsn(network, station, region)
                except Exception as fdsn_err:
                    print(f"FDSN inventory also failed for {network}.{station}: {fdsn_err}")
                    if inv_cache is not None:
                        inv_cache[cache_key] = None
                    return None

            if inv_cache is not None:
                inv_cache[cache_key] = inv

        # Reconcile location codes between trace data and inventory.
        # PDS waveforms sometimes use "00" (SCEDC) or "N1" (NCEDC analog telemetry)
        # while FDSN inventory uses "". Treat these as equivalent.
        EQUIV_LOCS = {"", "00", "--", "N1"}
        inv_locs = set()
        for net_obj in inv:
            for sta_obj in net_obj:
                for ch in sta_obj:
                    inv_locs.add(ch.location_code)
        trace_locs = {tr.stats.location for tr in stream_3c}
        if trace_locs != inv_locs:
            inv_equiv = inv_locs & EQUIV_LOCS
            trace_equiv = trace_locs & EQUIV_LOCS
            if inv_equiv and trace_equiv and not (inv_equiv & trace_equiv):
                target_loc = next(iter(inv_equiv))
                for tr in stream_3c:
                    if tr.stats.location in EQUIV_LOCS:
                        tr.stats.location = target_loc

        # Apply inventory: try directly, fall back to FDSN if response doesn't match
        try:
            stream_3c.remove_sensitivity(inv)
        except Exception:
            try:
                inv = _download_inv_from_fdsn(network, station, region)
                stream_3c.remove_sensitivity(inv)
                _upload_inv_to_gcs(inv, network, station, region, gcs_fs)
                if inv_cache is not None:
                    inv_cache[cache_key] = inv
            except Exception as err:
                print(f"Warning: skipping {network}.{station}: {err}")
                if inv_cache is not None:
                    inv_cache[cache_key] = None  # Don't retry this station
                return None
        stream_3c.detrend("constant")
        stream_3c.rotate("->ZNE", inventory=inv)
    except Exception as err:
        print(f"Error processing inventory: {err}")
        return None

    return stream_3c


def process_station_group(picks_df, config, token=None):
    """
    Process all picks for a single station (same mseed_3c).

    Load the stream once, then cut waveforms for each event-station pair.
    One record per event-station with both P and S phase info.
    """
    region = config["region"]
    records = []

    # All picks share the same mseed_3c (same station, same day)
    mseed_3c = picks_df.iloc[0]["mseed_3c"]
    network = picks_df.iloc[0]["network"]
    station = picks_df.iloc[0]["station"]

    # Create per-process GCS filesystem (ProcessPoolExecutor can't share fsspec objects)
    gcs_fs = fsspec.filesystem("gs", token=token)

    # Load and cache the stream for this station
    stream_3c = load_and_process_stream(
        mseed_3c, network, station, region, config["sampling_rate"], gcs_fs
    )
    if stream_3c is None:
        return records

    # Group by event_id - one record per event-station pair
    for event_id, event_picks in picks_df.groupby("event_id"):
        # Get P and S picks for this event-station (exclude phase_score==0 placeholder picks)
        # Sort by score (desc) then phase_time (asc) for deterministic pick selection
        p_picks = event_picks[(event_picks["phase_type"] == "P") & (event_picks["phase_score"] > 0)].sort_values(["phase_score", "phase_time"], ascending=[False, True])
        s_picks = event_picks[(event_picks["phase_type"] == "S") & (event_picks["phase_score"] > 0)].sort_values(["phase_score", "phase_time"], ascending=[False, True])

        # All picks share same metadata (begin_time, end_time, event/station info)
        pick = event_picks.iloc[0]
        begin_time = pick["begin_time"]
        end_time = pick["end_time"]

        # Cut waveform from cached stream
        t_begin = obspy.UTCDateTime(begin_time)
        t_end = obspy.UTCDateTime(end_time)
        tmp = stream_3c.slice(t_begin, t_end, keep_empty_traces=False, nearest_sample=True)
        # Pad with zeros if trace doesn't cover full window
        for tr in tmp:
            tr.trim(t_begin, t_end, pad=True, fill_value=0)

        waveform = np.zeros((3, config["nt"]), dtype=np.float32)
        components = []
        for i, ch in enumerate(["E", "N", "Z"]):
            trace = tmp.select(component=ch)
            if len(trace) == 1:
                components.append(ch)
                waveform[i, : len(trace[0].data)] = trace[0].data[: config["nt"]] * 1e6

        # Calculate SNR based on first P arrival (or first S if no P)
        if len(p_picks) > 0:
            snr_index = int(p_picks.iloc[0]["phase_index"])
        elif len(s_picks) > 0:
            snr_index = int(s_picks.iloc[0]["phase_index"])
        else:
            continue

        snr = calc_snr(waveform, snr_index)
        if max(snr) == 0:
            continue

        # Skip constant/near-constant traces (e.g. dead channels)
        if np.std(waveform) == 0:
            continue

        # Build record - one per event-station with both P and S
        record = {
            # Event info
            "event_id": event_id,
            "event_time": pick["event_time"].strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "event_latitude": float(pick["event_latitude"]) if pd.notna(pick.get("event_latitude")) else None,
            "event_longitude": float(pick["event_longitude"]) if pd.notna(pick.get("event_longitude")) else None,
            "event_depth_km": float(pick["event_depth_km"]) if pd.notna(pick.get("event_depth_km")) else None,
            "event_magnitude": float(pick["event_magnitude"]) if pd.notna(pick.get("event_magnitude")) else None,
            "event_magnitude_type": pick.get("event_magnitude_type") or None,
            "event_time_index": int((pick["event_time"] - begin_time).total_seconds() * config["sampling_rate"]) if pd.notna(pick.get("event_time")) else None,

            # Station info
            "network": network,
            "station": station,
            "location": pick["location"],
            "instrument": pick["instrument"],
            "station_latitude": float(pick["station_latitude"]) if pd.notna(pick.get("station_latitude")) else None,
            "station_longitude": float(pick["station_longitude"]) if pd.notna(pick.get("station_longitude")) else None,
            "station_elevation_m": float(pick["station_elevation_m"]) if pd.notna(pick.get("station_elevation_m")) else None,
            "station_depth_km": float(pick["station_depth_km"]) if pd.notna(pick.get("station_depth_km")) else None,

            # Waveform (keep as numpy, convert to list at write time)
            "waveform": waveform,
            "component": "".join(components),
            "snr": round(max(snr), 3),
            "begin_time": begin_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),

            # P phase info
            "p_phase_time": p_picks.iloc[0]["phase_time"].strftime("%Y-%m-%dT%H:%M:%S.%f") if len(p_picks) > 0 else None,
            "p_phase_index": int(p_picks.iloc[0]["phase_index"]) if len(p_picks) > 0 else None,
            "p_phase_score": float(p_picks.iloc[0]["phase_score"]) if len(p_picks) > 0 and pd.notna(p_picks.iloc[0].get("phase_score")) else None,
            "p_phase_polarity": p_picks.iloc[0].get("phase_polarity") if len(p_picks) > 0 and p_picks.iloc[0].get("phase_polarity") else None,
            "p_phase_remark": p_picks.iloc[0].get("phase_remark") if len(p_picks) > 0 and p_picks.iloc[0].get("phase_remark") else None,

            # S phase info
            "s_phase_time": s_picks.iloc[0]["phase_time"].strftime("%Y-%m-%dT%H:%M:%S.%f") if len(s_picks) > 0 else None,
            "s_phase_index": int(s_picks.iloc[0]["phase_index"]) if len(s_picks) > 0 else None,
            "s_phase_score": float(s_picks.iloc[0]["phase_score"]) if len(s_picks) > 0 and pd.notna(s_picks.iloc[0].get("phase_score")) else None,
            "s_phase_polarity": s_picks.iloc[0].get("phase_polarity") if len(s_picks) > 0 and s_picks.iloc[0].get("phase_polarity") else None,
            "s_phase_remark": s_picks.iloc[0].get("phase_remark") if len(s_picks) > 0 and s_picks.iloc[0].get("phase_remark") else None,
        }

        # Optional fields (from first pick)
        for col in ["azimuth", "back_azimuth", "takeoff_angle", "distance_km"]:
            if col in pick.index and pd.notna(pick[col]) and pick[col] != "":
                try:
                    record[col] = float(pick[col])
                except (ValueError, TypeError):
                    pass

        # Focal mechanism (float fields)
        for col in ["strike", "dip", "rake",
                    "num_first_motions", "first_motion_misfit",
                    "num_sp_ratios", "sp_ratio_misfit",
                    "plane1_uncertainty", "plane2_uncertainty"]:
            if col in pick.index and pd.notna(pick[col]) and pick[col] != "":
                try:
                    record[col] = float(pick[col])
                except (ValueError, TypeError):
                    pass

        # Focal mechanism quality (string field)
        if "quality" in pick.index and pd.notna(pick["quality"]) and pick["quality"] != "":
            record["fm_quality"] = str(pick["quality"])

        records.append(record)

    # Clear stream to free memory
    stream_3c.clear()
    return records


# ============================================================
# Supplement: fill gaps using pre-cut event waveforms from PDS S3
# ============================================================

def parse_days(days_str):
    """Parse days string like '5', '1-30', '1,5,10'."""
    days = []
    for part in days_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            days.extend(range(int(start), int(end) + 1))
        else:
            days.append(int(part))
    return sorted(set(days))


def list_event_waveforms_s3(region, year, jday):
    """List available event waveform files from PDS S3.

    Returns dict mapping event_id (e.g. 'nc73983571') to S3 path(s).
    """
    s3_fs = fsspec.filesystem("s3", anon=True)

    if region == "SC":
        prefix = f"scedc-pds/event_waveforms/{year}/{year}_{jday:03d}/"
    else:
        prefix = f"ncedc-pds/event_waveforms/{year}/{year}.{jday:03d}/"

    try:
        files = s3_fs.ls(prefix)
    except FileNotFoundError:
        return {}

    event_files = {}  # event_id -> list of s3 paths
    for fpath in files:
        fname = fpath.split("/")[-1]
        if not fname.endswith(".ms"):
            continue

        if region == "SC":
            # Format: {numeric_id}.ms -> ci{numeric_id}
            event_num = fname.replace(".ms", "")
            event_id = f"ci{event_num}"
            s3_path = f"s3://{fpath}"
        else:
            # Format: NC.{numeric_id}.{catalog}.ms -> nc{numeric_id}
            parts = fname.split(".")
            if len(parts) < 3:
                continue
            event_num = parts[1]
            event_id = f"nc{event_num}"
            s3_path = f"s3://{fpath}"

        if event_id not in event_files:
            event_files[event_id] = []
        event_files[event_id].append(s3_path)

    return event_files


def process_event_stream(stream, station_picks, begin_time, end_time, config, inv, event_info):
    """Process traces for one station from an event waveform stream.

    Args:
        stream: obspy Stream containing traces for this station
        station_picks: DataFrame of picks for this event-station
        begin_time: pd.Timestamp for window start
        end_time: pd.Timestamp for window end
        config: dict with region, nt, sampling_rate, etc.
        inv: obspy Inventory for response removal (or None to skip)
        event_info: dict with event metadata

    Returns:
        Record dict, None if processing fails, or "retry_inv" to signal inventory mismatch.
    """
    network = station_picks.iloc[0]["network"]
    station = station_picks.iloc[0]["station"]

    # Reconcile location codes between trace data and inventory.
    EQUIV_LOCS = {"", "00", "--", "N1"}
    if inv is not None:
        inv_channels = []
        for net_obj in inv:
            for sta_obj in net_obj:
                for ch in sta_obj:
                    inv_channels.append((ch.location_code, ch.code, ch.start_date, ch.end_date))

        inv_locs = set(ic[0] for ic in inv_channels)
        trace_locs = {tr.stats.location for tr in stream}
        if trace_locs != inv_locs:
            inv_equiv = inv_locs & EQUIV_LOCS
            trace_equiv = trace_locs & EQUIV_LOCS
            if inv_equiv and trace_equiv and not (inv_equiv & trace_equiv):
                target_loc = next(iter(inv_equiv))
                for tr in stream:
                    if tr.stats.location in EQUIV_LOCS:
                        tr.stats.location = target_loc

        # Remap placeholder "XXX" channel codes to correct inventory channel
        for tr in stream:
            if tr.stats.channel == "XXX":
                tr_time = tr.stats.starttime
                candidates = []
                for loc, code, start, end in inv_channels:
                    loc_match = (tr.stats.location == loc) or ({tr.stats.location, loc} <= EQUIV_LOCS)
                    time_match = (start is None or tr_time >= start) and (end is None or tr_time <= end)
                    if loc_match and time_match and len(code) == 3:
                        candidates.append(code)
                if candidates:
                    z_cands = [c for c in candidates if c.endswith("Z")]
                    tr.stats.channel = z_cands[0] if z_cands else candidates[0]

    # Resample to target sampling rate
    for tr in list(stream):
        if tr.stats.sampling_rate != config["sampling_rate"]:
            if tr.stats.sampling_rate % config["sampling_rate"] == 0:
                try:
                    tr.decimate(int(tr.stats.sampling_rate / config["sampling_rate"]))
                except ArithmeticError:
                    stream.remove(tr)
                    continue
            else:
                tr.resample(config["sampling_rate"])

    # Remove instrument response
    if inv is not None:
        try:
            stream.remove_sensitivity(inv)
        except Exception:
            return "retry_inv"
        stream.detrend("constant")
        try:
            stream.rotate("->ZNE", inventory=inv)
        except Exception:
            pass

    # Slice to window and pad
    t_begin = obspy.UTCDateTime(begin_time)
    t_end = obspy.UTCDateTime(end_time)
    tmp = stream.slice(t_begin, t_end, keep_empty_traces=False, nearest_sample=True)
    for tr in tmp:
        tr.trim(t_begin, t_end, pad=True, fill_value=0)

    waveform = np.zeros((3, config["nt"]), dtype=np.float32)
    components = []
    for i, ch in enumerate(["E", "N", "Z"]):
        trace = tmp.select(component=ch)
        if len(trace) == 1:
            components.append(ch)
            waveform[i, : len(trace[0].data)] = trace[0].data[: config["nt"]] * 1e6

    if not components:
        return None

    # Get P and S picks (exclude phase_score==0 placeholder picks)
    # Sort by score (desc) then phase_time (asc) for deterministic pick selection
    p_picks = station_picks[(station_picks["phase_type"] == "P") & (station_picks["phase_score"] > 0)].sort_values(["phase_score", "phase_time"], ascending=[False, True])
    s_picks = station_picks[(station_picks["phase_type"] == "S") & (station_picks["phase_score"] > 0)].sort_values(["phase_score", "phase_time"], ascending=[False, True])

    if len(p_picks) > 0:
        snr_index = int(p_picks.iloc[0]["phase_index"])
    elif len(s_picks) > 0:
        snr_index = int(s_picks.iloc[0]["phase_index"])
    else:
        return None

    snr = calc_snr(waveform, snr_index)
    if max(snr) == 0:
        return None

    pick = station_picks.iloc[0]

    record = {
        "event_id": event_info["event_id"],
        "event_time": event_info["event_time"].strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "event_latitude": float(event_info["event_latitude"]) if pd.notna(event_info.get("event_latitude")) else None,
        "event_longitude": float(event_info["event_longitude"]) if pd.notna(event_info.get("event_longitude")) else None,
        "event_depth_km": float(event_info["event_depth_km"]) if pd.notna(event_info.get("event_depth_km")) else None,
        "event_magnitude": float(event_info["event_magnitude"]) if pd.notna(event_info.get("event_magnitude")) else None,
        "event_magnitude_type": event_info.get("event_magnitude_type") or None,
        "event_time_index": int((event_info["event_time"] - begin_time).total_seconds() * config["sampling_rate"]) if pd.notna(event_info.get("event_time")) else None,
        "network": network,
        "station": station,
        "location": pick.get("location", ""),
        "instrument": pick.get("instrument", ""),
        "station_latitude": float(pick["station_latitude"]) if pd.notna(pick.get("station_latitude")) else None,
        "station_longitude": float(pick["station_longitude"]) if pd.notna(pick.get("station_longitude")) else None,
        "station_elevation_m": float(pick["station_elevation_m"]) if pd.notna(pick.get("station_elevation_m")) else None,
        "station_depth_km": float(pick["station_depth_km"]) if pd.notna(pick.get("station_depth_km")) else None,
        "waveform": waveform,
        "component": "".join(components),
        "snr": round(max(snr), 3),
        "begin_time": begin_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "p_phase_time": p_picks.iloc[0]["phase_time"].strftime("%Y-%m-%dT%H:%M:%S.%f") if len(p_picks) > 0 else None,
        "p_phase_index": int(p_picks.iloc[0]["phase_index"]) if len(p_picks) > 0 else None,
        "p_phase_score": float(p_picks.iloc[0]["phase_score"]) if len(p_picks) > 0 and pd.notna(p_picks.iloc[0].get("phase_score")) else None,
        "p_phase_polarity": p_picks.iloc[0].get("phase_polarity") if len(p_picks) > 0 and p_picks.iloc[0].get("phase_polarity") else None,
        "p_phase_remark": p_picks.iloc[0].get("phase_remark") if len(p_picks) > 0 and p_picks.iloc[0].get("phase_remark") else None,
        "s_phase_time": s_picks.iloc[0]["phase_time"].strftime("%Y-%m-%dT%H:%M:%S.%f") if len(s_picks) > 0 else None,
        "s_phase_index": int(s_picks.iloc[0]["phase_index"]) if len(s_picks) > 0 else None,
        "s_phase_score": float(s_picks.iloc[0]["phase_score"]) if len(s_picks) > 0 and pd.notna(s_picks.iloc[0].get("phase_score")) else None,
        "s_phase_polarity": s_picks.iloc[0].get("phase_polarity") if len(s_picks) > 0 and s_picks.iloc[0].get("phase_polarity") else None,
        "s_phase_remark": s_picks.iloc[0].get("phase_remark") if len(s_picks) > 0 and s_picks.iloc[0].get("phase_remark") else None,
    }

    for col in ["azimuth", "back_azimuth", "takeoff_angle", "distance_km"]:
        if col in pick.index and pd.notna(pick[col]) and pick[col] != "":
            try:
                record[col] = float(pick[col])
            except (ValueError, TypeError):
                pass

    for col in ["strike", "dip", "rake",
                "num_first_motions", "first_motion_misfit",
                "num_sp_ratios", "sp_ratio_misfit",
                "plane1_uncertainty", "plane2_uncertainty"]:
        if col in pick.index and pd.notna(pick[col]) and pick[col] != "":
            try:
                record[col] = float(pick[col])
            except (ValueError, TypeError):
                pass

    if "quality" in pick.index and pd.notna(pick["quality"]) and pick["quality"] != "":
        record["fm_quality"] = str(pick["quality"])

    return record


def process_event_waveform(event_id, s3_paths, missing_picks, config, gcs_fs, inv_cache, existing_time_windows):
    """Process one event's waveform files, return records for missing stations."""
    region = config["region"]

    # Load all event waveform files and merge into one stream
    full_stream = obspy.Stream()
    for s3_path in s3_paths:
        try:
            with fsspec.open(s3_path, "rb", anon=True) as f:
                st = obspy.read(f)
                full_stream += st
        except Exception as err:
            print(f"  Error reading {s3_path}: {err}")

    if len(full_stream) == 0:
        return []

    # Get time window
    pick0 = missing_picks.iloc[0]
    begin_time = pick0["begin_time"]
    end_time = pick0["end_time"]
    if event_id in existing_time_windows:
        begin_time, end_time = existing_time_windows[event_id]

    event_info = {
        "event_id": event_id,
        "event_time": pick0["event_time"],
        "event_latitude": pick0.get("event_latitude"),
        "event_longitude": pick0.get("event_longitude"),
        "event_depth_km": pick0.get("event_depth_km"),
        "event_magnitude": pick0.get("event_magnitude"),
        "event_magnitude_type": pick0.get("event_magnitude_type"),
    }

    records = []

    for (network, station), sta_picks in missing_picks.groupby(["network", "station"]):
        sta_stream = full_stream.select(network=network, station=station)
        if len(sta_stream) == 0:
            continue

        sta_stream = sta_stream.copy()

        # Get inventory for response removal
        cache_key = (network, station)
        if cache_key in inv_cache:
            inv = inv_cache[cache_key]
        else:
            inv = None
            folder = "SCEDC" if region == "SC" else "NCEDC"
            gcs_path = f"quakeflow_dataset/{folder}/FDSNstationXML/{network}/{network}.{station}.xml"
            try:
                with gcs_fs.open(gcs_path, "r") as f:
                    inv = obspy.read_inventory(f)
            except Exception:
                try:
                    inv = _download_inv_from_fdsn(network, station, region)
                except Exception as err:
                    print(f"  No inventory for {network}.{station}: {err}")
                    inv = None
            inv_cache[cache_key] = inv

        if inv is None:
            continue

        record = process_event_stream(
            sta_stream, sta_picks, begin_time, end_time, config, inv, event_info
        )

        # Retry with fresh FDSN inventory if response didn't match
        used_fdsn = False
        fdsn_tried_key = ("fdsn_tried", network, station)
        if record == "retry_inv" and fdsn_tried_key not in inv_cache:
            inv_cache[fdsn_tried_key] = True
            try:
                inv = _download_inv_from_fdsn(network, station, region)
                inv_cache[cache_key] = inv
                used_fdsn = True
                sta_stream = full_stream.select(network=network, station=station).copy()
                for tr in list(sta_stream):
                    if tr.stats.sampling_rate != config["sampling_rate"]:
                        if tr.stats.sampling_rate % config["sampling_rate"] == 0:
                            try:
                                tr.decimate(int(tr.stats.sampling_rate / config["sampling_rate"]))
                            except ArithmeticError:
                                sta_stream.remove(tr)
                                continue
                        else:
                            tr.resample(config["sampling_rate"])
                record = process_event_stream(
                    sta_stream, sta_picks, begin_time, end_time, config, inv, event_info
                )
            except Exception as err:
                print(f"  FDSN retry failed for {network}.{station}: {err}")
                inv_cache[cache_key] = None
                record = None
        elif record == "retry_inv":
            record = None

        if record is not None and record != "retry_inv":
            records.append(record)
            if used_fdsn and inv is not None:
                try:
                    _upload_inv_to_gcs(inv, network, station, region, gcs_fs)
                except Exception:
                    pass

    return records


def supplement_from_s3(year, day, region, config, gcs_fs, existing_pairs, picks, existing_time_windows):
    """Supplement a day's records with S3 event waveforms for missing pairs.

    Args:
        year, day: Date identifiers
        region: "NC" or "SC"
        config: Config dict with nt, sampling_rate, etc.
        gcs_fs: GCS filesystem
        existing_pairs: Set of (event_id, network, station) already processed
        picks: Prepared picks DataFrame
        existing_time_windows: Dict of event_id -> (begin_time, end_time)

    Returns:
        List of new record dicts.
    """
    # Find missing event-station pairs
    catalog_pairs = set(zip(picks["event_id"], picks["network"], picks["station"]))
    missing_pairs = catalog_pairs - existing_pairs
    if not missing_pairs:
        return []

    missing_event_ids = set(p[0] for p in missing_pairs)

    # List available event waveform files from S3
    event_files = list_event_waveforms_s3(region, year, day)
    available_event_ids = missing_event_ids & set(event_files.keys())

    if not available_event_ids:
        return []

    print(f"Supplement: {len(missing_pairs)} missing pairs, {len(available_event_ids)} events available on S3")

    inv_cache = {}
    new_records = []
    for event_id in tqdm(sorted(available_event_ids), desc=f"  S3 supplement", leave=False):
        event_picks = picks[picks["event_id"] == event_id]
        missing_sta = set((n, s) for e, n, s in missing_pairs if e == event_id)
        mask = event_picks.apply(lambda r: (r["network"], r["station"]) in missing_sta, axis=1)
        missing_event_picks = event_picks[mask]

        if len(missing_event_picks) == 0:
            continue

        records = process_event_waveform(
            event_id, event_files[event_id],
            missing_event_picks, config, gcs_fs, inv_cache,
            existing_time_windows,
        )
        new_records.extend(records)

    return new_records


def cut_templates(jdays, root_path, data_path, result_path, region, config, bucket, protocol, token, recheck=False):
    fs = fsspec.filesystem(protocol, token=token)
    gcs_fs = fsspec.filesystem("gs", token=token)

    config["nt"] = NT
    config["sampling_rate"] = SAMPLING_RATE
    config["time_before"] = TIME_BEFORE
    config["time_after"] = TIME_AFTER
    markers_path = f"{region}EDC/done_parquet"

    # Load station coordinates once (shared across all days)
    station_coords = load_station_csv(region, gcs_fs)
    print(f"Loaded station coordinates: {len(station_coords)} stations")

    # Note: inv_cache removed — each subprocess creates its own gcs_fs and inv_cache
    # to avoid obspy segfaults from concurrent C library access in threads

    for jday in jdays:
        year, day = jday.split(".")
        year, day = int(year), int(day)

        # Skip already-processed days (fast check via .done marker)
        if not recheck:
            remote_done = f"{bucket}/{markers_path}/{year:04d}/{day:03d}.done"
            if fs.exists(remote_done):
                print(f"Skipping {year:04d}.{day:03d}: already processed")
                continue

        # Skip days where events.csv or phases.csv is missing (some days have no phase data in SCEDC)
        if protocol == "file":
            events_file = f"{root_path}/{data_path}/{year:04d}/{day:03d}/events.csv"
            phases_file = f"{root_path}/{data_path}/{year:04d}/{day:03d}/phases.csv"
            if not os.path.exists(events_file) or not os.path.exists(phases_file):
                print(f"Skipping {year:04d}.{day:03d}: missing events.csv or phases.csv")
                marker = f"{bucket}/{markers_path}/{year:04d}/{day:03d}.done"
                fs.touch(marker)
                continue
        else:
            events_file = f"{bucket}/{data_path}/{year:04d}/{day:03d}/events.csv"
            phases_file = f"{bucket}/{data_path}/{year:04d}/{day:03d}/phases.csv"
            if not fs.exists(events_file) or not fs.exists(phases_file):
                print(f"Skipping {year:04d}.{day:03d}: missing events.csv or phases.csv")
                marker = f"{bucket}/{markers_path}/{year:04d}/{day:03d}.done"
                fs.touch(marker)
                continue

        os.makedirs(f"{root_path}/{result_path}/{year:04d}", exist_ok=True)

        # ============================================================
        # Step 1: Load events
        # ============================================================
        if protocol == "file":
            events = pd.read_csv(f"{root_path}/{data_path}/{year:04d}/{day:03d}/events.csv", dtype=str)
        else:
            with fs.open(f"{bucket}/{data_path}/{year:04d}/{day:03d}/events.csv", "r") as fp:
                events = pd.read_csv(fp, dtype=str)

        events.fillna("", inplace=True)
        events.rename(columns={
            "time": "event_time",
            "latitude": "event_latitude",
            "longitude": "event_longitude",
            "depth_km": "event_depth_km",
            "magnitude": "event_magnitude",
            "magnitude_type": "event_magnitude_type",
        }, inplace=True)
        events["event_time"] = pd.to_datetime(events["event_time"])
        print(f"Events: {len(events)}")

        # ============================================================
        # Step 2: Load picks
        # ============================================================
        if protocol == "file":
            picks = pd.read_csv(f"{root_path}/{data_path}/{year:04d}/{day:03d}/phases.csv", dtype=str)
        else:
            with fs.open(f"{bucket}/{data_path}/{year:04d}/{day:03d}/phases.csv", "r") as fp:
                picks = pd.read_csv(fp, dtype=str)

        picks.fillna("", inplace=True)
        picks["phase_time"] = pd.to_datetime(picks["phase_time"])
        picks["phase_score"] = pd.to_numeric(picks["phase_score"], errors="coerce")
        print(f"Picks: {len(picks)}")

        # ============================================================
        # Step 3: Merge station coordinates from station CSV
        # ============================================================
        picks = picks.merge(station_coords, on=["network", "station"], how="left")
        n_missing = picks["station_latitude"].isna().sum()
        if n_missing > 0:
            print(f"Warning: {n_missing}/{len(picks)} picks missing station coordinates")

        # ============================================================
        # Step 4: Prepare picks with consistent time windows per event
        # ============================================================
        picks = prepare_picks(picks, events, config)
        print(f"Picks after merge: {len(picks)}")

        # ============================================================
        # Step 4b: Recheck — decide action based on existing parquet
        #   action: "full_process" | "supplement" | "skip"
        # ============================================================
        recheck_action = "full_process"
        existing_df = None  # loaded for "supplement" action

        if recheck:
            remote_parquet = f"{bucket}/{result_path}/{year:04d}/{day:03d}.parquet"
            try:
                check_cols = [
                    "event_id", "network", "station", "location", "instrument",
                    "begin_time", "end_time",
                    "event_time", "event_latitude", "event_longitude",
                    "event_depth_km", "event_magnitude",
                    "p_phase_time", "p_phase_score",
                    "s_phase_time", "s_phase_score",
                    "station_latitude", "station_longitude", "station_elevation_m",
                ]
                with fs.open(remote_parquet, "rb") as f:
                    existing_meta = pq.read_table(f, columns=check_cols).to_pandas()

                existing_events = set(existing_meta["event_id"].unique())
                expected_events = set(picks["event_id"].unique())

                # --- Step A: Check metadata correctness on EXISTING records ---
                # Collect event_ids with wrong metadata (to delete and reprocess)
                common_events = expected_events & existing_events
                bad_event_ids = set()

                if common_events:
                    common_meta = existing_meta[existing_meta["event_id"].isin(common_events)]
                    common_picks = picks[picks["event_id"].isin(common_events)]

                    # A1. Time windows
                    expected_tw = common_picks.groupby("event_id")[["begin_time", "end_time"]].first().reset_index()
                    expected_tw["begin_time"] = expected_tw["begin_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                    expected_tw["end_time"] = expected_tw["end_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                    existing_tw = common_meta.groupby("event_id")[["begin_time", "end_time"]].first().reset_index()
                    merged_tw = expected_tw.merge(existing_tw, on="event_id", suffixes=("_new", "_old"))
                    tw_diff = (merged_tw["begin_time_new"] != merged_tw["begin_time_old"]) | \
                              (merged_tw["end_time_new"] != merged_tw["end_time_old"])
                    if tw_diff.any():
                        bad_event_ids |= set(merged_tw.loc[tw_diff, "event_id"])

                    # A2. Event metadata (location, magnitude)
                    ev_float_cols = ["event_latitude", "event_longitude",
                                     "event_depth_km", "event_magnitude"]
                    expected_ev = common_picks.groupby("event_id")[ev_float_cols].first().reset_index()
                    existing_ev = common_meta.groupby("event_id")[ev_float_cols].first().reset_index()
                    merged_ev = expected_ev.merge(existing_ev, on="event_id", suffixes=("_new", "_old"))
                    for col in ev_float_cols:
                        new_vals = pd.to_numeric(merged_ev[f"{col}_new"], errors="coerce")
                        old_vals = pd.to_numeric(merged_ev[f"{col}_old"], errors="coerce")
                        diff_mask = (new_vals - old_vals).abs() > 0.001
                        if diff_mask.any():
                            bad_event_ids |= set(merged_ev.loc[diff_mask, "event_id"])

                    # A3. Event time
                    expected_et = common_picks.groupby("event_id")["event_time"].first().reset_index()
                    expected_et["event_time"] = pd.to_datetime(expected_et["event_time"])
                    existing_et = common_meta.groupby("event_id")["event_time"].first().reset_index()
                    existing_et["event_time"] = pd.to_datetime(existing_et["event_time"])
                    merged_et = expected_et.merge(existing_et, on="event_id", suffixes=("_new", "_old"))
                    time_diff = (merged_et["event_time_new"] - merged_et["event_time_old"]).abs() > pd.Timedelta("1ms")
                    if time_diff.any():
                        bad_event_ids |= set(merged_et.loc[time_diff, "event_id"])

                    # A4. Phase pick times (use same sort as processing: best score, earliest time)
                    #   Skip records where parquet has phase_score==0 — those are handled separately
                    score_cols = {"P": "p_phase_score", "S": "s_phase_score"}
                    for phase_type, phase_col in [("P", "p_phase_time"), ("S", "s_phase_time")]:
                        type_picks = common_picks[common_picks["phase_type"] == phase_type].sort_values(
                            ["phase_score", "phase_time"], ascending=[False, True]
                        )
                        if len(type_picks) == 0:
                            continue
                        grp_cols = ["event_id", "network", "station", "location", "instrument"]
                        # catalog picks may not have location/instrument yet, use what's available
                        cat_grp = [c for c in grp_cols if c in type_picks.columns]
                        pq_grp = [c for c in grp_cols if c in common_meta.columns]
                        expected_phase = type_picks.groupby(cat_grp)["phase_time"].first()
                        existing_phase = common_meta.groupby(pq_grp)[phase_col].first()
                        existing_score = common_meta.groupby(pq_grp)[score_cols[phase_type]].first()
                        # Skip records with score==0 (will be patched); None means no pick (keep for comparison)
                        not_zero = existing_score != 0
                        existing_phase = existing_phase[not_zero]
                        common_idx = expected_phase.index.intersection(existing_phase.index)
                        if len(common_idx) > 0:
                            exp_vals = pd.to_datetime(expected_phase.loc[common_idx], errors="coerce")
                            ext_vals = pd.to_datetime(existing_phase.loc[common_idx], errors="coerce")
                            both_valid = exp_vals.notna() & ext_vals.notna()
                            if both_valid.any():
                                diff_mask = (exp_vals[both_valid] - ext_vals[both_valid]).abs() > pd.Timedelta("1ms")
                                if diff_mask.any():
                                    bad_event_ids |= set(idx[0] for idx in diff_mask[diff_mask].index)

                    # A5. Station coordinates — affects all events at that station
                    sta_cols = ["station_latitude", "station_longitude", "station_elevation_m"]
                    expected_sta = common_picks.groupby(["network", "station"])[sta_cols].first().reset_index()
                    existing_sta = common_meta.groupby(["network", "station"])[sta_cols].first().reset_index()
                    merged_sta = expected_sta.merge(existing_sta, on=["network", "station"], suffixes=("_new", "_old"))
                    for col in sta_cols:
                        new_vals = pd.to_numeric(merged_sta[f"{col}_new"], errors="coerce")
                        old_vals = pd.to_numeric(merged_sta[f"{col}_old"], errors="coerce")
                        diff_mask = (new_vals - old_vals).abs() > 0.001
                        if diff_mask.any():
                            bad_stations = set(zip(merged_sta.loc[diff_mask, "network"], merged_sta.loc[diff_mask, "station"]))
                            for eid, net, sta in zip(common_meta["event_id"], common_meta["network"], common_meta["station"]):
                                if (net, sta) in bad_stations:
                                    bad_event_ids.add(eid)

                # --- Step B: Collect all changes ---
                # Note: score==0 is bad; None means no pick (normal, not bad)
                new_event_ids = expected_events - existing_events
                p_bad = existing_meta["p_phase_score"] == 0
                s_bad = existing_meta["s_phase_score"] == 0
                both_bad = p_bad & s_bad
                n_both_bad = int(both_bad.sum())
                p_only_bad = p_bad & ~s_bad
                s_only_bad = ~p_bad & s_bad
                n_patch = int(p_only_bad.sum() + s_only_bad.sum())

                if bad_event_ids or new_event_ids or n_both_bad > 0 or n_patch > 0:
                    recheck_action = "supplement"
                    changes = []
                    if bad_event_ids:
                        changes.append(f"{len(bad_event_ids)} events with wrong metadata to reprocess")
                    if new_event_ids:
                        changes.append(f"{len(new_event_ids)} new events")
                    if n_both_bad > 0:
                        changes.append(f"{n_both_bad} records to delete (both phase_score<=0)")
                    if n_patch > 0:
                        changes.append(f"{n_patch} records to patch (one phase_score<=0)")
                    print(f"Supplement {year:04d}.{day:03d}: {'; '.join(changes)}")
                else:
                    recheck_action = "skip"
                    print(f"Skipping {year:04d}.{day:03d}: parquet up to date ({len(existing_events)} events, {len(existing_meta)} records)")

            except FileNotFoundError:
                recheck_action = "full_process"
            except Exception as err:
                recheck_action = "full_process"
                print(f"Could not check existing parquet: {err}")

        if recheck_action == "skip":
            del picks, events
            gc.collect()
            continue

        # ============================================================
        # Step 5: Recalculate azimuth, back_azimuth, distance_km
        # ============================================================
        has_coords = (
            picks["station_latitude"].notna()
            & picks["event_latitude"].notna()
            & (picks["event_latitude"] != "")
        )
        if has_coords.any():
            ev_lat = pd.to_numeric(picks.loc[has_coords, "event_latitude"]).values
            ev_lon = pd.to_numeric(picks.loc[has_coords, "event_longitude"]).values
            st_lat = picks.loc[has_coords, "station_latitude"].values.astype(float)
            st_lon = picks.loc[has_coords, "station_longitude"].values.astype(float)

            picks.loc[has_coords, "azimuth"] = calc_azimuth(ev_lat, ev_lon, st_lat, st_lon)
            picks.loc[has_coords, "back_azimuth"] = calc_azimuth(st_lat, st_lon, ev_lat, ev_lon)
            picks.loc[has_coords, "distance_km"] = calc_distance_km(ev_lat, ev_lon, st_lat, st_lon)

        # ============================================================
        # Step 6: Load focal mechanisms (optional)
        # ============================================================
        try:
            if protocol == "file":
                mechanisms = pd.read_csv(f"{root_path}/{data_path}/{year:04d}/{day:03d}/focal_mechanisms.csv", dtype=str)
            else:
                with fs.open(f"{bucket}/{data_path}/{year:04d}/{day:03d}/focal_mechanisms.csv", "r") as fp:
                    mechanisms = pd.read_csv(fp, dtype=str)
            fm_cols = ["event_id", "strike", "dip", "rake",
                       "num_first_motions", "first_motion_misfit",
                       "num_sp_ratios", "sp_ratio_misfit",
                       "plane1_uncertainty", "plane2_uncertainty", "quality"]
            fm_cols = [c for c in fm_cols if c in mechanisms.columns]
            picks = picks.merge(mechanisms[fm_cols], on="event_id", how="left")
            print(f"Focal mechanisms: {len(mechanisms)}")
        except Exception:
            pass  # Focal mechanisms are optional

        # ============================================================
        # Step 6b: For supplement mode, load existing parquet and filter picks
        # ============================================================
        if recheck_action == "supplement":
            with fs.open(f"{bucket}/{result_path}/{year:04d}/{day:03d}.parquet", "rb") as f:
                existing_df = pq.read_table(f).to_pandas()

            # Remove records for events with wrong metadata (will be reprocessed)
            n_deleted = 0
            if bad_event_ids:
                bad_meta_mask = existing_df["event_id"].isin(bad_event_ids)
                n_bad_meta = int(bad_meta_mask.sum())
                if n_bad_meta > 0:
                    print(f"Removing {n_bad_meta} records from {len(bad_event_ids)} events with wrong metadata")
                    existing_df = existing_df[~bad_meta_mask]
                    n_deleted += n_bad_meta

            # Clean up phase_score==0 records (None means no pick, which is fine)
            p_bad = existing_df["p_phase_score"] == 0
            s_bad = existing_df["s_phase_score"] == 0

            # Delete records where both P and S have score==0
            both_bad = p_bad & s_bad
            n_both_bad = int(both_bad.sum())
            if n_both_bad > 0:
                print(f"Deleting {n_both_bad} records with both phase_score==0")
                existing_df = existing_df[~both_bad]
                p_bad = p_bad[~both_bad]
                s_bad = s_bad[~both_bad]
                n_deleted += n_both_bad

            # Patch records where only one phase has score==0 (null out the bad phase fields)
            p_bad_only = p_bad & ~s_bad
            s_bad_only = ~p_bad & s_bad
            n_patched = int(p_bad_only.sum() + s_bad_only.sum())
            if p_bad_only.any():
                for col in ["p_phase_time", "p_phase_index", "p_phase_score", "p_phase_polarity", "p_phase_remark"]:
                    if col in existing_df.columns:
                        existing_df.loc[p_bad_only, col] = None
            if s_bad_only.any():
                for col in ["s_phase_time", "s_phase_index", "s_phase_score", "s_phase_polarity", "s_phase_remark"]:
                    if col in existing_df.columns:
                        existing_df.loc[s_bad_only, col] = None
            if n_patched > 0:
                print(f"Patched {n_patched} records (nulled out phase_score<=0 fields)")
                n_deleted += n_patched

            # Filter picks to only missing event-station pairs
            existing_pairs = set(zip(existing_df["event_id"], existing_df["network"], existing_df["station"]))
            catalog_pairs = set(zip(picks["event_id"], picks["network"], picks["station"]))
            missing_pairs = catalog_pairs - existing_pairs

            if not missing_pairs:
                # Only deletions/patches, no new events to process — write updated parquet
                if n_deleted > 0:
                    print(f"Writing updated parquet ({len(existing_df)} records after cleanup)")
                    existing_df.sort_values(by=["event_id", "distance_km"], inplace=True)
                    table = pa.Table.from_pandas(existing_df, schema=PARQUET_SCHEMA, preserve_index=False)
                    local_path = f"{root_path}/{result_path}/{year:04d}/{day:03d}.parquet"
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    pq.write_table(table, local_path, compression="zstd")
                    remote_path = f"{bucket}/{result_path}/{year:04d}/{day:03d}.parquet"
                    fs.put(local_path, remote_path)
                    os.remove(local_path)
                    print(f"Uploaded: {remote_path}")
                    del table
                del existing_df, picks, events
                gc.collect()
                continue

            # Keep only picks for missing pairs
            picks = picks[picks.apply(
                lambda r: (r["event_id"], r["network"], r["station"]) in missing_pairs, axis=1
            )]
            print(f"Supplement mode: {len(missing_pairs)} missing pairs, {len(picks)} picks to process")

        # ============================================================
        # Step 7: Match picks with mseed files
        # ============================================================
        region_folder = "SCEDC" if region == "SC" else "NCEDC"
        try:
            with gcs_fs.open(f"quakeflow_dataset/{region_folder}/mseed/{year}/{day:03d}.txt", "r") as f:
                mseeds = [x.strip() for x in f.readlines()]
        except FileNotFoundError:
            mseeds = []
            if recheck_action != "supplement":
                print(f"Skipping {year:04d}/{day:03d}: no mseed list found")
                marker = f"{bucket}/{markers_path}/{year:04d}/{day:03d}.done"
                fs.touch(marker)
                del picks, events
                gc.collect()
                continue

        os.makedirs(f"{root_path}/{result_path}/{year:04d}", exist_ok=True)
        local_path = f"{root_path}/{result_path}/{year:04d}/{day:03d}.parquet"
        all_records = []

        if mseeds:
            mseeds_df = pd.DataFrame(mseeds, columns=["mseed_3c"])
            mseeds_df["fname"] = mseeds_df["mseed_3c"].apply(lambda x: x.split("|")[0].split("/")[-1])

            if region == "SC":
                mseeds_df["network"] = mseeds_df["fname"].str[:2]
                mseeds_df["station"] = mseeds_df["fname"].str[2:7].str.rstrip("_")
                mseeds_df["instrument"] = mseeds_df["fname"].str[7:9]
                mseeds_df["location"] = mseeds_df["fname"].str[10:12].str.rstrip("_")
            else:
                mseeds_df["network"] = mseeds_df["fname"].apply(lambda x: x.split(".")[1])
                mseeds_df["station"] = mseeds_df["fname"].apply(lambda x: x.split(".")[0])
                mseeds_df["instrument"] = mseeds_df["fname"].apply(lambda x: x.split(".")[2][:-1])
                mseeds_df["location"] = mseeds_df["fname"].apply(lambda x: x.split(".")[3])

            picks = picks.merge(
                mseeds_df[["mseed_3c", "network", "station", "location", "instrument"]],
                on=["network", "station", "location", "instrument"],
                how="inner"
            )

        if len(picks) == 0 and recheck_action != "supplement":
            print(f"No picks matched for {year:04d}/{day:03d}")
            marker = f"{bucket}/{markers_path}/{year:04d}/{day:03d}.done"
            fs.touch(marker)
            del picks, events
            gc.collect()
            continue

        # ============================================================
        # Step 8: Process stations from continuous data
        # ============================================================
        if "mseed_3c" in picks.columns and len(picks) > 0:
            print(f"Matched picks: {len(picks)}")
            station_groups = list(picks.groupby("mseed_3c"))
            ncpu = min(8, multiprocessing.cpu_count() * 2)
            print(f"Processing {len(station_groups)} stations with {ncpu} workers")

            with ProcessPoolExecutor(max_workers=ncpu) as executor:
                futures = set()
                pbar = tqdm(total=len(station_groups), desc=f"{year:04d}/{day:03d}")

                for _, group_df in station_groups:
                    if len(futures) >= ncpu:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for f in done:
                            records = f.result()
                            if records:
                                all_records.extend(records)
                            pbar.update(1)

                    futures.add(
                        executor.submit(partial(process_station_group, config=config, token=token), group_df)
                    )

                for f in as_completed(futures):
                    records = f.result()
                    if records:
                        all_records.extend(records)
                    pbar.update(1)

                pbar.close()

        # ============================================================
        # Step 9: Supplement from S3 event waveforms for missing pairs
        # ============================================================
        # Build pairs from new records + existing parquet (supplement mode)
        new_pairs = set((r["event_id"], r["network"], r["station"]) for r in all_records)
        if recheck_action == "supplement" and existing_df is not None:
            parquet_pairs = set(zip(existing_df["event_id"], existing_df["network"], existing_df["station"]))
            s3_existing_pairs = new_pairs | parquet_pairs
        else:
            s3_existing_pairs = new_pairs

        existing_time_windows = {}
        for event_id, grp in picks.groupby("event_id"):
            row = grp.iloc[0]
            existing_time_windows[event_id] = (row["begin_time"], row["end_time"])

        try:
            supplement_records = supplement_from_s3(
                year, day, region, config, gcs_fs,
                s3_existing_pairs, picks, existing_time_windows,
            )
            if supplement_records:
                all_records.extend(supplement_records)
                print(f"Supplement: {len(supplement_records)} records from S3 event waveforms")
        except Exception as err:
            print(f"Supplement failed (non-fatal): {err}")

        if not all_records and recheck_action != "supplement":
            print(f"No records for {year:04d}/{day:03d}")
            marker = f"{bucket}/{markers_path}/{year:04d}/{day:03d}.done"
            fs.touch(marker)
            print(f"Marked done (no data): {marker}")
            del picks, events
            gc.collect()
            continue

        # ============================================================
        # Step 10: Write parquet (merge with existing in supplement mode)
        # ============================================================
        if all_records:
            new_df = pd.DataFrame(all_records)
            new_df["waveform"] = new_df["waveform"].apply(lambda x: x.tolist())
            for field in PARQUET_SCHEMA:
                if field.name not in new_df.columns:
                    new_df[field.name] = None
            new_df = new_df[[field.name for field in PARQUET_SCHEMA]]
            print(f"New records: {len(new_df)}")
        else:
            new_df = None

        if recheck_action == "supplement" and existing_df is not None:
            if new_df is None and n_deleted == 0:
                # No new records and no deletions — nothing changed
                print(f"Skipping {year:04d}.{day:03d}: supplement produced no changes")
                del existing_df, all_records, picks, events
                gc.collect()
                continue
            if new_df is not None:
                # Align columns to avoid FutureWarning about empty/all-NA columns
                for field in PARQUET_SCHEMA:
                    if field.name not in existing_df.columns:
                        existing_df[field.name] = None
                existing_df = existing_df[[field.name for field in PARQUET_SCHEMA]]
                df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                df = existing_df
            del existing_df
        elif new_df is not None:
            df = new_df
        else:
            del picks, events
            gc.collect()
            continue

        del all_records

        df.sort_values(by=["event_id", "distance_km"], inplace=True)
        table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
        pq.write_table(table, local_path, compression="zstd")
        print(f"Total records: {len(df)}")
        del df, table
        gc.collect()

        print(f"Saved: {local_path}")

        # Upload and cleanup
        remote_path = f"{bucket}/{result_path}/{year:04d}/{day:03d}.parquet"
        fs.put(local_path, remote_path)
        print(f"Uploaded: {remote_path}")
        os.remove(local_path)

        marker = f"{bucket}/{markers_path}/{year:04d}/{day:03d}.done"
        fs.touch(marker)
        print(f"Marked done: {marker}")

        # Free memory
        del picks, events
        gc.collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="SC")
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument("--bucket", type=str, default="quakeflow_dataset")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--days", type=str, help="Comma-separated days to process (e.g., '1,2,3' or '1-30')")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--recheck", action="store_true", help="Bypass .done markers and check parquet metadata to decide whether to reprocess")
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    protocol = "gs"
    with open(GCS_CREDENTIALS_PATH, "r") as fp:
        token = json.load(fp)

    args = parse_args()
    region = args.region
    root_path = args.root_path
    bucket = args.bucket
    year = args.year

    # Determine which days to process
    if args.days:
        day_nums = parse_days(args.days)
    else:
        # Fall back to node-based splitting for backward compatibility
        num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
        all_days = list(range(1, num_jday + 1))
        day_nums = list(np.array_split(all_days, args.num_nodes)[args.node_rank])

    jdays = [f"{year}.{d:03d}" for d in day_nums]
    print(f"Processing {len(jdays)} days: {min(jdays)} to {max(jdays)}")

    config = {"region": region}
    data_path = f"{region}EDC/catalog"
    result_path = f"{region}EDC/waveform_parquet"

    os.makedirs(f"{root_path}/{result_path}", exist_ok=True)
    cut_templates(jdays, root_path, data_path, result_path, region, config, bucket, protocol, token, recheck=args.recheck)

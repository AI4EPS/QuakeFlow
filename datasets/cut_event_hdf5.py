# %%
"""
Cut event waveforms and save to HDF5 format.

Key logic:
1. For each event, use the first P phase across ALL stations as the reference time
2. All stations for the same event share the same time window (begin_time, end_time)
3. Cache mseed streams per station to avoid redundant reads

HDF5 structure:
    /{event_id}/                    (group, event-level attributes)
    /{event_id}/{station_id}        (dataset, shape=(3, NT), station/phase attributes)
"""
import argparse
import gc
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import fsspec
import h5py
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm

np.random.seed(42)

# Constants
GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
SAMPLING_RATE = 100.0
TIME_BEFORE = 40.96
TIME_AFTER = 40.96 * 2
NT = int(round((TIME_BEFORE + TIME_AFTER) * SAMPLING_RATE))  # 12288 samples
EARTH_RADIUS_KM = 6371.0


def set_attr(obj, key, value):
    """Set an HDF5 attribute only if the value is not empty/None."""
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
        if all(v == "" for v in value):
            return
    obj.attrs[key] = value


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

    Aggregates to network+station level by taking the median coordinates.
    """
    folder = "SCEDC" if region == "SC" else "NCEDC"
    csv_path = f"quakeflow_dataset/{folder}/stations.csv"
    with gcs_fs.open(csv_path, "r") as f:
        stations = pd.read_csv(f)

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
    Calculate signal-to-noise ratio for each channel (vectorized).

    Window layout:
        |--noise_window--|--gap--|--signal_window--|
                                 ^
                               index0 (phase arrival)
    """
    n_channels, n_samples = data.shape

    if index0 < 0 or index0 >= n_samples:
        return [0.0] * n_channels

    noise_end = index0 - gap_window
    noise_start = noise_end - noise_window
    signal_start = index0
    signal_end = index0 + signal_window

    noise_start = max(0, noise_start)
    noise_end = max(0, noise_end)
    signal_end = min(n_samples, signal_end)

    if noise_end <= noise_start or signal_end <= signal_start:
        return [0.0] * n_channels

    noise_std = np.std(data[:, noise_start:noise_end], axis=1)
    signal_std = np.std(data[:, signal_start:signal_end], axis=1)

    valid = (noise_std > 0) & (signal_std > 0)
    snr = np.where(valid, signal_std / noise_std, 0.0)

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
    p_picks = picks[picks["phase_type"] == "P"]
    first_p_per_event = p_picks.groupby("event_id")["phase_time"].min().reset_index()
    first_p_per_event.columns = ["event_id", "first_p_time"]

    # For events without P picks, use the first S pick
    events_with_p = set(first_p_per_event["event_id"])
    s_only_picks = picks[~picks["event_id"].isin(events_with_p) & (picks["phase_type"] == "S")]
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
                        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
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
        else:
            folder = "SCEDC" if region == "SC" else "NCEDC"
            with gcs_fs.open(f"quakeflow_dataset/{folder}/FDSNstationXML/{network}/{network}.{station}.xml", "r") as f:
                inv = obspy.read_inventory(f)
            if inv_cache is not None:
                inv_cache[cache_key] = inv
        stream_3c.remove_sensitivity(inv)
        stream_3c.detrend("constant")
        stream_3c.rotate("->ZNE", inventory=inv)
    except Exception as err:
        print(f"Error processing inventory: {err}")
        return None

    return stream_3c


def process_station_group(picks_df, config, gcs_fs, inv_cache=None):
    """
    Process all picks for a single station (same mseed_3c).

    Load the stream once, then cut waveforms for each event-station pair.
    One record per event-station with both P and S phase info, plus all
    picks that fall within the time window (for HDF5 array attrs).
    """
    region = config["region"]
    records = []

    mseed_3c = picks_df.iloc[0]["mseed_3c"]
    network = picks_df.iloc[0]["network"]
    station = picks_df.iloc[0]["station"]

    # Load and cache the stream for this station
    stream_3c = load_and_process_stream(
        mseed_3c, network, station, region, config["sampling_rate"], gcs_fs, inv_cache
    )
    if stream_3c is None:
        return records

    # Group by event_id - one record per event-station pair
    for event_id, event_picks in picks_df.groupby("event_id"):
        p_picks = event_picks[event_picks["phase_type"] == "P"]
        s_picks = event_picks[event_picks["phase_type"] == "S"]

        pick = event_picks.iloc[0]
        begin_time = pick["begin_time"]
        end_time = pick["end_time"]

        # Cut waveform from cached stream
        tmp = stream_3c.slice(
            obspy.UTCDateTime(begin_time),
            obspy.UTCDateTime(end_time),
            keep_empty_traces=False,
            nearest_sample=True,
        )

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

        # Event time index (sample index of origin time relative to begin_time)
        event_time_index = None
        if pd.notna(pick.get("event_time")):
            event_time_index = int((pick["event_time"] - begin_time).total_seconds() * config["sampling_rate"])

        # Build record
        record = {
            # Event info
            "event_id": event_id,
            "event_time": pick["event_time"].strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "event_latitude": float(pick["event_latitude"]) if pd.notna(pick.get("event_latitude")) and pick["event_latitude"] != "" else None,
            "event_longitude": float(pick["event_longitude"]) if pd.notna(pick.get("event_longitude")) and pick["event_longitude"] != "" else None,
            "event_depth_km": float(pick["event_depth_km"]) if pd.notna(pick.get("event_depth_km")) and pick["event_depth_km"] != "" else None,
            "event_magnitude": float(pick["event_magnitude"]) if pd.notna(pick.get("event_magnitude")) and pick["event_magnitude"] != "" else None,
            "event_magnitude_type": pick.get("event_magnitude_type") or None,
            "event_time_index": event_time_index,

            # Station info
            "network": network,
            "station": station,
            "location": pick["location"],
            "instrument": pick["instrument"],
            "station_latitude": float(pick["station_latitude"]) if pd.notna(pick.get("station_latitude")) else None,
            "station_longitude": float(pick["station_longitude"]) if pd.notna(pick.get("station_longitude")) else None,
            "station_elevation_m": float(pick["station_elevation_m"]) if pd.notna(pick.get("station_elevation_m")) else None,
            "station_depth_km": float(pick["station_depth_km"]) if pd.notna(pick.get("station_depth_km")) else None,

            # Waveform
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

        # Optional float fields
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

        if "quality" in pick.index and pd.notna(pick["quality"]) and pick["quality"] != "":
            record["fm_quality"] = str(pick["quality"])

        # All picks in this time window for this station (multi-event array attrs)
        picks_in_window = picks_df[
            (picks_df["phase_time"] >= begin_time) & (picks_df["phase_time"] <= end_time)
        ].sort_values("phase_time")

        if len(picks_in_window) > 0:
            record["all_event_ids"] = picks_in_window["event_id"].values
            record["all_phase_types"] = picks_in_window["phase_type"].values
            record["all_phase_times"] = picks_in_window["phase_time"].apply(
                lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f")
            ).values
            record["all_phase_indices"] = picks_in_window["phase_index"].values
            record["all_phase_scores"] = picks_in_window["phase_score"].values
            if "phase_polarity" in picks_in_window.columns:
                record["all_phase_polarities"] = picks_in_window["phase_polarity"].values
            if "phase_remark" in picks_in_window.columns:
                record["all_phase_remarks"] = picks_in_window["phase_remark"].values

        records.append(record)

    # Clear stream to free memory
    stream_3c.clear()
    return records


def save_to_hdf5(records, output_path):
    """Save all records to an HDF5 file."""
    with h5py.File(output_path, "w") as fp:
        for record in records:
            event_id = record["event_id"]
            station_id = f"{record['network']}.{record['station']}.{record['location']}.{record['instrument']}"

            # Create event group (once per event)
            if event_id not in fp:
                gp = fp.create_group(event_id)
                for key in ["event_time", "event_latitude", "event_longitude",
                           "event_depth_km", "event_magnitude", "event_magnitude_type",
                           "event_time_index", "begin_time", "end_time"]:
                    if record.get(key) is not None:
                        set_attr(gp, key, record[key])

                # Focal mechanism attrs on event group
                for key in ["strike", "dip", "rake",
                           "num_first_motions", "first_motion_misfit",
                           "num_sp_ratios", "sp_ratio_misfit",
                           "plane1_uncertainty", "plane2_uncertainty", "fm_quality"]:
                    if record.get(key) is not None:
                        set_attr(gp, key, record[key])

            # Create station dataset with waveform data
            ds = fp.create_dataset(f"{event_id}/{station_id}", data=record["waveform"])
            ds.attrs["unit"] = "micro m/s"

            # Station attrs
            for key in ["network", "station", "location", "instrument",
                       "station_latitude", "station_longitude",
                       "station_elevation_m", "station_depth_km",
                       "component", "snr",
                       "azimuth", "back_azimuth", "takeoff_angle", "distance_km"]:
                if record.get(key) is not None:
                    set_attr(ds, key, record[key])

            # Per-phase attrs (p_* and s_*)
            for prefix in ["p", "s"]:
                for suffix in ["phase_time", "phase_index", "phase_score",
                              "phase_polarity", "phase_remark"]:
                    attr_key = f"{prefix}_{suffix}"
                    if record.get(attr_key) is not None:
                        set_attr(ds, attr_key, record[attr_key])

            # All picks in window (array attrs for multi-event windows)
            attr_map = {
                "all_event_ids": "event_id",
                "all_phase_types": "phase_type",
                "all_phase_times": "phase_time",
                "all_phase_indices": "phase_index",
                "all_phase_scores": "phase_score",
                "all_phase_polarities": "phase_polarity",
                "all_phase_remarks": "phase_remark",
            }
            for record_key, attr_name in attr_map.items():
                if record.get(record_key) is not None:
                    set_attr(ds, attr_name, record[record_key])


def cut_templates(jdays, root_path, data_path, result_path, region, config, bucket, protocol, token):
    fs = fsspec.filesystem(protocol, token=token)
    gcs_fs = fsspec.filesystem("gs", token=token)

    config["nt"] = NT
    config["sampling_rate"] = SAMPLING_RATE
    config["time_before"] = TIME_BEFORE
    config["time_after"] = TIME_AFTER

    # Load station coordinates once (shared across all days)
    station_coords = load_station_csv(region, gcs_fs)
    print(f"Loaded station coordinates: {len(station_coords)} stations")

    # Cache station inventories across days to avoid re-reading XML files
    inv_cache = {}

    for jday in jdays:
        year, day = jday.split(".")
        year, day = int(year), int(day)

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
        # Step 7: Match picks with mseed files
        # ============================================================
        region_folder = "SCEDC" if region == "SC" else "NCEDC"
        with gcs_fs.open(f"quakeflow_dataset/{region_folder}/mseed/{year}/{day:03d}.txt", "r") as f:
            mseeds = [x.strip() for x in f.readlines()]

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

        if len(picks) == 0:
            print(f"No picks matched for {year:04d}/{day:03d}")
            del picks, events
            gc.collect()
            continue

        print(f"Matched picks: {len(picks)}")

        # ============================================================
        # Step 8: Process by station (mseed_3c) - cache stream per station
        # ============================================================
        station_groups = list(picks.groupby("mseed_3c"))
        ncpu = min(8, multiprocessing.cpu_count() * 2)
        print(f"Processing {len(station_groups)} stations with {ncpu} workers")

        all_records = []
        with ThreadPoolExecutor(max_workers=ncpu) as executor:
            futures = [
                executor.submit(partial(process_station_group, config=config, gcs_fs=gcs_fs, inv_cache=inv_cache), group_df)
                for _, group_df in station_groups
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                records = future.result()
                if records:
                    all_records.extend(records)

        if not all_records:
            print(f"No records for {year:04d}/{day:03d}")
            del picks, events, station_groups, all_records
            gc.collect()
            continue

        print(f"Total records: {len(all_records)}")

        # ============================================================
        # Step 9: Save to HDF5
        # ============================================================
        local_path = f"{root_path}/{result_path}/{year:04d}/{day:03d}.h5"
        save_to_hdf5(all_records, local_path)
        print(f"Saved: {local_path}")

        # Upload and cleanup
        remote_path = f"{bucket}/{result_path}/{year:04d}/{day:03d}.h5"
        fs.put(local_path, remote_path)
        print(f"Uploaded: {remote_path}")
        os.remove(local_path)
        print(f"Deleted local: {local_path}")

        # Free memory
        del all_records, picks, events, station_groups
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
    return parser.parse_args()


if __name__ == "__main__":
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
        day_nums = [int(d.strip()) for d in args.days.split(",")]
    else:
        num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
        all_days = list(range(1, num_jday + 1))
        day_nums = list(np.array_split(all_days, args.num_nodes)[args.node_rank])

    jdays = [f"{year}.{d:03d}" for d in day_nums]
    print(f"Processing {len(jdays)} days: {jdays[0]} to {jdays[-1]}")

    config = {"region": region}
    data_path = f"{region}EDC/catalog"
    result_path = f"{region}EDC/dataset"

    os.makedirs(f"{root_path}/{result_path}", exist_ok=True)
    cut_templates(jdays, root_path, data_path, result_path, region, config, bucket, protocol, token)

# %%
"""
Supplement parquet dataset with PDS event waveforms.

Fills gaps in the existing parquet dataset by fetching pre-cut event waveforms
from the NCEDC/SCEDC PDS S3 buckets. For each day:
1. Loads catalog (events.csv + phases.csv)
2. Loads existing parquet (if any) to find already-processed event-station pairs
3. Lists available event waveform files from S3
4. Processes missing event-station pairs and merges into the parquet file
"""
import argparse
import gc
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import numpy as np
import obspy
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Reuse constants and schema from cut_event_parquet
from cut_event_parquet import (
    EARTH_RADIUS_KM,
    GCS_CREDENTIALS_PATH,
    NT,
    PARQUET_SCHEMA,
    SAMPLING_RATE,
    TIME_AFTER,
    TIME_BEFORE,
    _download_inv_from_fdsn,
    _upload_inv_to_gcs,
    calc_azimuth,
    calc_distance_km,
    calc_snr,
    load_station_csv,
    prepare_picks,
)


# %%
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


# %%
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
        Record dict or None if processing fails.
    """
    network = station_picks.iloc[0]["network"]
    station = station_picks.iloc[0]["station"]

    # Reconcile location codes between trace data and inventory.
    # PDS event waveforms often use "00" (SCEDC) or "N1" (NCEDC analog telemetry)
    # while FDSN inventory uses "". Treat these as equivalent.
    EQUIV_LOCS = {"", "00", "--", "N1"}
    if inv is not None:
        inv_locs = set()
        for net_obj in inv:
            for sta_obj in net_obj:
                for ch in sta_obj:
                    inv_locs.add(ch.location_code)
        trace_locs = {tr.stats.location for tr in stream}
        # If trace and inventory locations differ but both are in the equivalent set,
        # rewrite traces to use the inventory location code
        if trace_locs != inv_locs:
            inv_equiv = inv_locs & EQUIV_LOCS
            trace_equiv = trace_locs & EQUIV_LOCS
            if inv_equiv and trace_equiv and not (inv_equiv & trace_equiv):
                target_loc = next(iter(inv_equiv))
                for tr in stream:
                    if tr.stats.location in EQUIV_LOCS:
                        tr.stats.location = target_loc

    # Resample to target sampling rate
    for tr in stream:
        if tr.stats.sampling_rate != config["sampling_rate"]:
            if tr.stats.sampling_rate % config["sampling_rate"] == 0:
                tr.decimate(int(tr.stats.sampling_rate / config["sampling_rate"]))
            else:
                tr.resample(config["sampling_rate"])

    # Remove instrument response
    if inv is not None:
        try:
            stream.remove_sensitivity(inv)
        except Exception:
            return "retry_inv"  # Signal caller to retry with fresh FDSN inventory
        stream.detrend("constant")
        try:
            stream.rotate("->ZNE", inventory=inv)
        except Exception:
            pass  # May fail if not 3C, that's ok

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

    # Need at least one component
    if not components:
        return None

    # Get P and S picks
    p_picks = station_picks[station_picks["phase_type"] == "P"]
    s_picks = station_picks[station_picks["phase_type"] == "S"]

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

    # Optional fields
    for col in ["azimuth", "back_azimuth", "takeoff_angle", "distance_km"]:
        if col in pick.index and pd.notna(pick[col]) and pick[col] != "":
            try:
                record[col] = float(pick[col])
            except (ValueError, TypeError):
                pass

    # Focal mechanism
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


# %%
def process_event_waveform(event_id, s3_paths, missing_picks, config, gcs_fs, inv_cache, existing_time_windows):
    """Process one event's waveform files, return records for missing stations.

    Args:
        event_id: Event ID string (e.g. 'nc150121')
        s3_paths: List of S3 paths for this event's waveform files
        missing_picks: DataFrame of picks for missing stations of this event
        config: Config dict
        gcs_fs: GCS filesystem for inventory access
        inv_cache: Dict caching station inventories
        existing_time_windows: Dict of event_id -> (begin_time, end_time) from existing parquet

    Returns:
        List of record dicts.
    """
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

    # Get time window (reuse from existing parquet if available)
    pick0 = missing_picks.iloc[0]
    begin_time = pick0["begin_time"]
    end_time = pick0["end_time"]
    if event_id in existing_time_windows:
        begin_time, end_time = existing_time_windows[event_id]

    # Event metadata
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

    # Group missing picks by (network, station) to process per-station
    for (network, station), sta_picks in missing_picks.groupby(["network", "station"]):
        # Select traces for this station from the event stream
        sta_stream = full_stream.select(network=network, station=station)
        if len(sta_stream) == 0:
            continue

        sta_stream = sta_stream.copy()  # Don't modify the original

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
        if record == "retry_inv":
            try:
                inv = _download_inv_from_fdsn(network, station, region)
                inv_cache[cache_key] = inv
                sta_stream = full_stream.select(network=network, station=station).copy()
                for tr in sta_stream:
                    if tr.stats.sampling_rate != config["sampling_rate"]:
                        if tr.stats.sampling_rate % config["sampling_rate"] == 0:
                            tr.decimate(int(tr.stats.sampling_rate / config["sampling_rate"]))
                        else:
                            tr.resample(config["sampling_rate"])
                record = process_event_stream(
                    sta_stream, sta_picks, begin_time, end_time, config, inv, event_info
                )
            except Exception as err:
                print(f"  FDSN retry failed for {network}.{station}: {err}")
                inv_cache[cache_key] = None
                record = None

        # Only upload inventory to GCS after remove_sensitivity succeeded
        if record is not None and record != "retry_inv":
            records.append(record)
            if inv is not None:
                try:
                    _upload_inv_to_gcs(inv, network, station, region, gcs_fs)
                except Exception:
                    pass

    return records


# %%
def supplement_day(year, day, region, config, gcs_fs, inv_cache, dry_run=False):
    """Supplement one day's parquet with event waveforms.

    Returns number of new records added.
    """
    bucket = config["bucket"]
    data_path = f"{region}EDC/catalog"
    result_path = f"{region}EDC/waveform_parquet"

    # Step 1: Load catalog
    events_file = f"{bucket}/{data_path}/{year:04d}/{day:03d}/events.csv"
    phases_file = f"{bucket}/{data_path}/{year:04d}/{day:03d}/phases.csv"

    try:
        with gcs_fs.open(events_file, "r") as fp:
            events = pd.read_csv(fp, dtype=str)
        with gcs_fs.open(phases_file, "r") as fp:
            picks = pd.read_csv(fp, dtype=str)
    except FileNotFoundError:
        print(f"  No catalog for {year:04d}/{day:03d}")
        return 0

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

    picks.fillna("", inplace=True)
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    picks["phase_score"] = pd.to_numeric(picks["phase_score"], errors="coerce")

    # Merge station coordinates
    station_coords = config["station_coords"]
    picks = picks.merge(station_coords, on=["network", "station"], how="left")

    # Prepare picks (adds begin_time, end_time, phase_index)
    picks = prepare_picks(picks, events, config)

    # Recalculate azimuth and distance
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

    # Load focal mechanisms
    try:
        fm_file = f"{bucket}/{data_path}/{year:04d}/{day:03d}/focal_mechanisms.csv"
        with gcs_fs.open(fm_file, "r") as fp:
            mechanisms = pd.read_csv(fp, dtype=str)
        fm_cols = ["event_id", "strike", "dip", "rake",
                   "num_first_motions", "first_motion_misfit",
                   "num_sp_ratios", "sp_ratio_misfit",
                   "plane1_uncertainty", "plane2_uncertainty", "quality"]
        fm_cols = [c for c in fm_cols if c in mechanisms.columns]
        picks = picks.merge(mechanisms[fm_cols], on="event_id", how="left")
    except Exception:
        pass

    catalog_pairs = set(zip(picks["event_id"], picks["network"], picks["station"]))

    # Step 2: Load existing parquet
    parquet_path = f"{bucket}/{result_path}/{year:04d}/{day:03d}.parquet"
    existing_pairs = set()
    existing_time_windows = {}
    existing_df = None

    try:
        with gcs_fs.open(parquet_path, "rb") as f:
            existing_table = pq.read_table(f)
        existing_df = existing_table.to_pandas()
        existing_pairs = set(zip(existing_df["event_id"], existing_df["network"], existing_df["station"]))

        # Extract time windows per event from existing data
        for event_id, group in existing_df.groupby("event_id"):
            row = group.iloc[0]
            existing_time_windows[event_id] = (
                pd.Timestamp(row["begin_time"]),
                pd.Timestamp(row["end_time"]),
            )
    except (FileNotFoundError, Exception):
        pass

    # Step 3: List available event waveform files from S3
    event_files = list_event_waveforms_s3(region, year, day)

    # Step 4: Find missing pairs with available event waveforms
    missing_pairs = catalog_pairs - existing_pairs
    missing_event_ids = set(p[0] for p in missing_pairs)
    available_event_ids = missing_event_ids & set(event_files.keys())

    print(f"  Catalog: {len(catalog_pairs)} pairs, {len(set(p[0] for p in catalog_pairs))} events")
    print(f"  Existing: {len(existing_pairs)} pairs")
    print(f"  Missing: {len(missing_pairs)} pairs from {len(missing_event_ids)} events")
    print(f"  Available in event_waveforms: {len(available_event_ids)} events")

    if dry_run or len(available_event_ids) == 0:
        return 0

    # Step 5: Process missing events
    new_records = []
    for event_id in tqdm(sorted(available_event_ids), desc=f"  {year:04d}/{day:03d}"):
        # Get picks for missing stations of this event
        event_picks = picks[picks["event_id"] == event_id]
        missing_sta = set(
            (n, s) for e, n, s in missing_pairs if e == event_id
        )
        mask = event_picks.apply(
            lambda r: (r["network"], r["station"]) in missing_sta, axis=1
        )
        missing_event_picks = event_picks[mask]

        if len(missing_event_picks) == 0:
            continue

        records = process_event_waveform(
            event_id, event_files[event_id],
            missing_event_picks, config, gcs_fs, inv_cache,
            existing_time_windows,
        )
        new_records.extend(records)

    if not new_records:
        print(f"  No new records produced")
        return 0

    print(f"  New records: {len(new_records)}")

    # Step 6: Merge with existing parquet
    new_df = pd.DataFrame(new_records)
    new_df["waveform"] = new_df["waveform"].apply(lambda x: x.tolist())

    if existing_df is not None:
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged_df = new_df

    merged_df.sort_values(by=["event_id", "distance_km"], inplace=True)

    for field in PARQUET_SCHEMA:
        if field.name not in merged_df.columns:
            merged_df[field.name] = None
    merged_df = merged_df[[field.name for field in PARQUET_SCHEMA]]

    table = pa.Table.from_pandas(merged_df, schema=PARQUET_SCHEMA, preserve_index=False)

    # Write locally then upload
    root_path = config.get("root_path", "./")
    os.makedirs(f"{root_path}/{result_path}/{year:04d}", exist_ok=True)
    local_path = f"{root_path}/{result_path}/{year:04d}/{day:03d}.parquet"
    pq.write_table(table, local_path, compression="zstd")

    # Step 7: Upload
    gcs_fs.put(local_path, parquet_path)
    os.remove(local_path)
    print(f"  Uploaded: {parquet_path} ({len(existing_pairs)} existing + {len(new_records)} new = {len(merged_df)} total)")

    del merged_df, table, new_df, existing_df
    gc.collect()

    return len(new_records)


# %%
def list_available_years(region):
    """List years with event waveforms available on S3."""
    s3_fs = fsspec.filesystem("s3", anon=True)
    bucket = "scedc-pds" if region == "SC" else "ncedc-pds"
    entries = s3_fs.ls(f"{bucket}/event_waveforms/")
    years = sorted(int(e.split("/")[-1]) for e in entries if e.split("/")[-1].isdigit())
    return years


def list_available_days(region, year):
    """List days with event waveforms available on S3 for a given year."""
    s3_fs = fsspec.filesystem("s3", anon=True)
    if region == "SC":
        prefix = f"scedc-pds/event_waveforms/{year}/"
        entries = s3_fs.ls(prefix)
        # Format: {year}_{jday}
        days = []
        for e in entries:
            name = e.split("/")[-1]
            if "_" in name:
                try:
                    days.append(int(name.split("_")[1]))
                except (ValueError, IndexError):
                    pass
    else:
        prefix = f"ncedc-pds/event_waveforms/{year}/"
        entries = s3_fs.ls(prefix)
        # Format: {year}.{jday}
        days = []
        for e in entries:
            name = e.split("/")[-1]
            if "." in name:
                try:
                    days.append(int(name.split(".")[1]))
                except (ValueError, IndexError):
                    pass
    return sorted(days)


def parse_args():
    parser = argparse.ArgumentParser(description="Supplement parquet dataset with PDS event waveforms")
    parser.add_argument("--region", type=str, nargs="+", default=["NC", "SC"], help="Region(s): NC SC (default: both)")
    parser.add_argument("--year", type=int, nargs="+", default=None, help="Year(s) to process (default: all available)")
    parser.add_argument("--days", type=str, default=None, help="Days to process: '5' or '1-30' or '1,5,10' (default: all available)")
    parser.add_argument("--bucket", type=str, default="quakeflow_dataset")
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument("--dry_run", action="store_true", help="Only report gaps, don't process")
    return parser.parse_args()


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


def run_region(region, years, days_str, bucket, root_path, dry_run, gcs_fs):
    """Run supplement for one region across years and days."""
    # Load station coordinates once per region
    station_coords = load_station_csv(region, gcs_fs)
    print(f"Loaded station coordinates for {region}: {len(station_coords)} stations")

    config = {
        "region": region,
        "nt": NT,
        "sampling_rate": SAMPLING_RATE,
        "time_before": TIME_BEFORE,
        "time_after": TIME_AFTER,
        "bucket": bucket,
        "root_path": root_path,
        "station_coords": station_coords,
    }

    inv_cache = {}
    total_new = 0

    for year in years:
        # Determine days for this year
        if days_str is not None:
            day_nums = parse_days(days_str)
        else:
            print(f"\nListing available days for {region} {year}...")
            day_nums = list_available_days(region, year)
            if not day_nums:
                print(f"  No event waveforms found for {region} {year}")
                continue
            print(f"  Found {len(day_nums)} days: {day_nums[0]}-{day_nums[-1]}")

        for day in day_nums:
            print(f"\n{'='*60}")
            print(f"Processing {region} {year:04d}/{day:03d}")
            print(f"{'='*60}")
            n_new = supplement_day(year, day, region, config, gcs_fs, inv_cache, dry_run=dry_run)
            total_new += n_new

    return total_new


# %%
if __name__ == "__main__":
    with open(GCS_CREDENTIALS_PATH, "r") as fp:
        token = json.load(fp)

    args = parse_args()
    gcs_fs = fsspec.filesystem("gs", token=token)

    grand_total = 0

    for region in args.region:
        # Determine years for this region
        if args.year is not None:
            years = args.year
        else:
            print(f"\nListing available years for {region}...")
            years = list_available_years(region)
            print(f"  Found {len(years)} years: {years[0]}-{years[-1]}")

        print(f"\n{'#'*60}")
        print(f"# Region: {region}, Years: {years[0]}-{years[-1]} ({len(years)} years)")
        print(f"{'#'*60}")

        n = run_region(region, years, args.days, args.bucket, args.root_path, args.dry_run, gcs_fs)
        grand_total += n

    print(f"\nDone! Grand total new records added: {grand_total}")

# %%

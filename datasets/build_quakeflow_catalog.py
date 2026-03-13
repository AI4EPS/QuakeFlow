"""
Build merged catalog CSVs from QuakeFlow yearly parquet files.

Reads the output parquet files from quakeflow_nc/quakeflow_sc (skipping the
waveform column), extracts unique events, phases, and focal mechanisms, and
writes single merged CSV files per region.

Output files per region (e.g. quakeflow_nc/):
  - events.csv:            unique events (deduplicated by event_id)
  - phases.csv:            all phase picks (one row per event-station pair)
  - focal_mechanisms.csv:  unique focal mechanisms (deduplicated by event_id)
  - stations.csv:          unique stations (deduplicated by network.station)

Usage:
    python build_quakeflow_catalog.py                          # both regions
    python build_quakeflow_catalog.py --region NC              # NC only
    python build_quakeflow_catalog.py --dry-run                # show counts only
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
BUCKET = "quakeflow_dataset"

REGION_MAP = {
    "NC": "quakeflow_nc",
    "SC": "quakeflow_sc",
}

EVENT_COLUMNS = [
    "event_id", "event_time", "event_latitude", "event_longitude",
    "event_depth_km", "event_magnitude", "event_magnitude_type",
]

PHASE_COLUMNS = [
    "event_id", "network", "station", "location", "instrument",
    "p_phase_time", "p_phase_index", "p_phase_score", "p_phase_polarity", "p_phase_remark",
    "s_phase_time", "s_phase_index", "s_phase_score", "s_phase_polarity", "s_phase_remark",
    "azimuth", "back_azimuth", "takeoff_angle", "distance_km",
]

FM_COLUMNS = [
    "event_id",
    "strike", "dip", "rake",
    "num_first_motions", "first_motion_misfit",
    "num_sp_ratios", "sp_ratio_misfit",
    "plane1_uncertainty", "plane2_uncertainty", "fm_quality",
]

STATION_COLUMNS = [
    "network", "station", "location", "instrument",
    "station_latitude", "station_longitude", "station_elevation_m", "station_depth_km",
]

# Read all metadata columns (everything except waveform)
READ_COLUMNS = list(set(EVENT_COLUMNS + PHASE_COLUMNS + FM_COLUMNS + STATION_COLUMNS))


def read_parquet_metadata(token, path):
    """Read one parquet file, returning only metadata columns (no waveform)."""
    fs = fsspec.filesystem("gs", token=token)
    try:
        with fs.open(path, "rb") as f:
            table = pq.read_table(f, columns=READ_COLUMNS)
        return table.to_pandas()
    except Exception as e:
        return None


def process_region(region, token, max_workers, dry_run, bucket=BUCKET):
    """Read all daily parquet files for a region, build merged CSVs."""
    fs = fsspec.filesystem("gs", token=token)
    dst_name = REGION_MAP[region]
    base = f"{bucket}/{dst_name}/waveform_parquet"

    # List all daily parquet files under {year}/{jday}.parquet
    files = []
    try:
        year_dirs = sorted(fs.ls(base, detail=False))
    except FileNotFoundError:
        print(f"  No parquet files found at {base}")
        return

    for year_dir in year_dirs:
        try:
            day_files = fs.ls(year_dir, detail=False)
            files.extend(sorted(f for f in day_files if f.endswith(".parquet")))
        except FileNotFoundError:
            continue

    if not files:
        print(f"  No parquet files found at {base}")
        return

    print(f"  Found {len(files)} daily parquet files")

    # Read all files in parallel (only metadata columns)
    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_parquet_metadata, token, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {region}"):
            df = future.result()
            if df is not None and len(df) > 0:
                dfs.append(df)

    if not dfs:
        print(f"  No data found")
        return

    all_data = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {len(all_data):,}")

    # Build events.csv — deduplicate by event_id
    events = all_data[EVENT_COLUMNS].drop_duplicates(subset=["event_id"]).sort_values("event_time")
    print(f"  Events: {len(events):,}")

    # Build phases.csv — one row per event-station pair
    phases = all_data[PHASE_COLUMNS].copy()
    print(f"  Phases: {len(phases):,}")

    # Build focal_mechanisms.csv — deduplicate by event_id, drop rows with all-null FM columns
    fm_value_cols = [c for c in FM_COLUMNS if c != "event_id"]
    fm = all_data[FM_COLUMNS].dropna(subset=fm_value_cols, how="all").drop_duplicates(subset=["event_id"])
    print(f"  Focal mechanisms: {len(fm):,}")

    # Build stations.csv — deduplicate by network+station
    stations = all_data[STATION_COLUMNS].drop_duplicates(subset=["network", "station"]).sort_values(["network", "station"])
    print(f"  Stations: {len(stations):,}")

    if dry_run:
        return

    # Write CSVs to GCS
    for name, df in [("events.csv", events), ("phases.csv", phases),
                     ("focal_mechanisms.csv", fm), ("stations.csv", stations)]:
        dst_path = f"{bucket}/{dst_name}/{name}"
        csv_text = df.to_csv(index=False)
        with fs.open(dst_path, "wb") as f:
            f.write(csv_text.encode("utf-8"))
        size_mb = len(csv_text) / 1024**2
        print(f"  Wrote {dst_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build merged catalog CSVs from QuakeFlow parquet files")
    parser.add_argument("--region", type=str, nargs="+", default=["NC", "SC"], choices=["NC", "SC"])
    parser.add_argument("--workers", type=int, default=8, help="Number of threads (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing")
    parser.add_argument("--bucket", type=str, default=BUCKET)
    args = parser.parse_args()

    with open(GCS_CREDENTIALS_PATH, "r") as f:
        token = json.load(f)

    for region in args.region:
        print(f"\n{'=' * 50}")
        print(f"  Region: {region}")
        print(f"{'=' * 50}")
        process_region(region, token, args.workers, args.dry_run, args.bucket)


if __name__ == "__main__":
    main()

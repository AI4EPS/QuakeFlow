# %%
"""
Repair parquet files affected by the score==0 time window bug.

In older versions of cut_event_parquet.py, prepare_picks() did not filter out
phase_score==0 picks when determining first_p_time. This caused the time window
to be anchored to the event origin time (from placeholder picks) instead of the
first real P arrival. This script:

1. Scans parquet files to identify affected days
2. Optionally re-runs cut_event_parquet on those days with --overwrite
"""
import argparse
import json
import os
import subprocess
import sys

import fsspec
import pandas as pd
import pyarrow.parquet as pq

from cut_event_parquet import (
    GCS_CREDENTIALS_PATH,
    SAMPLING_RATE,
    TIME_AFTER,
    TIME_BEFORE,
    load_station_csv,
    prepare_picks,
)


def check_day(year, day, region, gcs_fs, bucket="quakeflow_dataset"):
    """Check if a day's parquet has events affected by the score==0 window bug.

    Returns:
        List of (event_id, shift_samples) for affected events, or empty list.
    """
    data_path = f"{region}EDC/catalog"
    result_path = f"{region}EDC/waveform_parquet"

    # Load catalog
    try:
        with gcs_fs.open(f"{bucket}/{data_path}/{year:04d}/{day:03d}/phases.csv", "r") as f:
            picks = pd.read_csv(f, dtype=str)
        with gcs_fs.open(f"{bucket}/{data_path}/{year:04d}/{day:03d}/events.csv", "r") as f:
            events = pd.read_csv(f, dtype=str)
    except FileNotFoundError:
        return []

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

    # Check catalog: are there events where score==0 picks shift the window?
    p_picks = picks[picks["phase_type"] == "P"]
    affected_events = []
    for event_id, grp in p_picks.groupby("event_id"):
        first_all = grp["phase_time"].min()
        pos_picks = grp[grp["phase_score"] > 0]
        if len(pos_picks) == 0:
            continue
        first_pos = pos_picks["phase_time"].min()
        if first_all < first_pos:
            shift = (first_pos - first_all).total_seconds() * SAMPLING_RATE
            affected_events.append((event_id, int(shift)))

    if not affected_events:
        return []

    # Verify against actual parquet (maybe it was already regenerated)
    parquet_path = f"{bucket}/{result_path}/{year:04d}/{day:03d}.parquet"
    try:
        with gcs_fs.open(parquet_path, "rb") as f:
            df = pq.read_table(f, columns=["event_id", "begin_time"]).to_pandas()
    except FileNotFoundError:
        return []

    # Recalculate correct begin_time using fixed prepare_picks
    config = {
        "time_before": TIME_BEFORE,
        "time_after": TIME_AFTER,
        "sampling_rate": SAMPLING_RATE,
    }
    fixed_picks = prepare_picks(picks.copy(), events.copy(), config)

    # Compare begin_time per event
    confirmed = []
    for event_id, shift in affected_events:
        parquet_rows = df[df["event_id"] == event_id]
        if len(parquet_rows) == 0:
            continue
        parquet_begin = pd.Timestamp(parquet_rows.iloc[0]["begin_time"])

        fixed_rows = fixed_picks[fixed_picks["event_id"] == event_id]
        if len(fixed_rows) == 0:
            continue
        fixed_begin = fixed_rows.iloc[0]["begin_time"]

        if abs((parquet_begin - fixed_begin).total_seconds()) > 0.005:  # >0.5ms difference
            confirmed.append((event_id, shift))

    return confirmed


def scan_region_year(region, year, gcs_fs, bucket="quakeflow_dataset", day_filter=None):
    """Scan days in a year for the score==0 window bug.

    Args:
        day_filter: Optional list of specific days to check. If None, scan all.

    Returns:
        Dict of {day: [(event_id, shift_samples), ...]} for affected days.
    """
    result_path = f"{region}EDC/waveform_parquet"

    if day_filter is not None:
        days = sorted(day_filter)
    else:
        # List available parquet files for this year
        try:
            files = gcs_fs.ls(f"{bucket}/{result_path}/{year:04d}/")
        except FileNotFoundError:
            return {}

        days = []
        for f in files:
            name = f.split("/")[-1]
            if name.endswith(".parquet"):
                try:
                    days.append(int(name.replace(".parquet", "")))
                except ValueError:
                    pass

    affected = {}
    for day in sorted(days):
        events = check_day(year, day, region, gcs_fs, bucket)
        if events:
            affected[day] = events
            n_events = len(events)
            max_shift = max(s for _, s in events)
            print(f"  {year:04d}/{day:03d}: {n_events} affected events (max shift: {max_shift} samples)")

    return affected


def parse_args():
    parser = argparse.ArgumentParser(description="Repair parquet files with score==0 window bug")
    parser.add_argument("--region", type=str, nargs="+", default=["NC"], help="Region(s): NC SC")
    parser.add_argument("--year", type=int, nargs="+", required=True, help="Year(s) to scan")
    parser.add_argument("--bucket", type=str, default="quakeflow_dataset")
    parser.add_argument("--days", type=str, default=None, help="Specific days to check: '201' or '1,5,10' or '1-30'")
    parser.add_argument("--fix", action="store_true", help="Re-run cut_event_parquet on affected days")
    parser.add_argument("--root_path", type=str, default="./", help="Local root path for cut_event_parquet")
    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = parse_args()

    with open(GCS_CREDENTIALS_PATH, "r") as fp:
        token = json.load(fp)
    gcs_fs = fsspec.filesystem("gs", token=token)

    # Parse --days filter
    day_filter = None
    if args.days is not None:
        from supplement_event_waveforms import parse_days
        day_filter = parse_days(args.days)

    all_affected = {}  # (region, year) -> {day: [...]}
    grand_total_days = 0
    grand_total_events = 0

    for region in args.region:
        for year in args.year:
            print(f"\nScanning {region} {year}...")
            affected = scan_region_year(region, year, gcs_fs, args.bucket, day_filter=day_filter)
            if not affected:
                print(f"  No affected days found")
                continue

            all_affected[(region, year)] = affected
            total_events = sum(len(v) for v in affected.values())
            grand_total_days += len(affected)
            grand_total_events += total_events
            print(f"  Total: {len(affected)} days, {total_events} events affected")

            days_str = ",".join(str(d) for d in sorted(affected.keys()))
            print(f"  Step 1: python cut_event_parquet.py --region {region} --year {year} --days {days_str} --overwrite")
            print(f"  Step 2: python supplement_event_waveforms.py --region {region} --year {year} --days {days_str}")

            # Fix immediately per year if --fix is set
            if args.fix:
                # Step 1: Re-cut from continuous data
                cmd_cut = [
                    sys.executable, "cut_event_parquet.py",
                    "--region", region,
                    "--year", str(year),
                    "--days", days_str,
                    "--overwrite",
                    "--root_path", args.root_path,
                    "--bucket", args.bucket,
                ]
                print(f"\n  Step 1 - Re-cut: {' '.join(cmd_cut)}")
                subprocess.run(cmd_cut, check=True)

                # Step 2: Supplement with event waveforms
                cmd_supp = [
                    sys.executable, "supplement_event_waveforms.py",
                    "--region", region,
                    "--year", str(year),
                    "--days", days_str,
                    "--root_path", args.root_path,
                    "--bucket", args.bucket,
                ]
                print(f"\n  Step 2 - Supplement: {' '.join(cmd_supp)}")
                subprocess.run(cmd_supp, check=True)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    if not all_affected:
        print("No affected parquet files found.")
        sys.exit(0)

    print(f"\nGrand total: {grand_total_days} days, {grand_total_events} events across {len(all_affected)} region-years")
    for (region, year), affected in all_affected.items():
        days_str = ",".join(str(d) for d in sorted(affected.keys()))
        total_events = sum(len(v) for v in affected.values())
        print(f"  {region} {year}: {len(affected)} days, {total_events} events — Days: {days_str}")

# %%

"""
Run cut_event_parquet.py --recheck for each year (one job per year).

Scans GCS waveform_parquet to find available years, then runs recheck
on each year sequentially. Useful for fixing phase_score==0 records,
patching metadata, and supplementing missing events across all years.

Usage:
    python submit_fix.py --regions NC SC                    # Both regions, all years
    python submit_fix.py --regions NC --start_year 2020     # NC from 2020 onward
    python submit_fix.py --regions SC --year 2025           # SC specific year
    python submit_fix.py --regions NC --dry_run             # Preview only
"""
import argparse
import json
import os
import subprocess
import sys

import fsspec

BUCKET = "quakeflow_dataset"
CREDENTIALS = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cut_event_parquet.py")


def get_available_years(region, fs):
    """List years with existing waveform_parquet on GCS."""
    path = f"{BUCKET}/{region}EDC/waveform_parquet/"
    try:
        entries = fs.ls(path)
        years = sorted(int(e.split("/")[-1]) for e in entries if e.split("/")[-1].isdigit())
        return years
    except FileNotFoundError:
        return []


def main():
    parser = argparse.ArgumentParser(description="Run cut_event_parquet --recheck for each year")
    parser.add_argument("--regions", nargs="+", default=["NC", "SC"], choices=["NC", "SC"])
    parser.add_argument("--start_year", type=int, default=None, help="Start year (default: earliest available)")
    parser.add_argument("--end_year", type=int, default=None, help="End year (default: latest available)")
    parser.add_argument("--year", type=int, nargs="+", help="Specific year(s) to process")
    parser.add_argument("--days", type=str, help="Days: '5' or '1-30' or '1,5,10'")
    parser.add_argument("--dry_run", action="store_true", help="Preview commands without running")
    args = parser.parse_args()

    with open(CREDENTIALS) as f:
        token = json.load(f)
    fs = fsspec.filesystem("gs", token=token)

    for region in args.regions:
        available_years = get_available_years(region, fs)
        if not available_years:
            print(f"{region}: no waveform_parquet found")
            continue

        if args.year:
            years = [y for y in args.year if y in available_years]
        else:
            start = args.start_year or available_years[0]
            end = args.end_year or available_years[-1]
            years = [y for y in available_years if start <= y <= end]

        print(f"{region}: {len(years)} years ({years[0]}-{years[-1]})")

        for year in years:
            cmd = [sys.executable, SCRIPT, "--region", region, "--year", str(year), "--recheck"]
            if args.days:
                cmd.extend(["--days", args.days])

            if args.dry_run:
                print(f"  [DRY RUN] {' '.join(cmd)}")
            else:
                print(f"\n{'='*60}")
                print(f"  {region} {year}")
                print(f"{'='*60}")
                subprocess.run(cmd)


if __name__ == "__main__":
    main()

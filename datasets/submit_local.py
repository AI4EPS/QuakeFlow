"""
Run cut_event jobs locally (for days that need more memory than cloud spots offer).

Usage:
    python submit_local.py --year 1982 --regions SC                    # Run specific year
    python submit_local.py --year 1982 --regions SC --days 1-30        # Specific days
    python submit_local.py --year 1982 --regions SC --dry_run          # Preview only
    python submit_local.py --script supplement_event_waveforms.py ...  # Custom script
"""
import argparse
import json
import os
import subprocess
import sys

import fsspec

BUCKET = "quakeflow_dataset"
CREDENTIALS = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")

SCRIPT_MAP = {
    "parquet": "cut_event_parquet.py",
    "hdf5": "cut_event_hdf5.py",
    "supplement": "supplement_event_waveforms.py",
}

MARKERS_DIR = {
    "parquet": "done_parquet",
    "hdf5": "done_h5",
}


def fetch_available_days(region, fs):
    folder = f"{region}EDC"
    available = {}
    try:
        year_dirs = fs.ls(f"{BUCKET}/{folder}/mseed/")
        for year_dir in year_dirs:
            try:
                year = int(year_dir.split("/")[-1])
            except ValueError:
                continue
            days = sorted(int(f.split("/")[-1].replace(".txt", ""))
                          for f in fs.ls(year_dir) if f.endswith(".txt"))
            if days:
                available[year] = days
    except Exception:
        pass
    return available


def fetch_done_days(region, fs, fmt):
    folder = f"{region}EDC"
    done = {}
    if fmt not in MARKERS_DIR:
        return done
    try:
        for f in fs.find(f"{BUCKET}/{folder}/{MARKERS_DIR[fmt]}/"):
            if f.endswith(".done"):
                parts = f.split("/")
                year = int(parts[-2])
                day = int(parts[-1].replace(".done", ""))
                done.setdefault(year, set()).add(day)
    except Exception:
        pass
    return done


def parse_days(days_str):
    days = []
    for part in days_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            days.extend(range(int(start), int(end) + 1))
        else:
            days.append(int(part))
    return sorted(set(days))


def main():
    parser = argparse.ArgumentParser(description="Run cut_event jobs locally")
    parser.add_argument("--regions", nargs="+", default=["NC", "SC"], choices=["NC", "SC"])
    parser.add_argument("--start_year", type=int, default=1990)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--year", type=int, nargs="+", help="Year(s) to process (default: start_year to end_year)")
    parser.add_argument("--days", type=str, help="Days: '5' or '1-30' or '1,5,10' (default: all available)")
    parser.add_argument("--format", choices=["hdf5", "parquet", "supplement"], default="parquet")
    parser.add_argument("--script", type=str, help="Custom script path (overrides --format)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--extra_args", type=str, default="", help="Extra args to pass to the script")
    args = parser.parse_args()

    years = args.year or list(range(args.start_year, args.end_year + 1))

    script = args.script or SCRIPT_MAP[args.format]
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        sys.exit(1)

    with open(CREDENTIALS) as f:
        token = json.load(f)
    fs = fsspec.filesystem("gs", token=token)

    for region in args.regions:
        if args.days:
            day_nums = parse_days(args.days)
            for year in years:
                days_arg = ",".join(str(d) for d in day_nums)
                cmd = [sys.executable, script_path, "--region", region, "--year", str(year), "--days", days_arg]
                if args.overwrite and args.format != "supplement":
                    cmd.append("--overwrite")
                if args.extra_args:
                    cmd.extend(args.extra_args.split())
                print(f"{'[DRY RUN] ' if args.dry_run else ''}Running: {' '.join(cmd)}")
                if not args.dry_run:
                    subprocess.run(cmd)
        else:
            available_all = fetch_available_days(region, fs)
            done_all = fetch_done_days(region, fs, args.format)
            for year in years:
                available = available_all.get(year, [])
                done = done_all.get(year, set())
                remaining = [d for d in available if d not in done] if not args.overwrite else available
                if not remaining:
                    print(f"{region} {year}: no remaining days")
                    continue
                days_arg = ",".join(str(d) for d in remaining)
                cmd = [sys.executable, script_path, "--region", region, "--year", str(year), "--days", days_arg]
                if args.overwrite and args.format != "supplement":
                    cmd.append("--overwrite")
                if args.extra_args:
                    cmd.extend(args.extra_args.split())
                print(f"{'[DRY RUN] ' if args.dry_run else ''}{region} {year}: {len(remaining)} days")
                print(f"  {' '.join(cmd)}")
                if not args.dry_run:
                    subprocess.run(cmd)


if __name__ == "__main__":
    main()

"""
Submit cut_event_parquet.py --recheck jobs via sky launch (one job per year).

Scans GCS waveform_parquet to find available years, generates YAML files,
and launches sky clusters. One cluster per region-year. Monitors and
relaunches preempted jobs automatically.

Usage:
    python submit_fix.py --dry_run                          # Preview jobs
    python submit_fix.py --regions NC SC                    # Both regions, all years
    python submit_fix.py --regions NC --start_year 2020     # NC from 2020 onward
    python submit_fix.py --regions SC --year 2025           # SC specific year
    python submit_fix.py --max_concurrent 10                # Limit parallelism
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime

import fsspec

BUCKET = "quakeflow_dataset"
CREDENTIALS = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_template.yaml")

MAX_RETRIES = 5
POLL_INTERVAL = 60


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def get_available_years(region, fs):
    """List years with existing waveform_parquet on GCS."""
    path = f"{BUCKET}/{region}EDC/waveform_parquet/"
    try:
        entries = fs.ls(path)
        return sorted(int(e.split("/")[-1]) for e in entries if e.split("/")[-1].isdigit())
    except FileNotFoundError:
        return []


def generate_yaml(name, region, year, days_arg=None):
    """Generate a sky YAML for one recheck job."""
    cmd = f"python cut_event_parquet.py --region {region} --year {year} --recheck"
    if days_arg:
        cmd += f' --days "{days_arg}"'

    with open(TEMPLATE) as f:
        content = f.read()
    content = content.replace("{cmd}", cmd)

    os.makedirs(JOBS_DIR, exist_ok=True)
    path = os.path.join(JOBS_DIR, f"{name}.yaml")
    with open(path, "w") as f:
        f.write(content)
    return path


def get_active_clusters():
    result = subprocess.run("sky status --no-show-managed-jobs 2>/dev/null",
                            shell=True, capture_output=True, text=True)
    active = set()
    for line in result.stdout.split("\n"):
        parts = line.split()
        if len(parts) >= 2 and {"UP", "INIT"} & set(parts[1:]):
            active.add(parts[0])
    return active


def sky_launch(yaml_path, name):
    os.system(f"sky launch {yaml_path} -c {name} -y -d --idle-minutes-to-autostop 10 --down")


def monitor_loop(jobs, max_concurrent):
    retries = {}
    completed = set()
    total_jobs = len(jobs)

    while True:
        active = get_active_clusters()
        job_names = {j["name"] for j in jobs}
        # Jobs whose clusters are no longer active are either done or failed
        for job in jobs:
            name = job["name"]
            if name in completed:
                continue
            if name not in active and retries.get(name, 0) > 0:
                # Was launched before but cluster is gone — assume done or will relaunch
                completed.add(name)

        pending = [j for j in jobs if j["name"] not in completed]

        print(f"\n[{timestamp()}] Jobs: {len(completed)}/{total_jobs} complete, "
              f"{len(pending)} pending, {len(active & job_names)} running")

        if not pending:
            break

        launched = 0
        for job in pending:
            name = job["name"]
            if name in active:
                print(f"  {name}: running")
            elif retries.get(name, 0) >= MAX_RETRIES:
                print(f"  {name}: max retries reached, skipping")
                completed.add(name)
            elif len(active) + launched < max_concurrent:
                retries[name] = retries.get(name, 0) + 1
                attempt = retries[name]
                label = "Launching" if attempt == 1 else f"Relaunching (attempt {attempt}/{MAX_RETRIES})"
                print(f"  {name}: {label}")
                sky_launch(job["yaml"], name)
                launched += 1
            else:
                print(f"  {name}: waiting (max concurrent reached)")

        time.sleep(POLL_INTERVAL)

    print(f"\n[{timestamp()}] All done. {len(completed)} jobs completed.")


def main():
    parser = argparse.ArgumentParser(description="Submit cut_event_parquet --recheck via sky (one job per year)")
    parser.add_argument("--regions", nargs="+", default=["NC", "SC"], choices=["NC", "SC"])
    parser.add_argument("--start_year", type=int, default=None, help="Start year (default: earliest available)")
    parser.add_argument("--end_year", type=int, default=None, help="End year (default: latest available)")
    parser.add_argument("--year", type=int, nargs="+", help="Specific year(s) to process")
    parser.add_argument("--days", type=str, help="Days: '5' or '1-30' or '1,5,10'")
    parser.add_argument("--max_concurrent", type=int, default=64, help="Max concurrent sky clusters")
    parser.add_argument("--dry_run", action="store_true", help="Preview jobs without launching")
    args = parser.parse_args()

    with open(CREDENTIALS) as f:
        token = json.load(f)
    fs = fsspec.filesystem("gs", token=token)

    # Build job list
    jobs = []
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
            name = f"fix-{region.lower()}-{year}"
            yaml = generate_yaml(name, region, year, args.days)
            jobs.append({"name": name, "yaml": yaml, "region": region, "year": year})

    print(f"\nTotal jobs: {len(jobs)}")
    for job in jobs:
        print(f"  {job['name']}")

    if args.dry_run or not jobs:
        return

    monitor_loop(jobs, args.max_concurrent)


if __name__ == "__main__":
    main()

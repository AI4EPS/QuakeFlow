"""
Submit cut_event jobs via sky launch with YAML files.

Generates YAML files from a template, launches clusters, and monitors them.
Relaunches preempted/failed clusters automatically.
cut_event_{parquet,hdf5}.py skip already-processed days, so relaunches are safe.

Usage:
    python submit.py --dry_run                              # Preview jobs + generate YAMLs
    python submit.py --year 2024 --regions SC               # Launch + monitor
    python submit.py --max_concurrent 5 --max_jobs 2        # Limit parallelism
"""
import argparse
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime

import fsspec

BUCKET = "quakeflow_dataset"
CREDENTIALS = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_template.yaml")

SCRIPT_MAP = {
    "hdf5": "cut_event_hdf5.py",
    "parquet": "cut_event_parquet.py",
}

MARKERS_DIR = {
    "parquet": "done_parquet",
    "hdf5": "done_h5",
}

# Estimated files per day (for auto batch sizing)
FILES_PER_DAY = {
    "NC": {1990: 18, 1995: 322, 2000: 189, 2005: 736, 2010: 794, 2015: 863, 2020: 1236, 2022: 1318},
    "SC": {2000: 103, 2005: 175, 2010: 372, 2015: 501, 2020: 1175, 2022: 1198},
}

MAX_RETRIES = 5
POLL_INTERVAL = 60


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def fetch_available_days(region, fs):
    """Fetch available mseed days for a region. Returns {year: sorted_list_of_days}."""
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
    """Fetch done marker days for a region. Returns {year: set_of_days}."""
    folder = f"{region}EDC"
    done = {}
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


def auto_batch_size(region, year, target_files=300_000):
    data = FILES_PER_DAY.get(region, FILES_PER_DAY["NC"])
    nearest = min(data.keys(), key=lambda y: abs(y - year))
    files_per_day = data[nearest]
    return max(30, min(366, round(target_files / max(1, files_per_day))))


def batch(items, size):
    if not items:
        return []
    num_chunks = math.ceil(len(items) / size)
    chunk_size = math.ceil(len(items) / num_chunks)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def generate_yaml(name, region, year, days, fmt, overwrite=False):
    days_arg = ",".join(str(d) for d in days)
    cmd = f'python {SCRIPT_MAP[fmt]} --region {region} --year {year} --days "{days_arg}"'
    if overwrite:
        cmd += " --overwrite"

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


def monitor_loop(jobs, max_concurrent, fs, fmt, overwrite=False):
    retries = {}
    completed = set()

    total_jobs = len(jobs)
    jobs_by_name = {job["name"]: job for job in jobs}

    while True:
        # 1. Bulk-fetch done markers per region (one GCS call each)
        done_by_region = {}
        active_regions = {job["region"] for name, job in jobs_by_name.items() if name not in completed}
        for region in active_regions:
            done_by_region[region] = fetch_done_days(region, fs, fmt)

        # 2. Count complete days and identify pending jobs
        pending = {}  # name -> (job, remaining_count)
        total_days = 0
        completed_days = 0

        for name, job in jobs_by_name.items():
            if name in completed:
                continue
            done_days = done_by_region.get(job["region"], {}).get(job["year"], set())
            remaining = sum(1 for d in job["days"] if d not in done_days)
            total_days += len(job["days"])
            completed_days += len(job["days"]) - remaining

            if remaining == 0:
                completed.add(name)
            else:
                pending[name] = (job, remaining)

        # Print summary
        print(f"\n[{timestamp()}] Jobs: {len(completed)}/{total_jobs} complete, "
              f"Days: {completed_days}/{total_days} done, "
              f"{len(pending)} jobs remaining")

        if not pending:
            break

        # 3. Check which clusters are running
        running_clusters = get_active_clusters()

        # 4. For each pending job, launch if not running
        launched = 0
        for name, (job, remaining) in pending.items():
            if name in running_clusters:
                print(f"  {name}: running ({remaining} days left)")
            elif retries.get(name, 0) >= MAX_RETRIES:
                print(f"  {name}: max retries reached, skipping")
                completed.add(name)
            elif len(running_clusters) + launched < max_concurrent:
                retries[name] = retries.get(name, 0) + 1
                attempt = retries[name]
                label = "Launching" if attempt == 1 else f"Relaunching (attempt {attempt}/{MAX_RETRIES})"
                print(f"  {name}: {label} ({remaining} days left)")
                random.shuffle(job["days"])
                generate_yaml(job["name"], job["region"], job["year"], job["days"], fmt, overwrite)
                sky_launch(job["yaml"], name)
                launched += 1
            else:
                print(f"  {name}: waiting (max concurrent reached)")

        # 5. Wait before next check
        time.sleep(POLL_INTERVAL)

    print(f"\n[{timestamp()}] All done. {len(completed)} jobs completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="+", default=["NC", "SC"], choices=["NC", "SC"])
    parser.add_argument("--start_year", type=int, default=1990)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--year", type=int, help="Process single year")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_jobs", type=int)
    parser.add_argument("--max_concurrent", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--format", choices=["hdf5", "parquet"], default="parquet")
    args = parser.parse_args()

    years = [args.year] if args.year else range(args.start_year, args.end_year + 1)
    with open(CREDENTIALS) as f:
        token = json.load(f)
    fs = fsspec.filesystem("gs", token=token)

    # Build job list — bulk-fetch per region
    jobs = []
    for region in args.regions:
        print(f"Listing {region}...")
        available_all = fetch_available_days(region, fs)
        for year in years:
            available = available_all.get(year, [])
            if not available:
                continue
            batch_size = args.batch_size or auto_batch_size(region, year)
            for day_batch in batch(available, batch_size):
                name = f"{args.format}-{region.lower()}-{year}-{day_batch[0]:03d}-{day_batch[-1]:03d}"
                jobs.append({"name": name, "yaml": None, "region": region, "year": year, "days": day_batch})

    if args.max_jobs:
        jobs = jobs[:args.max_jobs]

    # Generate YAMLs
    for job in jobs:
        job["yaml"] = generate_yaml(job["name"], job["region"], job["year"], job["days"], args.format, args.overwrite)

    print(f"Total jobs: {len(jobs)} (format={args.format})")
    for job in jobs:
        print(f"  {job['name']}: {len(job['days'])} days")

    if args.dry_run or not jobs:
        return

    monitor_loop(jobs, args.max_concurrent, fs, args.format, args.overwrite)


if __name__ == "__main__":
    main()

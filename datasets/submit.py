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


def ts():
    return datetime.now().strftime("%H:%M:%S")


def bulk_list(region, fs, fmt):
    """Fetch all available and done days for a region.

    Returns (available, done) where:
        available = {year: sorted_list_of_days}
        done = {year: set_of_days}
    """
    folder = f"{'SC' if region == 'SC' else 'NC'}EDC"

    # Available: list year dirs first, then ls each year
    available = {}
    try:
        year_dirs = fs.ls(f"{BUCKET}/{folder}/mseed/")
        for yd in year_dirs:
            try:
                year = int(yd.split("/")[-1])
            except ValueError:
                continue
            days = sorted(int(f.split("/")[-1].replace(".txt", ""))
                          for f in fs.ls(yd) if f.endswith(".txt"))
            if days:
                available[year] = days
    except Exception:
        pass

    # Done markers: small directory, find() is fast
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

    return available, done


def get_done_days(region, year, fs, fmt):
    folder = f"{'SC' if region == 'SC' else 'NC'}EDC"
    try:
        return {int(f.split("/")[-1].replace(".done", ""))
                for f in fs.ls(f"{BUCKET}/{folder}/{MARKERS_DIR[fmt]}/{year:04d}/")
                if f.endswith(".done")}
    except Exception:
        return set()


def auto_batch_size(region, year, target_files=300_000):
    data = FILES_PER_DAY.get(region, FILES_PER_DAY["NC"])
    nearest = min(data.keys(), key=lambda y: abs(y - year))
    fpd = data[nearest]
    return max(30, min(366, round(target_files / max(1, fpd))))


def batch(items, size):
    if not items:
        return []
    n = math.ceil(len(items) / size)
    even = math.ceil(len(items) / n)
    return [items[i:i + even] for i in range(0, len(items), even)]


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


def monitor_loop(queue, active, max_concurrent, fs, fmt):
    done = set()
    retries = {}

    while queue or active:
        # Fill slots
        while queue and len(active) < max_concurrent:
            job = queue.pop(0)
            name = job["name"]
            print(f"[{ts()}] Launching {name}...")
            sky_launch(job["yaml"], name)
            active[name] = job
            time.sleep(10)

        if not active:
            break

        print(f"\n[{ts()}] Active: {len(active)}, Queued: {len(queue)}, Done: {len(done)}")
        time.sleep(POLL_INTERVAL)

        up_clusters = get_active_clusters()
        to_remove = []
        to_relaunch = []

        # Cache done_days per (region, year) to avoid duplicate GCS calls
        done_cache = {}
        for name, job in list(active.items()):
            key = (job["region"], job["year"])
            if key not in done_cache:
                done_cache[key] = get_done_days(job["region"], job["year"], fs, fmt)
            done_days = done_cache[key]
            remaining = sum(1 for d in job["days"] if d not in done_days)

            if remaining == 0:
                print(f"  {name}: completed")
                if name in up_clusters:
                    os.system(f"sky down {name} -y &")
                to_remove.append(name)
                done.add(name)
            elif name in up_clusters:
                print(f"  {name}: running ({remaining} days left)")
            elif retries.get(name, 0) < MAX_RETRIES:
                retries[name] = retries.get(name, 0) + 1
                print(f"  {name}: gone, relaunch ({retries[name]}/{MAX_RETRIES}, {remaining} days left)")
                to_relaunch.append(name)
            else:
                print(f"  {name}: max retries, giving up ({remaining} days left)")
                to_remove.append(name)

        for name in to_remove:
            active.pop(name, None)
        for name in to_relaunch:
            job = active.pop(name)
            sky_launch(job["yaml"], name)
            active[name] = job
            time.sleep(10)

    print(f"\n[{ts()}] All done. {len(done)} jobs completed.")


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

    # Build job list — bulk-fetch per region (2 GCS calls each)
    jobs = []
    for region in args.regions:
        print(f"Listing {region}...")
        available_all, done_all = bulk_list(region, fs, args.format)
        for year in years:
            available = available_all.get(year, [])
            if not available:
                continue
            done = done_all.get(year, set())
            if not args.overwrite and done >= set(available):
                continue
            bs = args.batch_size or auto_batch_size(region, year)
            for day_batch in batch(available, bs):
                if not args.overwrite and all(d in done for d in day_batch):
                    continue
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

    # If cluster already running, monitor it; otherwise queue for launch
    existing = get_active_clusters()
    queue = []
    active = {}
    for job in jobs:
        if job["name"] in existing:
            print(f"  {job['name']}: already running")
            active[job["name"]] = job
        else:
            queue.append(job)

    print(f"Resuming {len(active)}, queuing {len(queue)}")
    monitor_loop(queue, active, args.max_concurrent, fs, args.format)


if __name__ == "__main__":
    main()

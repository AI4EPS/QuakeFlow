"""
Submit cut_event jobs to SkyPilot - batched by day blocks.

Usage:
    python submit.py                              # Submit all pending HDF5 jobs
    python submit.py --dry_run                    # Preview jobs
    python submit.py --regions SC --year 2024    # Specific region/year
    python submit.py --batch_size 30             # Days per job (default: 30)
    python submit.py --format parquet            # Use parquet instead of hdf5
    python submit.py --overwrite                 # Reprocess all days (ignore existing)
"""
import argparse
import json
import math
import os
import time

import fsspec
import sky

BUCKET = "quakeflow_dataset"
CREDENTIALS = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")

SCRIPT_MAP = {
    "hdf5": ("cut_event_hdf5.py", ".h5"),
    "parquet": ("cut_event_parquet.py", ".parquet"),
}


def get_pending_days(region: str, year: int, fs, fmt: str, overwrite: bool = False) -> list:
    """Return days that have mseed data but haven't been processed yet."""
    folder = f"{'SC' if region == 'SC' else 'NC'}EDC"
    _, ext = SCRIPT_MAP[fmt]

    # Get days with available mseed data
    available = set()
    try:
        for f in fs.ls(f"{BUCKET}/{folder}/mseed/{year}/"):
            day = int(f.split("/")[-1].replace(".txt", ""))
            available.add(day)
    except Exception:
        return []

    if overwrite:
        return sorted(available)

    # Get already processed days
    done = set()
    try:
        for f in fs.glob(f"{BUCKET}/{folder}/{'waveform_h5' if ext == '.h5' else 'waveform_parquet'}/{year:04d}/*{ext}"):
            done.add(int(f.split("/")[-1].replace(ext, "")))
    except Exception:
        pass

    return sorted(d for d in available if d not in done)


def batch(items: list, size: int) -> list:
    """Split items into chunks of roughly equal size (no bigger than `size`)."""
    if not items:
        return []
    n = math.ceil(len(items) / size)
    even = math.ceil(len(items) / n)
    return [items[i:i + even] for i in range(0, len(items), even)]


def auto_batch_size(region: str, year: int, target_files: int = 300_000) -> int:
    """Scale batch size so each job processes roughly the same number of mseed files.

    Uses observed mseed file counts per day (from NCEDC/SCEDC S3 buckets) to keep
    work per job roughly constant. Batch size = target_files / files_per_day,
    clamped to [30, 366].
    """
    # Observed avg mseed files/day by year (sampled from bucket listings)
    volume = {
        "NC": {1990: 18, 1995: 322, 2000: 189, 2005: 736, 2010: 794, 2015: 863, 2020: 1236, 2022: 1318},
        "SC": {2000: 103, 2005: 175, 2010: 372, 2015: 501, 2020: 1175, 2022: 1198},
    }
    data = volume.get(region, volume["NC"])
    years_sorted = sorted(data.keys())
    vols = [data[y] for y in years_sorted]
    # Linear interpolation (clamp to boundary values outside range)
    if year <= years_sorted[0]:
        files_per_day = vols[0]
    elif year >= years_sorted[-1]:
        files_per_day = vols[-1]
    else:
        for i in range(len(years_sorted) - 1):
            if years_sorted[i] <= year <= years_sorted[i + 1]:
                t = (year - years_sorted[i]) / (years_sorted[i + 1] - years_sorted[i])
                files_per_day = vols[i] + t * (vols[i + 1] - vols[i])
                break
    return max(30, min(366, round(target_files / max(1, files_per_day))))


def make_task(region: str, year: int, days: list, fmt: str, overwrite: bool = False) -> sky.Task:
    script, _ = SCRIPT_MAP[fmt]
    days_arg = ",".join(str(d) for d in days)
    cmd = f"python {script} --region {region} --year {year} --days \"{days_arg}\""
    if overwrite:
        cmd += " --overwrite"
    task = sky.Task(
        name=f"{fmt}-{region.lower()}-{year}-{days[0]:03d}-{days[-1]:03d}",
        setup="pip install -q h5py tqdm pandas scipy 'numpy<2' fsspec gcsfs s3fs obspy pyproj pyarrow",
        run=cmd,
        workdir=".",
    )
    task.set_resources(sky.Resources(
        cloud=sky.GCP(),
        region="us-west1",
        cpus="4+",
        memory="32+",
        disk_size=20,
        use_spot=True,
    ))
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="+", default=["NC", "SC"], choices=["NC", "SC"])
    parser.add_argument("--start_year", type=int, default=1990)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--year", type=int, help="Process single year")
    parser.add_argument("--batch_size", type=int, default=None, help="Days per job (default: auto-scale by data volume)")
    parser.add_argument("--max_jobs", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess all days (ignore existing files)")
    parser.add_argument("--format", choices=["hdf5", "parquet"], default="hdf5", help="Output format")
    args = parser.parse_args()

    years = [args.year] if args.year else range(args.start_year, args.end_year + 1)
    with open(CREDENTIALS) as f:
        token = json.load(f)
    fs = fsspec.filesystem("gs", token=token)

    # Collect pending jobs (batched by days)
    jobs = []
    for region in args.regions:
        for year in years:
            pending = get_pending_days(region, year, fs, args.format, args.overwrite)
            bs = args.batch_size if args.batch_size else auto_batch_size(region, year)
            for day_batch in batch(pending, bs):
                jobs.append((region, year, day_batch))

    if args.max_jobs:
        jobs = jobs[:args.max_jobs]

    batch_mode = f"{args.batch_size} days/job" if args.batch_size else "auto"
    print(f"Total jobs: {len(jobs)} (batch={batch_mode}, format={args.format}, overwrite={args.overwrite})")

    if args.dry_run or not jobs:
        for r, y, days in jobs:
            print(f"  {r} {y} days {days[0]:03d}-{days[-1]:03d} ({len(days)} days)")
        return

    # Submit all jobs asynchronously first
    request_ids = []
    for region, year, days in jobs:
        try:
            name = f"{args.format}-{region.lower()}-{year}-{days[0]:03d}-{days[-1]:03d}"
            print(f"Submitting job {name}...")
            request_id = sky.jobs.launch(make_task(region, year, days, args.format, overwrite=args.overwrite), name=name)
            request_ids.append((request_id, region, year, days))
            print(f"  Request ID: {request_id}")
            time.sleep(30)  # Avoid overwhelming local CPU
        except Exception as e:
            print(f"Failed to submit {region} {year} days {days[0]:03d}-{days[-1]:03d}: {e}")

    print(f"\nSubmitted {len(request_ids)} requests. Use 'sky jobs queue' to check status.")

    # Wait for submissions to complete
    for request_id, region, year, days in request_ids:
        try:
            job_id, handle = sky.get(request_id)
            print(f"Confirmed {region} {year} days {days[0]:03d}-{days[-1]:03d} -> job_id={job_id}")
        except Exception as e:
            print(f"Failed {region} {year} days {days[0]:03d}-{days[-1]:03d}: {e}")


if __name__ == "__main__":
    main()

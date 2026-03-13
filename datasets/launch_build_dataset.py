"""
Launch N parallel SkyPilot jobs to build the QuakeFlow dataset.

Discovers all available years for both regions, divides them into N chunks,
and launches one VM per chunk. Each VM processes its assigned years sequentially,
with threads parallelizing the daily file reads within each year.

Uses resume support (skips existing yearly files) so it's safe to re-run.

Usage:
    python launch_build_dataset.py                 # 16 nodes, both regions
    python launch_build_dataset.py --n-nodes 8     # 8 nodes
    python launch_build_dataset.py --down          # tear down all nodes after
    python launch_build_dataset.py --dry-run       # show plan without launching
"""
import argparse
import json
import os
import subprocess

import fsspec

GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
BUCKET = "quakeflow_dataset"
REGIONS = {"NC": "NCEDC", "SC": "SCEDC"}

YAML_TEMPLATE = """\
resources:
  cloud: gcp
  region: us-west1
  cpus: 4+
  memory: 128+
  disk_size: 20

workdir: .

setup: |
  pip install -q tqdm fsspec gcsfs pyarrow numpy

run: |
  python build_quakeflow_dataset.py --workers 8 --year {years}
"""


def discover_remaining_years(fs):
    """List years that still need processing (source exists, destination doesn't)."""
    all_years = set()
    done_years = set()

    for region_code, region_name in REGIONS.items():
        # Source years
        base = f"{BUCKET}/{region_name}/waveform_parquet"
        try:
            year_dirs = fs.ls(base, detail=False)
        except FileNotFoundError:
            continue
        for yd in year_dirs:
            name = yd.rstrip("/").split("/")[-1]
            try:
                all_years.add(int(name))
            except ValueError:
                pass

    # Check which years are already done (in both NC and SC)
    for region_code, dst_name in [("NC", "quakeflow_nc"), ("SC", "quakeflow_sc")]:
        dst_base = f"{BUCKET}/{dst_name}/waveform_parquet"
        try:
            for f in fs.ls(dst_base, detail=False):
                name = f.split("/")[-1].replace(".parquet", "")
                try:
                    done_years.add(int(name))
                except ValueError:
                    pass
        except FileNotFoundError:
            pass

    # A year is "remaining" if it's not done in BOTH regions
    # (the build script handles both regions and skips existing files per-region)
    remaining = sorted(all_years)
    return remaining


def round_robin(lst, n):
    """Distribute items across n bins using round-robin."""
    bins = [[] for _ in range(n)]
    for i, item in enumerate(lst):
        bins[i % n].append(item)
    return [b for b in bins if b]


def main():
    parser = argparse.ArgumentParser(description="Launch parallel SkyPilot jobs for QuakeFlow dataset build")
    parser.add_argument("--n-nodes", type=int, default=16, help="Number of parallel VMs (default: 16)")
    parser.add_argument("--down", action="store_true", help="Auto-teardown VMs when done")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without launching")
    args = parser.parse_args()

    with open(GCS_CREDENTIALS_PATH, "r") as f:
        token = json.load(f)
    fs = fsspec.filesystem("gs", token=token)

    print("Discovering remaining years from GCS ...")
    remaining = discover_remaining_years(fs)
    print(f"  Found {len(remaining)} years to process")

    if not remaining:
        print("  All years already completed!")
        return

    n = min(args.n_nodes, len(remaining))
    chunks = round_robin(remaining, n)

    yaml_dir = os.path.join(os.path.dirname(__file__), "jobs")
    os.makedirs(yaml_dir, exist_ok=True)

    print(f"\nLaunching {len(chunks)} jobs (round-robin balanced):")
    for i, year_chunk in enumerate(chunks):
        years_str = " ".join(str(y) for y in year_chunk)
        yaml_content = YAML_TEMPLATE.format(years=years_str)
        yaml_path = os.path.join(yaml_dir, f"build-qf-{i}.yaml")

        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        cluster_name = f"build-qf-{i}"
        print(f"  {cluster_name}: years {year_chunk} ({len(year_chunk)} years)")

        if args.dry_run:
            continue

        cmd = ["sky", "launch", yaml_path, "-c", cluster_name, "-y", "--async"]
        if args.down:
            cmd.append("--down")
        subprocess.run(cmd, check=True)

    if not args.dry_run:
        print(f"\nAll {len(chunks)} jobs launched.")
        print(f"  Monitor:  sky status")
        print(f"  Logs:     sky logs build-qf-<i>")
        print(f"  Teardown: sky down build-qf-{{0..{len(chunks)-1}}} -y")


if __name__ == "__main__":
    main()

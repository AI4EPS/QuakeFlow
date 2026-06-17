"""
Launch SkyPilot nodes to run parquet_to_hdf5.py (add unit + local_depth_m to each year's
parquet and build its yearly HDF5).

Sizing is driven by the data. Peak RAM is set by the BUSIEST SINGLE DAY, not the year total,
because the converter works day-by-day and each day's waveforms decompress into a DataFrame in
RAM (a Ridgecrest-scale day is tens of GB). So each node's memory is sized from that year's
biggest daily parquet; disk is sized from the year's total (one HDF5 at a time).

Big years get their own node (one OOM-prone year must not take down neighbors); only small early
years are packed together, since dedicating a VM to a 1-2 GB year is wasteful. Nodes are
on-demand and tear down on completion (--down). Resume is per-year: any year whose
{name}/waveform_h5/{year}.h5 already exists is skipped, so re-running picks up only what's left.

Usage:
    python launch_parquet_to_hdf5.py --dry-run            # show the plan + per-node RAM/disk
    python launch_parquet_to_hdf5.py                      # launch
    python launch_parquet_to_hdf5.py --region SC --year 2019
"""
import argparse
import json
import math
import os
import subprocess

import fsspec

GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
BUCKET = "quakeflow_dataset"
REGIONS = {"NC": "NCEDC", "SC": "SCEDC"}
OWN_NODE_GB = 15          # a year with >= this much parquet gets its own node
PACK_TARGET_GB = 60       # small years are packed into bins of ~this total parquet size

YAML_TEMPLATE = """\
resources:
  cloud: gcp
  region: us-west1
  cpus: {cpus}+
  memory: {mem}+
  disk_size: {disk_gb}

workdir: .

setup: |
  pip install -q h5py tqdm pandas 'numpy<2' fsspec gcsfs obspy pyarrow

run: |
  python parquet_to_hdf5.py --region {region} --year {years} --workers {cpus}
"""


def node_ram_gb(busiest_day_gb):
    """RAM for a node whose densest day is busiest_day_gb of parquet: ~8x that day (decompressed
    waveforms + table copy + prepared buffer, with margin) + 16 GB for h5py/OS, rounded up to a
    machine tier. Floor 64 GB. SC 2019's 8 GB day -> 128 GB; ordinary years -> 64 GB."""
    need = busiest_day_gb * 8 + 16
    for tier in (64, 128, 256, 512):
        if need <= tier:
            return tier
    return 1024


def node_disk_gb(total_parquet_gb):
    """Boot disk: OS+env + one yearly HDF5 (<= parquet) + generous margin."""
    return max(80, math.ceil(total_parquet_gb * 1.6 + 40))


def discover_remaining(fs, regions, year_filter):
    """{region: [(year, total_gb, busiest_day_gb), ...]} for years whose HDF5 does not yet exist."""
    out = {}
    for code in regions:
        name = REGIONS[code]
        try:
            src = {int(d.rstrip("/").split("/")[-1])
                   for d in fs.ls(f"{BUCKET}/{name}/waveform_parquet", detail=False)
                   if d.rstrip("/").split("/")[-1].isdigit()}
        except FileNotFoundError:
            continue
        try:
            done = {int(f.split("/")[-1].replace(".h5", ""))
                    for f in fs.ls(f"{BUCKET}/{name}/waveform_h5", detail=False)
                    if f.endswith(".h5") and f.split("/")[-1].replace(".h5", "").isdigit()}
        except FileNotFoundError:
            done = set()
        sized = []
        for y in sorted(y for y in src - done if not year_filter or y in year_filter):
            sizes = [f["size"] for f in fs.ls(f"{BUCKET}/{name}/waveform_parquet/{y}", detail=True)
                     if f["name"].endswith(".parquet")]
            if sizes:
                sized.append((y, sum(sizes) / 1e9, max(sizes) / 1e9))
        if sized:
            out[code] = sized
    return out


def make_bins(year_sizes):
    """Big years -> one bin each; small years -> first-fit bins of ~PACK_TARGET_GB total.
    Each bin is [years, total_gb, busiest_day_gb]."""
    big = [([y], tot, busy) for y, tot, busy in year_sizes if tot >= OWN_NODE_GB]
    small = sorted((y for y in year_sizes if y[1] < OWN_NODE_GB), key=lambda t: -t[1])
    packed = []
    for y, tot, busy in small:
        for b in packed:
            if b[1] + tot <= PACK_TARGET_GB:
                b[0].append(y); b[1] += tot; b[2] = max(b[2], busy); break
        else:
            packed.append([[y], tot, busy])
    return [([*sorted(ys)], tot, busy) for ys, tot, busy in big + [tuple(b) for b in packed]]


def main():
    p = argparse.ArgumentParser(description="Launch SkyPilot nodes for parquet->HDF5, sized by data")
    p.add_argument("--region", nargs="+", default=["NC", "SC"], choices=["NC", "SC"])
    p.add_argument("--year", type=int, nargs="+", default=None, help="Restrict to specific year(s)")
    p.add_argument("--limit", type=int, default=None, help="Max nodes to launch this run")
    p.add_argument("--no-down", action="store_true", help="Keep nodes up after the job (default: tear down)")
    p.add_argument("--dry-run", action="store_true", help="Show the plan without launching")
    args = p.parse_args()

    with open(GCS_CREDENTIALS_PATH) as f:
        token = json.load(f)
    fs = fsspec.filesystem("gs", token=token)

    print("Discovering remaining years and sizes ...")
    remaining = discover_remaining(fs, args.region, set(args.year) if args.year else None)

    plan = []  # (cluster, region, years, total_gb, busiest_gb, mem, cpus, disk)
    for code in args.region:
        for i, (years, tot, busy) in enumerate(make_bins(remaining.get(code, []))):
            mem = node_ram_gb(busy)
            cpus = max(8, mem // 8)
            tag = str(years[0]) if len(years) == 1 else f"b{i}"
            plan.append((f"p2h5-{code.lower()}{tag}", code, years, tot, busy, mem, cpus, node_disk_gb(tot)))
    if args.limit:
        plan = plan[: args.limit]

    print(f"\n{len(plan)} node(s) for {sum(len(y) for _,_,y,*_ in plan)} year(s):")
    for cluster, code, years, tot, busy, mem, cpus, disk in plan:
        print(f"  {cluster:13} {code} {','.join(map(str,years)):20} "
              f"tot={tot:6.1f} busiest={busy:4.1f} -> mem={mem:4d} cpus={cpus:2d} disk={disk:3d}")
    if not plan:
        print("  Nothing to do.")
        return

    yaml_dir = os.path.join(os.path.dirname(__file__) or ".", "jobs")
    os.makedirs(yaml_dir, exist_ok=True)
    for cluster, code, years, tot, busy, mem, cpus, disk in plan:
        yaml_path = os.path.join(yaml_dir, f"{cluster}.yaml")
        with open(yaml_path, "w") as f:
            f.write(YAML_TEMPLATE.format(region=code, years=" ".join(map(str, years)),
                                         mem=mem, cpus=cpus, disk_gb=disk))
        if args.dry_run:
            continue
        cmd = ["sky", "launch", yaml_path, "-c", cluster, "-y", "--async"]
        if not args.no_down:
            cmd.append("--down")
        subprocess.run(cmd, check=True)

    if not args.dry_run:
        print(f"\nLaunched {len(plan)} node(s).  Monitor: sky status | sky logs <cluster>")
        print("Re-run anytime to pick up only the years still missing.")


if __name__ == "__main__":
    main()

"""
Build curated sharded QuakeFlow datasets from raw NCEDC/SCEDC parquet files on GCS.

Reads daily parquet files from NCEDC/SCEDC waveform_parquet, filters for rows
with both P and S picks (and SNR above threshold), and writes sharded parquet
files organized by year.

Output structure:
  quakeflow_{nc,sc}/waveform_parquet/{year}/
    index.parquet          <- metadata (no waveform) + shard_id, row_in_shard
    shard_000.parquet      <- all columns (~1 GB each)
    shard_001.parquet
    ...

Shard assignment is deterministic:
  shard_id = int(event_id.strip_letters()) % num_shards

So you can find any event's shard from just its event_id without the index.

Usage:
    python build_quakeflow_dataset.py --region NC --year 2020
    python build_quakeflow_dataset.py --region NC
    python build_quakeflow_dataset.py --dry-run
    python build_quakeflow_dataset.py --target-size 1.0
"""
import argparse
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
BUCKET = "quakeflow_dataset"

REGION_MAP = {
    "NC": {"src": "NCEDC", "dst": "quakeflow_nc"},
    "SC": {"src": "SCEDC", "dst": "quakeflow_sc"},
}

WAVEFORM_COLUMN = "waveform"


def extract_event_number(event_id_str):
    """Extract numeric part from event_id (e.g., 'nc12345' -> 12345)."""
    digits = re.sub(r"[^0-9]", "", event_id_str)
    return int(digits) if digits else 0


def list_parquet_by_year(fs, base_path):
    """List all parquet files grouped by year. Returns dict {year: [paths]}."""
    try:
        year_dirs = fs.ls(base_path, detail=False)
    except FileNotFoundError:
        return {}

    by_year = {}
    for year_dir in sorted(year_dirs):
        year_name = year_dir.rstrip("/").split("/")[-1]
        try:
            year_int = int(year_name)
        except ValueError:
            continue

        try:
            day_files = sorted(fs.ls(year_dir, detail=False))
        except FileNotFoundError:
            continue

        paths = [f for f in day_files if f.endswith(".parquet")]
        if paths:
            by_year[year_int] = paths

    return by_year


def read_and_filter(token, src_path, min_snr):
    """Read one daily parquet file and filter for P+S picks."""
    fs = fsspec.filesystem("gs", token=token)
    try:
        with fs.open(src_path, "rb") as f:
            table = pq.read_table(f)
    except Exception as e:
        return None, 0, str(e)

    total = table.num_rows
    if total == 0:
        return None, 0, None

    mask = pc.and_(
        pc.is_valid(table.column("p_phase_index")),
        pc.is_valid(table.column("s_phase_index")),
    )
    if min_snr > 0:
        mask = pc.and_(mask, pc.and_(
            pc.is_valid(table.column("snr")),
            pc.greater(table.column("snr"), min_snr),
        ))

    filtered = table.filter(mask)
    if filtered.num_rows == 0:
        return None, total, None

    return filtered, total, None


def estimate_compressed_size(table):
    """Estimate zstd-compressed parquet size by writing to memory."""
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.tell()


def process_year(region, year, day_paths, token, max_workers, min_snr, dst_base,
                 target_size_gb, overwrite, dry_run, bucket):
    """Read, filter, and shard all daily files for one year."""
    fs = fsspec.filesystem("gs", token=token)
    year_base = f"{dst_base}/{year}"

    # Check if already done (index.parquet exists)
    if not overwrite and not dry_run:
        try:
            if fs.exists(f"{year_base}/index.parquet"):
                print(f"    {year}: skipped (index.parquet exists)")
                return {"year": year, "total": 0, "kept": 0}
        except Exception:
            pass

    # Read and filter all daily files in parallel
    tables = []
    total_rows = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_and_filter, token, path, min_snr): path for path in day_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"    {year} read", leave=False):
            filtered, total, err = future.result()
            total_rows += total
            if err:
                errors += 1
                tqdm.write(f"      Error: {futures[future].split('/')[-1]} — {err}")
            elif filtered is not None:
                tables.append(filtered)

    if not tables:
        print(f"    {year}: {total_rows:,} rows -> 0 kept, {errors} errors")
        return {"year": year, "total": total_rows, "kept": 0}

    combined = pa.concat_tables(tables)
    kept = combined.num_rows
    pct = kept / total_rows * 100 if total_rows > 0 else 0

    # Estimate compressed size to determine number of shards
    estimated_bytes = estimate_compressed_size(combined)
    estimated_gb = estimated_bytes / 1024**3
    num_shards = max(1, round(estimated_gb / target_size_gb))
    # NC event IDs increment by 5; SC event IDs increment by 8
    if num_shards % 5 == 0:
        num_shards -= 1
    if region == "SC" and year in (2013, 2017, 2019, 2021, 2023, 2024) and num_shards % 2 == 0:
        num_shards -= 1

    event_ids = combined.column("event_id").to_pylist()
    shard_ids = np.array([extract_event_number(eid) % num_shards for eid in event_ids], dtype=np.int32)

    print(f"    {year}: {total_rows:,} -> {kept:,} kept ({pct:.1f}%), {estimated_gb:.1f} GB, {num_shards} shards, {errors} errors")

    if dry_run:
        for sid in range(num_shards):
            count = int(np.sum(shard_ids == sid))
            print(f"      [dry-run] shard_{sid:03d}: {count:,} rows")
        return {"year": year, "total": total_rows, "kept": kept}

    # Clean up existing output files
    try:
        existing = fs.ls(year_base, detail=False)
        for path in existing:
            name = path.split("/")[-1]
            if name.startswith("shard_") or name == "index.parquet":
                fs.rm(path)
    except FileNotFoundError:
        pass

    # Write shard files (all columns)
    all_columns = combined.column_names
    index_columns = [c for c in all_columns if c != WAVEFORM_COLUMN]

    for sid in range(num_shards):
        mask = shard_ids == sid
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        shard_table = combined.take(indices)
        shard_path = f"{year_base}/shard_{sid:03d}.parquet"
        with fs.open(shard_path, "wb") as f:
            pq.write_table(shard_table, f, compression="zstd")
        size_mb = fs.info(shard_path)["size"] / 1024**2
        print(f"      shard_{sid:03d}: {len(indices):,} rows, {size_mb:.0f} MB")

    # Build and write index (metadata + shard_id + row_in_shard)
    idx_table = combined.select(index_columns)
    row_in_shard = np.zeros(kept, dtype=np.int32)
    shard_counters = {}
    for i in range(kept):
        sid = int(shard_ids[i])
        row_in_shard[i] = shard_counters.get(sid, 0)
        shard_counters[sid] = shard_counters.get(sid, 0) + 1

    idx_table = idx_table.append_column("shard_id", pa.array(shard_ids))
    idx_table = idx_table.append_column("row_in_shard", pa.array(row_in_shard))

    index_path = f"{year_base}/index.parquet"
    with fs.open(index_path, "wb") as f:
        pq.write_table(idx_table, f, compression="zstd")
    size_mb = fs.info(index_path)["size"] / 1024**2
    print(f"      index: {kept:,} rows, {size_mb:.1f} MB")

    return {"year": year, "total": total_rows, "kept": kept}


def process_region(region, token, max_workers, year_filter, overwrite, dry_run, min_snr,
                   target_size_gb, bucket=BUCKET):
    """Process all years for a region."""
    fs = fsspec.filesystem("gs", token=token)
    cfg = REGION_MAP[region]
    src_base = f"{bucket}/{cfg['src']}/waveform_parquet"
    dst_base = f"{bucket}/{cfg['dst']}/waveform_parquet"

    by_year = list_parquet_by_year(fs, src_base)
    if year_filter:
        by_year = {y: paths for y, paths in by_year.items() if y in year_filter}

    if not by_year:
        print(f"  No data found for {region}")
        return

    print(f"  {len(by_year)} years, processing sequentially ...")

    grand_total = 0
    grand_kept = 0
    for year in sorted(by_year):
        result = process_year(region, year, by_year[year], token, max_workers, min_snr,
                              dst_base, target_size_gb, overwrite, dry_run, bucket)
        grand_total += result["total"]
        grand_kept += result["kept"]

    pct = (grand_kept / grand_total * 100) if grand_total > 0 else 0
    print(f"\n  {'─' * 40}")
    print(f"  {region} Summary:")
    print(f"    Total rows:  {grand_total:,}")
    print(f"    Kept rows:   {grand_kept:,} ({pct:.1f}%)")
    print(f"  {'─' * 40}")


def main():
    parser = argparse.ArgumentParser(description="Build sharded QuakeFlow datasets from NCEDC/SCEDC")
    parser.add_argument("--region", type=str, nargs="+", default=["NC", "SC"], choices=["NC", "SC"])
    parser.add_argument("--year", type=int, nargs="+", default=None, help="Process specific year(s)")
    parser.add_argument("--workers", type=int, default=8, help="Threads for daily file reads (default: 8)")
    parser.add_argument("--min-snr", type=float, default=1.1, help="Minimum SNR threshold (default: 1.1)")
    parser.add_argument("--target-size", type=float, default=1.0, help="Target shard size in GB (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing")
    parser.add_argument("--overwrite", action="store_true", help="Re-process even if output exists")
    parser.add_argument("--bucket", type=str, default=BUCKET, help="GCS bucket name")
    args = parser.parse_args()

    with open(GCS_CREDENTIALS_PATH, "r") as f:
        token = json.load(f)

    year_set = set(args.year) if args.year else None

    for region in args.region:
        print(f"\n{'=' * 50}")
        print(f"  Region: {region} ({'dry run' if args.dry_run else 'live'})")
        snr_msg = f"SNR > {args.min_snr}" if args.min_snr > 0 else "no SNR filter"
        print(f"  Filter: both P+S, {snr_msg}")
        print(f"  Target: ~{args.target_size} GB/shard")
        print(f"{'=' * 50}")
        process_region(region, token, args.workers, year_set, args.overwrite, args.dry_run,
                       args.min_snr, args.target_size, args.bucket)


if __name__ == "__main__":
    main()

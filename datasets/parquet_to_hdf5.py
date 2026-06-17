"""
Add `unit` to the parquet files and, in the same pass, build yearly HDF5 files from them.

For each region/year: read every daily parquet once, resolve `unit` from the StationXML and
rewrite the parquet with the `unit` column, and accumulate the records into one yearly HDF5
file that follows the documented waveform.h5 format
(https://ai4eps.github.io/homepage/ml4earth/seismic_event_format1/).

HDF5 layout (two levels, so location fields drop the event_/station_ prefix):
  /{event_id}                                  event group
      attrs: event_id, event_time, event_time_index, begin_time, end_time,
             latitude, longitude, depth_km, magnitude, magnitude_type,
             sampling_rate, nt, nx, source, + focal mechanism (strike, dip, ...)
  /{event_id}/{net.sta.loc.instrument}         station dataset, waveform (3, 12288) float32
      attrs: network, station, location, instrument, component, unit, snr, dt_s,
             latitude, longitude, elevation_m, local_depth_m (from StationXML <Depth>),
             distance_km, azimuth, back_azimuth, takeoff_angle,
             p_phase_* / s_phase_*,
             and the per-window pick arrays (bare names, indices recomputed to this window):
               event_id, phase_type, phase_index, phase_time, phase_score,
               phase_polarity, phase_remark

Usage:
    python parquet_to_hdf5.py --region NC --year 2020
    python parquet_to_hdf5.py --region NC --year 1984 --days 61 --dry-run   # local h5 only, no GCS writes
"""
import argparse
import json
import os
import re
import tempfile
import threading
import traceback
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import fsspec
import h5py
import numpy as np
import obspy
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from backfill_unit import load_inventory, reorder_columns
from cut_event_parquet import SAMPLING_RATE, NT, PARQUET_SCHEMA, ROW_GROUP_SIZE, waveform_unit

# Read/write daily parquet in row batches: a dense Ridgecrest day has so many records that the
# waveform list<list<float>> column exceeds Arrow's 32-bit list-offset limit (2^31 elements) if
# the whole file is materialized as one table. Reuse the cut's ROW_GROUP_SIZE so the converter's
# rewrites match the cut's row-group layout (50000 * 3 * 12288 = 1.8e9 < 2^31).
READ_BATCH = ROW_GROUP_SIZE

GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
BUCKET = "quakeflow_dataset"
REGION_MAP = {"NC": "NCEDC", "SC": "SCEDC"}
DT_S = 1.0 / SAMPLING_RATE

EVENT_ATTR_MAP = {  # event-group attr -> parquet column (location prefix dropped)
    "event_time": "event_time", "event_time_index": "event_time_index",
    "begin_time": "begin_time", "end_time": "end_time",
    "latitude": "event_latitude", "longitude": "event_longitude", "depth_km": "event_depth_km",
    "magnitude": "event_magnitude", "magnitude_type": "event_magnitude_type",
}
FOCAL_ATTRS = ["strike", "dip", "rake", "num_first_motions", "first_motion_misfit",
               "num_sp_ratios", "sp_ratio_misfit", "plane1_uncertainty", "plane2_uncertainty", "fm_quality"]
STATION_ATTR_MAP = {  # station-dataset attr -> parquet column (location prefix dropped)
    "network": "network", "station": "station", "location": "location", "instrument": "instrument",
    "component": "component", "unit": "unit", "snr": "snr",
    "latitude": "station_latitude", "longitude": "station_longitude", "elevation_m": "station_elevation_m",
    "distance_km": "distance_km", "azimuth": "azimuth", "back_azimuth": "back_azimuth", "takeoff_angle": "takeoff_angle",
}
PHASE_ATTRS = ["p_phase_time", "p_phase_index", "p_phase_score", "p_phase_polarity", "p_phase_remark",
               "s_phase_time", "s_phase_index", "s_phase_score", "s_phase_polarity", "s_phase_remark"]
STR_ARRAYS = {"event_id", "phase_type", "phase_time", "phase_polarity", "phase_remark"}
INT_ATTRS = {"event_time_index", "p_phase_index", "s_phase_index", "sampling_rate", "nt", "nx"}


def set_attr(obj, key, value):
    """Set an HDF5 attribute, skipping None/NaN/empty; force integer-typed fields to int."""
    if value is None:
        return
    if isinstance(value, float) and np.isnan(value):
        return
    if isinstance(value, str) and value == "":
        return
    if key in INT_ATTRS:
        value = int(value)
    obj.attrs[key] = value


def source_from_event_id(event_id):
    """Catalog source from the event_id prefix, e.g. 'nc73324146' -> 'NC'."""
    m = re.match(r"^[A-Za-z]+", event_id or "")
    return m.group(0).upper() if m else ""


def resolve_channel(fs, folder, net, sta, loc, instr, comp, begin_time, inv_cache, ch_cache):
    """Return (unit, local_depth_m) for a channel, both read from the StationXML (cached)."""
    key = (folder, net, sta, loc, instr)
    if key in ch_cache:
        return ch_cache[key]
    inv = load_inventory(fs, folder, net, sta, inv_cache)
    comp_char = comp[-1] if comp else "Z"
    channel = f"{instr}{comp_char}" if instr else comp_char
    try:
        t = obspy.UTCDateTime(begin_time) if begin_time else None
    except Exception:
        t = None
    tr = obspy.Trace(data=np.zeros(1, dtype=np.float32))
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = net or "", sta or "", loc or "", channel
    if t is not None:
        tr.stats.starttime = t
    unit = waveform_unit(inv, obspy.Stream([tr]))
    local_depth = None
    if inv is not None:
        try:
            local_depth = inv.get_channel_metadata(f"{net}.{sta}.{loc}.{channel}", t).get("local_depth")
        except Exception:
            local_depth = None
    ch_cache[key] = (unit, local_depth)
    return unit, local_depth


def build_station_pick_index(records):
    """station_id -> list of (event_id, phase_type, phase_time_ts, score, polarity, remark)."""
    picks = defaultdict(list)
    for rec in records:
        sid = f"{rec['network']}.{rec['station']}.{rec['location']}.{rec['instrument']}"
        for ptype, t, score, pol, rem in (
            ("P", rec.get("p_phase_time"), rec.get("p_phase_score"), rec.get("p_phase_polarity"), rec.get("p_phase_remark")),
            ("S", rec.get("s_phase_time"), rec.get("s_phase_score"), rec.get("s_phase_polarity"), rec.get("s_phase_remark")),
        ):
            if t:
                picks[sid].append((rec["event_id"], ptype, pd.Timestamp(t), score, pol, rem))
    return picks


def window_arrays(picks_for_station, begin_ts, end_ts):
    """Pick arrays for one window: phase_index recomputed relative to begin_ts."""
    rows = sorted((p for p in picks_for_station if begin_ts <= p[2] <= end_ts), key=lambda p: p[2])
    out = {k: [] for k in ("event_id", "phase_type", "phase_index", "phase_time",
                           "phase_score", "phase_polarity", "phase_remark")}
    for eid, ptype, t, score, pol, rem in rows:
        out["event_id"].append(str(eid))
        out["phase_type"].append(str(ptype))
        out["phase_index"].append(int(round((t - begin_ts).total_seconds() * SAMPLING_RATE)))
        out["phase_time"].append(t.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        out["phase_score"].append(float(score) if score is not None and not pd.isna(score) else np.nan)
        out["phase_polarity"].append("" if pol is None or (isinstance(pol, float) and pd.isna(pol)) else str(pol))
        out["phase_remark"].append("" if rem is None or (isinstance(rem, float) and pd.isna(rem)) else str(rem))
    return out


def prepare_day(fs, folder, path, inv_cache, ch_cache, dry_run, gzip_level):
    """Worker (runs in a thread pool): read one daily parquet, add unit + local_depth_m and
    rewrite it, and return per-record write instructions with the waveform already
    gzip-compressed -- so the dominant cost (compression) runs off the serial writer thread.
    zlib.compress releases the GIL, so multiple day-workers compress on different cores."""
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        have_unit = "unit" in pf.schema_arrow.names
        have_depth = "local_depth_m" in pf.schema_arrow.names
        df = pd.concat([b.to_pandas() for b in pf.iter_batches(batch_size=READ_BATCH)],
                       ignore_index=True)

    uniq = {}
    for net, sta, loc, instr, comp, bt in zip(
            df["network"], df["station"], df["location"], df["instrument"], df["component"], df["begin_time"]):
        uniq.setdefault((net or "", sta or "", loc or "", instr or ""), (comp, bt))
    for it in uniq.items():
        resolve_channel(fs, folder, *it[0], it[1][0], it[1][1], inv_cache, ch_cache)

    units, depths_row, depth_by_sid = [], [], {}
    for net, sta, loc, instr in zip(df["network"], df["station"], df["location"], df["instrument"]):
        u, d = ch_cache[(folder, net or "", sta or "", loc or "", instr or "")]
        units.append(u)
        depths_row.append(d)
        depth_by_sid[f"{net}.{sta}.{loc}.{instr}"] = d
    if not have_unit:
        df["unit"] = units
    if not have_depth:
        df["local_depth_m"] = depths_row

    if not dry_run and not (have_unit and have_depth):
        # df already carries unit/local_depth_m (set above); write back in the same row batches
        # to the canonical column order, so the waveform column never exceeds the 32-bit limit.
        out_schema = pa.schema([fld for fld in PARQUET_SCHEMA if fld.name in df.columns])
        cols = out_schema.names
        tmp = path + ".tmp"
        with fs.open(tmp, "wb") as f:
            writer = pq.ParquetWriter(f, out_schema, compression="zstd")
            for s in range(0, len(df), READ_BATCH):
                writer.write_table(pa.Table.from_pandas(
                    df.iloc[s:s + READ_BATCH][cols], schema=out_schema, preserve_index=False))
            writer.close()
        fs.mv(tmp, path)

    records = df.to_dict("records")
    picks_index = build_station_pick_index(records)
    prepared, seen = [], set()
    for rec in records:
        event_id = rec["event_id"]
        sid = f"{rec['network']}.{rec['station']}.{rec['location']}.{rec['instrument']}"
        if (event_id, sid) in seen:
            continue
        seen.add((event_id, sid))
        wf = np.ascontiguousarray(np.stack([np.asarray(c, dtype=np.float32) for c in rec["waveform"]]), dtype=np.float32)
        wbytes = zlib.compress(wf.tobytes(), gzip_level)
        eattrs = {"event_id": event_id}
        for a, col in EVENT_ATTR_MAP.items():
            eattrs[a] = rec.get(col)
        for a in FOCAL_ATTRS:
            eattrs[a] = rec.get(a)
        eattrs["sampling_rate"] = int(SAMPLING_RATE)
        eattrs["nt"] = int(NT)
        eattrs["source"] = source_from_event_id(event_id)
        sattrs = {a: rec.get(col) for a, col in STATION_ATTR_MAP.items()}
        sattrs["local_depth_m"] = depth_by_sid.get(sid)
        sattrs["dt_s"] = DT_S
        for a in PHASE_ATTRS:
            sattrs[a] = rec.get(a)
        raw = window_arrays(picks_index.get(sid, []),
                            pd.Timestamp(rec["begin_time"]), pd.Timestamp(rec["end_time"]))
        arrays = {a: (np.array(v, dtype=h5py.string_dtype()) if a in STR_ARRAYS else np.array(v))
                  for a, v in raw.items() if v}
        prepared.append((event_id, sid, eattrs, sattrs, arrays, wbytes))
    return prepared


def write_prepared(fp, prepared, gzip_level):
    """Main thread: write one day's prepared records into the HDF5 (waveform already compressed)."""
    for event_id, sid, eattrs, sattrs, arrays, wbytes in prepared:
        if event_id not in fp:
            gp = fp.create_group(event_id)
            for k, v in eattrs.items():
                set_attr(gp, k, v)
        gp = fp[event_id]
        if sid in gp:
            continue
        ds = gp.create_dataset(sid, shape=(3, NT), dtype=np.float32, chunks=(3, NT),
                               compression="gzip", compression_opts=gzip_level)
        ds.id.write_direct_chunk((0, 0), wbytes)
        for k, v in sattrs.items():
            set_attr(ds, k, v)
        for a, arr in arrays.items():   # numpy arrays already built in the worker
            ds.attrs[a] = arr


class ByteBudget:
    """Admission control by bytes. A day acquires permits ~ its in-RAM footprint before it is
    read and releases them after it is written, so total in-flight memory is bounded no matter
    how many records a single day holds. Capping day count (a plain semaphore) is not enough:
    a Ridgecrest-scale day decompresses to tens of GB, so 8 such days at once OOM a 64 GB node.
    A day larger than the whole budget is admitted alone (runs when nothing else holds permits)."""

    def __init__(self, total_bytes):
        self.total = max(1, int(total_bytes))
        self.used = 0
        self.closed = False
        self.cv = threading.Condition()

    def acquire(self, n):
        n = min(int(n), self.total)
        with self.cv:
            while not self.closed and self.used and self.used + n > self.total:
                self.cv.wait()
            self.used += n
        return n

    def release(self, n):
        with self.cv:
            self.used -= n
            self.cv.notify_all()

    def close(self):
        """Unblock every waiter so the thread pool can shut down (called on error, instead of
        hanging in shutdown() forever because the dead writer loop never releases permits)."""
        with self.cv:
            self.closed = True
            self.cv.notify_all()


# A day's peak RAM (decompressed waveforms ~3.3x parquet + the table copy during rewrite ~1x +
# the compressed prepared buffer ~1.1x). Used to convert a daily parquet size into byte permits.
DAY_RAM_FACTOR = 10


def default_budget_bytes():
    """~55% of node RAM for in-flight day data; the rest is h5py's growing file + OS."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total = 64 * 1024**3
    return int(total * 0.55)


def rss_mb():
    """Resident set size of this process in MB (Linux); -1 elsewhere."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def process_year(fs, region, year, days, dry_run, out_dir, overwrite=False, workers=8, gzip_level=4,
                 budget_bytes=None):
    folder = REGION_MAP[region]
    remote_h5 = f"{BUCKET}/{folder}/waveform_h5/{year}.h5"
    if not dry_run and not overwrite and not days and fs.exists(remote_h5):
        print(f"{region} {year}: skip (exists gs://{remote_h5})")
        return
    src_base = f"{BUCKET}/{folder}/waveform_parquet/{year}"
    listing = {f["name"]: f["size"] for f in fs.ls(src_base, detail=True) if f["name"].endswith(".parquet")}
    files = sorted(listing)
    if days:
        files = [f for f in files if int(f.split("/")[-1].split(".")[0]) in set(days)]
    if not files:
        print(f"No parquet for {region} {year}")
        return

    local_h5 = os.path.join(out_dir, f"{folder}_{year}.h5")
    inv_cache, ch_cache = {}, {}
    budget = ByteBudget(budget_bytes if budget_bytes else default_budget_bytes())
    busiest = max(listing[f] for f in files)
    print(f"{region} {year}: {len(files)} daily files -> {local_h5} ({workers} workers, gzip={gzip_level}, "
          f"RAM budget {budget.total/1e9:.0f} GB, busiest day {busiest/1e9:.1f} GB parquet)")
    # Day-workers download/resolve/rewrite-parquet/compress in parallel; the main thread serially
    # writes the pre-compressed chunks (the only step that touches the h5py handle). Admission is
    # by BYTES (ByteBudget): a day acquires permits ~ its in-RAM footprint before it is read and
    # releases them after it is written, so peak memory is bounded regardless of how dense a single
    # day is. A day larger than the whole budget runs alone.
    def prepare(path):
        cost = budget.acquire(listing[path] * DAY_RAM_FACTOR)
        if budget.closed:
            return None, cost  # an error elsewhere is tearing the run down; skip the work
        try:
            return prepare_day(fs, folder, path, inv_cache, ch_cache, dry_run, gzip_level), cost
        except BaseException:
            budget.release(cost)
            raise

    with h5py.File(local_h5, "w") as fp:
        # Bounded submission in COMPLETION order: keep ~workers+4 day-futures alive, write
        # whichever finishes first (freeing its ByteBudget permits immediately), and DROP each
        # future after writing. Two failure modes this avoids: as_completed() over all 365
        # futures retains every finished result (RAM grew ~1.2 MB/event -> OOM); a FIFO deque
        # deadlocks because the writer waits on the oldest day while later days hold the budget
        # it needs and can't be written until it is. Completion order has neither problem.
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            paths = iter(files)
            pending = set()
            for _ in range(workers + 4):
                p = next(paths, None)
                if p is None:
                    break
                pending.add(ex.submit(prepare, p))
            pbar = tqdm(total=len(files), desc=f"{region}{year}")
            done = 0
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    prepared, cost = fut.result()
                    write_prepared(fp, prepared, gzip_level)
                    budget.release(cost)
                    del prepared
                    p = next(paths, None)
                    if p is not None:
                        pending.add(ex.submit(prepare, p))
                    done += 1
                    pbar.update(1)
                    if done % 20 == 0:
                        fp.flush()
                        print(f"  [{region}{year} day {done}/{len(files)}] RSS={rss_mb()}MB "
                              f"events={len(fp)} budget_used={budget.used // 10**6}MB", flush=True)
            pbar.close()
        except BaseException:
            budget.close()  # release workers blocked in acquire() so the pool shuts down (no hang)
            raise
        finally:
            ex.shutdown(wait=True)
        for gp in fp.values():
            gp.attrs["nx"] = len(gp)

    size_gb = os.path.getsize(local_h5) / 1024**3
    print(f"  wrote {local_h5} ({size_gb:.2f} GB)", flush=True)
    if not dry_run:
        for attempt in range(1, 4):
            try:
                fs.put(local_h5, remote_h5)
                break
            except Exception as e:
                print(f"  upload attempt {attempt}/3 failed ({size_gb:.1f} GB): {e}", flush=True)
                if attempt == 3:
                    raise
        os.remove(local_h5)
        print(f"  uploaded gs://{remote_h5}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Add unit/local_depth to parquet and build yearly HDF5 from it")
    p.add_argument("--region", nargs="+", default=["NC", "SC"], choices=["NC", "SC"])
    p.add_argument("--year", type=int, nargs="+", required=True)
    p.add_argument("--days", type=int, nargs="+", default=None, help="Limit to specific day-of-year (testing)")
    p.add_argument("--dry-run", action="store_true", help="Local HDF5 only; do not rewrite parquet or upload")
    p.add_argument("--overwrite", action="store_true", help="Rebuild even if the yearly HDF5 already exists")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="Parallel day-workers within a year (download/resolve/compress; default: all cores)")
    p.add_argument("--gzip-level", type=int, default=4, help="HDF5 gzip level for waveforms (default: 4)")
    p.add_argument("--out-dir", default=tempfile.gettempdir(), help="Local dir for the HDF5 file")
    p.add_argument("--budget-gb", type=float, default=None,
                   help="In-flight RAM budget for day data (default: ~55%% of node RAM)")
    args = p.parse_args()

    with open(GCS_CREDENTIALS_PATH) as f:
        token = json.load(f)
    fs = fsspec.filesystem("gs", token=token)

    # One year at a time on this node (each yearly HDF5 is large); scale across nodes via the launcher.
    for region in args.region:
        for year in args.year:
            try:
                process_year(fs, region, year, args.days, args.dry_run, args.out_dir,
                             args.overwrite, args.workers, args.gzip_level,
                             budget_bytes=int(args.budget_gb * 1e9) if args.budget_gb else None)
            except Exception as e:
                print(f"ERROR processing {region} {year}: {e}\n{traceback.format_exc()}", flush=True)
                leftover = os.path.join(args.out_dir, f"{REGION_MAP[region]}_{year}.h5")
                if os.path.exists(leftover):
                    os.remove(leftover)  # free disk so the next year isn't blocked


if __name__ == "__main__":
    main()

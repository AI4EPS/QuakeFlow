"""
Dataset validation for NCEDC and SCEDC processing pipeline.

Usage:
    python test_dataset.py                                    # Default: all tests for NC 2024/001
    python test_dataset.py --region SC                        # All tests for SCEDC
    python test_dataset.py --test event                       # Event catalog validation only
    python test_dataset.py --test phase                       # Phase catalog validation only
    python test_dataset.py --test fm                          # Focal mechanism validation only
    python test_dataset.py --test parquet                     # Parquet file validation
    python test_dataset.py --test hdf5                        # HDF5 file validation

Test options:
    all      - Run all tests (default)
    event    - Validate event catalog: counts, IDs, and attribute values
    phase    - Validate phase catalog: counts, keys, and attribute values
    fm       - Validate focal mechanism catalog: counts, IDs, and attribute values
    parquet  - Validate parquet file vs CSV, generate moveout and zoom plots
    hdf5     - Validate HDF5 waveform file
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from io import StringIO

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
DEFAULT_BUCKET = "quakeflow_dataset"
OUTPUT_DIR = "test_results"


# =============================================================================
# FILESYSTEM HELPERS
# =============================================================================

def get_gcs_fs():
    """Get GCS filesystem with credentials."""
    with open(GCS_CREDENTIALS_PATH, "r") as f:
        token = json.load(f)
    return fsspec.filesystem("gs", token=token)


def get_s3_fs():
    """Get S3 filesystem for raw data access."""
    return fsspec.filesystem("s3", anon=True)


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_processed_catalog(region, year, jday, bucket=DEFAULT_BUCKET):
    """Load processed catalog files from GCS."""
    fs = get_gcs_fs()
    base_path = f"{bucket}/{region}EDC/catalog/{year:04d}/{jday:03d}"

    data = {}
    for name in ["events", "phases", "focal_mechanisms"]:
        try:
            with fs.open(f"{base_path}/{name}.csv", "r") as f:
                data[name] = pd.read_csv(f)
        except FileNotFoundError:
            data[name] = None
    return data


def load_parquet(region, year, jday, bucket=DEFAULT_BUCKET):
    """Load parquet file from GCS."""
    fs = get_gcs_fs()
    path = f"{bucket}/{region}EDC/dataset/{year:04d}/{jday:03d}.parquet"
    if not fs.exists(path):
        return None
    with fs.open(path, 'rb') as f:
        return pd.read_parquet(f)


def download_raw_ncedc_events(year, jday):
    """Download raw NCEDC events for a specific jday."""
    s3_fs = get_s3_fs()
    csv_file = f"ncedc-pds/earthquake_catalogs/NCEDC/{year}.ehpcsv"

    if not s3_fs.exists(csv_file):
        return None

    with s3_fs.open(csv_file, 'rb') as f:
        content = f.read().decode('latin-1')

    df = pd.read_csv(StringIO(content), dtype=str)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df['jday'] = df['time'].dt.dayofyear
    df = df[df['jday'] == jday]
    df['event_id'] = 'nc' + df['id'].astype(str)
    return df


def download_raw_scedc_events(year, jday):
    """Download raw SCEDC events for a specific jday."""
    s3_fs = get_s3_fs()
    csv_file = f"scedc-pds/earthquake_catalogs/{year}.csv"

    if not s3_fs.exists(csv_file):
        return None

    with s3_fs.open(csv_file, 'r') as f:
        df = pd.read_csv(f, dtype=str)

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df['jday'] = df['time'].dt.dayofyear
    return df[df['jday'] == jday]


def download_raw_ncedc_phases(year, jday):
    """Download and parse raw NCEDC phases for a specific jday."""
    s3_fs = get_s3_fs()
    date = datetime.strptime(f"{year}-{jday:03d}", "%Y-%j")
    month = date.month
    phase_file = f"ncedc-pds/event_phases/{year}/{year}.{month:02d}.phase.Z"

    if not s3_fs.exists(phase_file):
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = f"{tmpdir}/{year}.{month:02d}.phase.Z"
        s3_fs.get(phase_file, local_file)
        subprocess.run(["uncompress", "-f", local_file], check=True, capture_output=True)

        with open(local_file.replace('.Z', ''), 'r') as f:
            lines = f.readlines()

    # Parse phases
    phases = []
    current_event_id = None
    current_event_jday = None

    for line in lines:
        if len(line) > 147:  # Event line
            try:
                evt_date = datetime(int(line[0:4]), int(line[4:6]), int(line[6:8]))
                current_event_jday = evt_date.timetuple().tm_yday
                current_event_id = "nc" + line[136:146].strip()
            except:
                current_event_jday = None
        elif len(line) > 93 and current_event_jday == jday:
            if line[13:15].strip():  # P phase
                phases.append({
                    'event_id': current_event_id,
                    'network': line[5:7].strip(),
                    'station': line[0:5].strip(),
                    'component': line[11:12].strip(),
                    'phase_type': 'P',
                    'phase_polarity': line[15:16].strip() if line[15:16].strip() in ('U', 'D') else '',
                    'distance_km': float(line[74:78]) / 10 if line[74:78].strip() else None,
                })

    return pd.DataFrame(phases) if phases else None


def download_raw_scedc_phases(year, jday):
    """Download and parse raw SCEDC phases for a specific jday."""
    s3_fs = get_s3_fs()
    phase_dir = f"scedc-pds/event_phases/{year}/{year}_{jday:03d}"
    phase_files = s3_fs.glob(f"{phase_dir}/*.phase")

    if not phase_files:
        return None

    def parse_polarity(p):
        if p == '.': return ''
        if p in ('c', '+', 'u'): return 'U'
        if p in ('d', '-', 'r'): return 'D'
        return ''

    phases = []
    for pf in phase_files:
        try:
            with s3_fs.open(pf, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue

            event_id = "ci" + lines[0].strip().split()[0]
            for line in lines[1:]:
                fields = line.strip().split()
                if len(fields) < 13:
                    continue
                phases.append({
                    'event_id': event_id,
                    'network': fields[0],
                    'station': fields[1],
                    'component': fields[2][-1] if fields[2] else '',  # Last char of channel
                    'phase_type': fields[7],
                    'phase_polarity': parse_polarity(fields[8][0]) if fields[8] else '',
                    'distance_km': float(fields[11]),
                    'phase_score': float(fields[10]),
                })
        except:
            continue

    return pd.DataFrame(phases) if phases else None


def download_raw_ncedc_fm(year, jday):
    """Download raw NCEDC focal mechanisms for a specific jday."""
    s3_fs = get_s3_fs()
    date = datetime.strptime(f"{year}-{jday:03d}", "%Y-%j")
    month = date.month
    mech_files = s3_fs.glob(f"ncedc-pds/mechanism/{year}/{year}.{month:02d}.mech")

    if not mech_files:
        return None

    fms = []
    for mf in mech_files:
        with s3_fs.open(mf, 'r') as f:
            for line in f:
                line = line.rstrip('\n').ljust(142)
                if not line.strip():
                    continue
                try:
                    line_date = datetime(int(line[0:4]), int(line[4:6]), int(line[6:8]))
                    if line_date.timetuple().tm_yday != jday:
                        continue

                    dip_dir = float(line[83:86]) if line[83:86].strip() else None
                    fms.append({
                        'event_id': 'nc' + line[131:141].strip(),
                        'strike': (dip_dir - 90) % 360 if dip_dir else None,
                        'dip': float(line[87:89]) if line[87:89].strip() else None,
                        'rake': float(line[89:93]) if line[89:93].strip() else None,
                    })
                except:
                    continue

    return pd.DataFrame(fms).drop_duplicates(subset=['event_id']) if fms else None


def download_raw_scedc_fm(year, jday):
    """Download raw SCEDC focal mechanisms for a specific jday."""
    import requests

    if year <= 2010:
        url = "https://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_focal/YSH_2010.hash"
    else:
        url = f"https://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_focal/sc{year}_hash_ABCD_so.focmec.scedc"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
    except:
        return None

    fms = []
    for line in response.text.split('\n'):
        tokens = line.strip().split()
        if len(tokens) != 21:
            continue
        try:
            fm_date = datetime(int(tokens[0]), int(tokens[1]), int(tokens[2]))
            if fm_date.timetuple().tm_yday != jday:
                continue
            fms.append({
                'event_id': f"ci{tokens[6]}",
                'strike': float(tokens[11]),
                'dip': float(tokens[12]),
                'rake': float(tokens[13]),
            })
        except:
            continue

    return pd.DataFrame(fms) if fms else None


# =============================================================================
# EVENT VALIDATION
# =============================================================================

def validate_events(region, year, jday, n_samples=20):
    """Validate event catalog: counts, IDs, and attribute values."""
    print(f"\n{'='*60}")
    print(f"EVENT VALIDATION: {region}EDC {year}/{jday:03d}")
    print(f"{'='*60}")

    # Load processed events
    catalog = load_processed_catalog(region, year, jday)
    if catalog["events"] is None:
        print("  ERROR: Events file not found")
        return False

    processed = catalog["events"]
    print(f"\n1. Processed events: {len(processed)}")

    # Download raw events
    print("\n2. Downloading raw events...")
    if region == "NC":
        raw = download_raw_ncedc_events(year, jday)
    else:
        raw = download_raw_scedc_events(year, jday)

    if raw is None:
        print("  Could not download raw data")
        return True

    print(f"   Raw events: {len(raw)}")

    # Check counts
    print("\n3. Checking counts...")
    if len(processed) == len(raw):
        print(f"   ✓ Counts match: {len(processed)}")
    else:
        print(f"   ✗ Count mismatch: processed={len(processed)}, raw={len(raw)}")

    # Check IDs
    print("\n4. Checking event IDs...")
    processed_ids = set(processed['event_id'])
    raw_ids = set(raw['event_id'])

    missing_in_processed = raw_ids - processed_ids
    extra_in_processed = processed_ids - raw_ids

    if not missing_in_processed and not extra_in_processed:
        print(f"   ✓ All {len(processed_ids)} event IDs match")
    else:
        if missing_in_processed:
            print(f"   ✗ Missing in processed: {list(missing_in_processed)[:5]}")
        if extra_in_processed:
            print(f"   ✗ Extra in processed: {list(extra_in_processed)[:5]}")

    # Check attribute values
    print(f"\n5. Checking attribute values ({n_samples} samples)...")

    # Field mappings
    if region == "NC":
        field_map = [
            ('latitude', 'latitude', 0.0001),
            ('longitude', 'longitude', 0.0001),
            ('depth', 'depth_km', 0.01),
            ('mag', 'magnitude', 0.01),
        ]
    else:
        field_map = [
            ('latitude', 'latitude', 0.0001),
            ('longitude', 'longitude', 0.0001),
            ('depth_km', 'depth_km', 0.01),
            ('magnitude', 'magnitude', 0.01),
        ]

    processed = processed.set_index('event_id')
    raw = raw.set_index('event_id')
    common_ids = list(processed_ids & raw_ids)[:n_samples]

    mismatches = []
    for event_id in common_ids:
        proc_row = processed.loc[event_id]
        raw_row = raw.loc[event_id]

        for raw_col, proc_col, tol in field_map:
            if raw_col not in raw.columns or proc_col not in processed.columns:
                continue
            try:
                raw_val = float(raw_row[raw_col])
                proc_val = float(proc_row[proc_col])
                if abs(raw_val - proc_val) > tol:
                    mismatches.append({
                        'event_id': event_id, 'field': proc_col,
                        'raw': raw_val, 'processed': proc_val
                    })
            except:
                pass

    if mismatches:
        print(f"   ✗ Found {len(mismatches)} value mismatches:")
        for m in mismatches[:5]:
            print(f"      {m['event_id']}.{m['field']}: raw={m['raw']:.4f}, proc={m['processed']:.4f}")
        return False
    else:
        print(f"   ✓ All sampled attribute values match")
        return True


# =============================================================================
# PHASE VALIDATION
# =============================================================================

def validate_phases(region, year, jday, n_samples=50):
    """Validate phase catalog: counts, keys, and attribute values."""
    print(f"\n{'='*60}")
    print(f"PHASE VALIDATION: {region}EDC {year}/{jday:03d}")
    print(f"{'='*60}")

    # Load processed phases
    catalog = load_processed_catalog(region, year, jday)
    if catalog["phases"] is None:
        print("  ERROR: Phases file not found")
        return False

    processed = catalog["phases"]
    print(f"\n1. Processed phases: {len(processed)}")

    # Download raw phases
    print("\n2. Downloading raw phases...")
    if region == "NC":
        raw = download_raw_ncedc_phases(year, jday)
    else:
        raw = download_raw_scedc_phases(year, jday)

    if raw is None:
        print("  Could not download raw data")
        return True

    print(f"   Raw phases: {len(raw)}")

    # Check counts
    print("\n3. Checking counts...")
    proc_p = processed[processed['phase_type'] == 'P']
    raw_p = raw[raw['phase_type'] == 'P']

    print(f"   Processed P phases: {len(proc_p)}")
    print(f"   Raw P phases: {len(raw_p)}")

    if abs(len(proc_p) - len(raw_p)) <= 5:
        print(f"   ✓ P phase counts approximately match")
    else:
        print(f"   ! P phase count difference: {abs(len(proc_p) - len(raw_p))}")

    # Check keys (event_id + network + station + component)
    print("\n4. Checking phase keys...")
    processed['key'] = processed['event_id'] + '_' + processed['network'] + '_' + processed['station'] + '_' + processed['component']
    raw['key'] = raw['event_id'] + '_' + raw['network'] + '_' + raw['station'] + '_' + raw['component']

    proc_keys = set(processed[processed['phase_type'] == 'P']['key'])
    raw_keys = set(raw[raw['phase_type'] == 'P']['key'])

    common_keys = proc_keys & raw_keys
    print(f"   Common P phase keys: {len(common_keys)}")

    missing = raw_keys - proc_keys
    if missing:
        print(f"   Missing in processed: {len(missing)} (first 3: {list(missing)[:3]})")

    # Check attribute values
    print(f"\n5. Checking attribute values ({n_samples} samples)...")

    sample_keys = list(common_keys)[:n_samples]
    proc_indexed = processed.set_index('key')
    raw_indexed = raw.set_index('key')

    mismatches = []
    for key in sample_keys:
        try:
            proc_row = proc_indexed.loc[key]
            raw_row = raw_indexed.loc[key]

            # Handle multiple matches
            if isinstance(proc_row, pd.DataFrame):
                proc_row = proc_row.iloc[0]
            if isinstance(raw_row, pd.DataFrame):
                raw_row = raw_row.iloc[0]

            # Check distance
            if 'distance_km' in raw.columns and pd.notna(raw_row['distance_km']) and pd.notna(proc_row['distance_km']):
                if abs(float(raw_row['distance_km']) - float(proc_row['distance_km'])) > 0.5:
                    mismatches.append({'key': key, 'field': 'distance_km',
                                      'raw': raw_row['distance_km'], 'processed': proc_row['distance_km']})

            # Check polarity
            raw_pol = str(raw_row.get('phase_polarity', '')).strip()
            proc_pol = str(proc_row.get('phase_polarity', '')).strip() if pd.notna(proc_row.get('phase_polarity')) else ''
            if raw_pol and raw_pol != proc_pol:
                mismatches.append({'key': key, 'field': 'phase_polarity',
                                  'raw': raw_pol, 'processed': proc_pol})
        except:
            pass

    if mismatches:
        print(f"   ✗ Found {len(mismatches)} value mismatches:")
        for m in mismatches[:5]:
            print(f"      {m['key']}.{m['field']}: raw={m['raw']}, proc={m['processed']}")
        return False
    else:
        print(f"   ✓ All sampled attribute values match")
        return True


# =============================================================================
# FOCAL MECHANISM VALIDATION
# =============================================================================

def validate_fm(region, year, jday, n_samples=20):
    """Validate focal mechanism catalog: counts, IDs, and attribute values."""
    print(f"\n{'='*60}")
    print(f"FOCAL MECHANISM VALIDATION: {region}EDC {year}/{jday:03d}")
    print(f"{'='*60}")

    # Load processed FMs
    catalog = load_processed_catalog(region, year, jday)
    if catalog["focal_mechanisms"] is None:
        print("  No focal mechanisms file found (may be expected)")
        return True

    processed = catalog["focal_mechanisms"]
    print(f"\n1. Processed focal mechanisms: {len(processed)}")

    # Download raw FMs
    print("\n2. Downloading raw focal mechanisms...")
    if region == "NC":
        raw = download_raw_ncedc_fm(year, jday)
    else:
        raw = download_raw_scedc_fm(year, jday)

    if raw is None:
        print("  Could not download raw data")
        return True

    print(f"   Raw focal mechanisms: {len(raw)}")

    # Check counts
    print("\n3. Checking counts...")
    if len(processed) == len(raw):
        print(f"   ✓ Counts match: {len(processed)}")
    else:
        print(f"   ! Count difference: processed={len(processed)}, raw={len(raw)}")

    # Check IDs
    print("\n4. Checking event IDs...")
    processed_ids = set(processed['event_id'])
    raw_ids = set(raw['event_id'])

    common_ids = processed_ids & raw_ids
    print(f"   Common IDs: {len(common_ids)}")

    missing = raw_ids - processed_ids
    if missing:
        print(f"   Missing in processed: {list(missing)[:3]}")

    # Check attribute values
    print(f"\n5. Checking attribute values ({n_samples} samples)...")

    processed = processed.set_index('event_id')
    raw = raw.set_index('event_id')
    sample_ids = list(common_ids)[:n_samples]

    mismatches = []
    for event_id in sample_ids:
        try:
            proc_row = processed.loc[event_id]
            raw_row = raw.loc[event_id]

            for field in ['strike', 'dip', 'rake']:
                if pd.notna(raw_row[field]) and pd.notna(proc_row[field]):
                    if abs(float(raw_row[field]) - float(proc_row[field])) > 0.5:
                        mismatches.append({
                            'event_id': event_id, 'field': field,
                            'raw': raw_row[field], 'processed': proc_row[field]
                        })
        except:
            pass

    if mismatches:
        print(f"   ✗ Found {len(mismatches)} value mismatches:")
        for m in mismatches[:5]:
            print(f"      {m['event_id']}.{m['field']}: raw={m['raw']:.1f}, proc={m['processed']:.1f}")
        return False
    else:
        print(f"   ✓ All sampled attribute values match")
        return True


# =============================================================================
# PARQUET VALIDATION
# =============================================================================

def normalize_trace(trace):
    """Normalize trace for plotting."""
    trace = np.array(trace, dtype=float)
    trace = trace - np.mean(trace)
    max_val = np.max(np.abs(trace))
    return trace / max_val if max_val > 0 else trace


def validate_parquet(region, year, jday, n_events=20, output_dir=OUTPUT_DIR):
    """Validate parquet file vs CSV, generate moveout and zoom plots."""
    print(f"\n{'='*60}")
    print(f"PARQUET VALIDATION: {region}EDC {year}/{jday:03d}")
    print(f"{'='*60}")

    # Use region-specific output directory
    output_dir = f"{output_dir}/{region}EDC"
    os.makedirs(output_dir, exist_ok=True)

    # Load parquet
    parquet_df = load_parquet(region, year, jday)
    if parquet_df is None:
        print("\n  Parquet file not found. Run cut_event_parquet.py first.")
        return False

    print(f"\n1. Parquet loaded: {len(parquet_df)} rows, {len(parquet_df['event_id'].unique())} events")

    # Load CSVs
    catalog = load_processed_catalog(region, year, jday)
    if catalog["events"] is None:
        print("  Events CSV not found")
        return False

    events_csv = catalog["events"].set_index('event_id')
    phases_csv = catalog["phases"]

    # Validate attributes
    print("\n2. Validating parquet vs CSV attributes...")
    sample_events = parquet_df['event_id'].unique()[:n_events]
    mismatches = []

    for event_id in sample_events:
        pq_row = parquet_df[parquet_df['event_id'] == event_id].iloc[0]
        if event_id not in events_csv.index:
            continue
        csv_row = events_csv.loc[event_id]

        for pq_field, csv_field, tol in [
            ('event_latitude', 'latitude', 0.0001),
            ('event_longitude', 'longitude', 0.0001),
            ('event_depth_km', 'depth_km', 0.01),
            ('event_magnitude', 'magnitude', 0.01),
        ]:
            try:
                if abs(float(pq_row[pq_field]) - float(csv_row[csv_field])) > tol:
                    mismatches.append({'id': event_id, 'field': pq_field})
            except:
                pass

    if mismatches:
        print(f"   ✗ Found {len(mismatches)} mismatches")
    else:
        print(f"   ✓ All parquet attributes match CSV")

    # Generate moveout plots
    print("\n3. Generating moveout plots...")
    for event_id in sample_events[:5]:
        _plot_moveout(parquet_df, event_id, region, output_dir)

    # Generate zoom plots
    print("\n4. Generating zoom-in views...")
    events_with_ps = parquet_df[parquet_df['s_phase_index'].notna()]['event_id'].unique()[:10]
    count = 0
    for event_id in events_with_ps:
        event_data = parquet_df[parquet_df['event_id'] == event_id]
        for _, row in event_data[event_data['s_phase_index'].notna()].head(2).iterrows():
            _plot_zoom(row, region, output_dir)
            count += 1
            if count >= n_events:
                break
        if count >= n_events:
            break

    print(f"\n   Generated 5 moveout plots and {count} zoom-in views in {output_dir}/")
    return len(mismatches) == 0


def _plot_moveout(df, event_id, region, output_dir):
    """Generate moveout plot for an event."""
    event_data = df[df['event_id'] == event_id].copy()
    event_data = event_data[event_data['distance_km'].notna()].sort_values('distance_km')

    if len(event_data) == 0:
        return

    evt = event_data.iloc[0]
    n_traces = len(event_data)
    fig, ax = plt.subplots(figsize=(14, max(8, n_traces * 0.4)))

    sampling_rate = 100.0

    for i, (_, row) in enumerate(event_data.iterrows()):
        z_trace = np.array(row['waveform'][2])
        nt = len(z_trace)
        time = np.arange(nt) / sampling_rate

        ax.plot(time, normalize_trace(z_trace) * 0.4 + i, 'k', lw=0.5, alpha=0.8)

        if pd.notna(row['p_phase_index']):
            p_time = row['p_phase_index'] / sampling_rate
            ax.plot([p_time, p_time], [i - 0.35, i + 0.35], 'r', lw=1.5)
            ax.text(p_time + 0.5, i, 'P', color='red', fontsize=7, va='center', fontweight='bold')

        if pd.notna(row['s_phase_index']):
            s_time = row['s_phase_index'] / sampling_rate
            ax.plot([s_time, s_time], [i - 0.35, i + 0.35], 'b', lw=1.5)
            ax.text(s_time + 0.5, i, 'S', color='blue', fontsize=7, va='center', fontweight='bold')

    ylabels = [f"{r['network']}.{r['station']} ({r['distance_km']:.1f}km)" for _, r in event_data.iterrows()]
    ax.set_yticks(range(n_traces))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{region}EDC {event_id} | M{evt["event_magnitude"]:.2f}')
    ax.set_xlim(0, nt / sampling_rate)
    ax.set_ylim(-0.5, n_traces - 0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/moveout_{event_id}.png', dpi=150)
    plt.close()
    print(f"   Saved: moveout_{event_id}.png")


def _plot_zoom(row, region, output_dir):
    """Generate zoom-in view for a single trace."""
    waveform = row['waveform']
    sampling_rate = 100.0
    traces = [np.array(waveform[i]) for i in range(3)]
    nt = len(traces[0])
    time = np.arange(nt) / sampling_rate

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.6])

    components = ['E', 'N', 'Z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Full waveform
    ax_full = fig.add_subplot(gs[0, :2])
    for i, (trace, comp, color) in enumerate(zip(traces, components, colors)):
        ax_full.plot(time, normalize_trace(trace) + i, color=color, lw=0.5, label=comp)

    p_idx, s_idx = row['p_phase_index'], row['s_phase_index']
    if pd.notna(p_idx):
        ax_full.axvline(p_idx / sampling_rate, color='red', ls='--', lw=1.5)
    if pd.notna(s_idx):
        ax_full.axvline(s_idx / sampling_rate, color='blue', ls='--', lw=1.5)

    ax_full.set_title(f"{row['event_id']} | {row['network']}.{row['station']}")
    ax_full.set_yticks([0, 1, 2])
    ax_full.set_yticklabels(components)
    ax_full.legend(loc='upper right')
    ax_full.set_xlim(0, time[-1])

    # P zoom
    ax_p = fig.add_subplot(gs[1, 0])
    if pd.notna(p_idx):
        p_time = p_idx / sampling_rate
        t_start, t_end = max(0, p_time - 2), min(time[-1], p_time + 3)
        idx_s, idx_e = int(t_start * sampling_rate), int(t_end * sampling_rate)
        for i, (trace, color) in enumerate(zip(traces, colors)):
            ax_p.plot(time[idx_s:idx_e], normalize_trace(trace[idx_s:idx_e]) + i, color=color, lw=0.8)
        ax_p.axvline(p_time, color='red', ls='--', lw=2)
        ax_p.set_xlim(t_start, t_end)
        pol = row['p_phase_polarity'] if pd.notna(row['p_phase_polarity']) else 'N/A'
        ax_p.set_title(f'P-wave | Polarity: {pol}')
    ax_p.set_xlabel('Time (s)')

    # S zoom
    ax_s = fig.add_subplot(gs[1, 1])
    if pd.notna(s_idx):
        s_time = s_idx / sampling_rate
        t_start, t_end = max(0, s_time - 2), min(time[-1], s_time + 5)
        idx_s, idx_e = int(t_start * sampling_rate), int(t_end * sampling_rate)
        for i, (trace, color) in enumerate(zip(traces, colors)):
            ax_s.plot(time[idx_s:idx_e], normalize_trace(trace[idx_s:idx_e]) + i, color=color, lw=0.8)
        ax_s.axvline(s_time, color='blue', ls='--', lw=2)
        ax_s.set_xlim(t_start, t_end)
        ax_s.set_title('S-wave')
    ax_s.set_xlabel('Time (s)')

    # Attributes
    ax_attr = fig.add_subplot(gs[:, 2])
    ax_attr.axis('off')
    attr_text = f"""EVENT
event_id: {row['event_id']}
magnitude: {row['event_magnitude']:.2f}
depth_km: {row['event_depth_km']:.2f}

STATION
network: {row['network']}
station: {row['station']}
distance_km: {row['distance_km']:.2f}
azimuth: {row['azimuth']:.1f}

P-PHASE
score: {f"{row['p_phase_score']:.2f}" if pd.notna(row['p_phase_score']) else 'N/A'}
polarity: {row['p_phase_polarity'] if pd.notna(row['p_phase_polarity']) else 'N/A'}

S-PHASE
score: {f"{row['s_phase_score']:.2f}" if pd.notna(row['s_phase_score']) else 'N/A'}
"""
    ax_attr.text(0.05, 0.95, attr_text, transform=ax_attr.transAxes, fontsize=10,
                va='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    station_id = f"{row['network']}_{row['station']}"
    plt.savefig(f'{output_dir}/zoom_{row["event_id"]}_{station_id}.png', dpi=150)
    plt.close()


# =============================================================================
# HDF5 VALIDATION
# =============================================================================

def validate_hdf5(region, year, jday, bucket=DEFAULT_BUCKET):
    """Validate HDF5 waveform file."""
    print(f"\n{'='*60}")
    print(f"HDF5 VALIDATION: {region}EDC {year}/{jday:03d}")
    print(f"{'='*60}")

    import h5py

    fs = get_gcs_fs()
    h5_path = f"{bucket}/{region}EDC/dataset/{year:04d}/{jday:03d}/waveform.h5"

    print(f"\n1. Checking HDF5 file exists...")
    if not fs.exists(h5_path):
        print(f"   ✗ HDF5 file not found: {h5_path}")
        return False

    print(f"   ✓ Found: {h5_path}")

    # Open and inspect
    print("\n2. Inspecting HDF5 structure...")
    try:
        with fs.open(h5_path, 'rb') as f:
            with h5py.File(f, 'r') as h5:
                n_events = len(h5.keys())
                print(f"   Events in HDF5: {n_events}")

                # Sample event
                if n_events > 0:
                    sample_event = list(h5.keys())[0]
                    event_grp = h5[sample_event]
                    print(f"\n   Sample event: {sample_event}")
                    print(f"   Stations: {len(event_grp.keys())}")

                    if len(event_grp.keys()) > 0:
                        sample_station = list(event_grp.keys())[0]
                        station_data = event_grp[sample_station]
                        print(f"   Sample station: {sample_station}")
                        print(f"   Waveform shape: {station_data.shape}")
                        print(f"   Attributes: {list(station_data.attrs.keys())}")
    except Exception as e:
        print(f"   ✗ Error reading HDF5: {e}")
        return False

    # Cross-check with CSV
    print("\n3. Cross-checking with event CSV...")
    catalog = load_processed_catalog(region, year, jday, bucket)
    if catalog["events"] is not None:
        csv_events = set(catalog["events"]['event_id'])
        print(f"   Events in CSV: {len(csv_events)}")

        with fs.open(h5_path, 'rb') as f:
            with h5py.File(f, 'r') as h5:
                h5_events = set(h5.keys())

        common = csv_events & h5_events
        print(f"   Common events: {len(common)}")

        missing_in_h5 = csv_events - h5_events
        if missing_in_h5:
            print(f"   Missing in HDF5: {len(missing_in_h5)}")

    print("\n   ✓ HDF5 validation complete")
    return True


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests(region, year, jday, n_events=20, output_dir=OUTPUT_DIR):
    """Run all validation tests."""
    print(f"\n{'#'*60}")
    print(f"# FULL VALIDATION: {region}EDC {year}/{jday:03d}")
    print(f"{'#'*60}")

    results = {}
    results['event'] = validate_events(region, year, jday)
    results['phase'] = validate_phases(region, year, jday)
    results['fm'] = validate_fm(region, year, jday)
    results['parquet'] = validate_parquet(region, year, jday, n_events, output_dir)
    results['hdf5'] = validate_hdf5(region, year, jday)

    print(f"\n{'#'*60}")
    print("# SUMMARY")
    print(f"{'#'*60}")
    for test, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test}: {status}")

    return all(results.values())


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset validation for NCEDC/SCEDC")
    parser.add_argument("--region", type=str, default="NC", choices=["NC", "SC"])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--jday", type=int, default=1)
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "event", "phase", "fm", "parquet", "hdf5"])
    parser.add_argument("--n_events", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.test == "all":
        success = run_all_tests(args.region, args.year, args.jday, args.n_events, args.output_dir)
    elif args.test == "event":
        success = validate_events(args.region, args.year, args.jday)
    elif args.test == "phase":
        success = validate_phases(args.region, args.year, args.jday)
    elif args.test == "fm":
        success = validate_fm(args.region, args.year, args.jday)
    elif args.test == "parquet":
        success = validate_parquet(args.region, args.year, args.jday, args.n_events, args.output_dir)
    elif args.test == "hdf5":
        success = validate_hdf5(args.region, args.year, args.jday)
    else:
        print(f"Unknown test: {args.test}")
        success = False

    sys.exit(0 if success else 1)

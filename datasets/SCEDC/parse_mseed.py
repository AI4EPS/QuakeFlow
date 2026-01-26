# %%
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import fsspec
from tqdm import tqdm

# %%
input_protocol = "s3"
input_bucket = "scedc-pds"
input_folder = "continuous_waveforms"

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset"
output_folder = "SCEDC/mseed"

valid_instruments = {"BH", "HH", "EH", "HN", "DP", "SH", "EP"}
valid_components = {"3", "2", "1", "E", "N", "Z"}

input_fs = fsspec.filesystem(input_protocol, anon=True)


# %%
def parse_fname(fname):
    """Parse SCEDC mseed filename.

    Format: {network}{station}{location}{channel}___{year}{jday}.ms
    Example: AZCRY__BHE___2026001.ms
    """
    return {
        "network": fname[:2],
        "station": fname[2:7].rstrip("_"),
        "location": fname[10:12].rstrip("_"),
        "instrument": fname[7:9],
        "component": fname[9],
        "year": fname[14:18],
        "jday": fname[17:20],
    }


def discover_jday_folders():
    """Discover all jday folders: years -> jdays."""
    print("Discovering folder structure...")

    # Step 1: List years
    years = input_fs.ls(f"{input_bucket}/{input_folder}", detail=False)
    years = [y for y in years if y.split("/")[-1].isdigit()]
    print(f"  Found {len(years)} years")

    # Step 2: List jdays per year (parallel)
    def list_jdays(year_path):
        return input_fs.ls(year_path, detail=False)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(list_jdays, years))

    jday_folders = [f for result in results for f in result]
    print(f"  Found {len(jday_folders)} jday folders")
    return jday_folders


def process_jday_folder(jday_path):
    """List files in jday folder, group by station, save and upload."""
    # Parse year/jday from path: .../year/year_jday
    year_jday = jday_path.split("/")[-1]  # e.g., "2024_001"
    year, jday = year_jday.split("_")

    # List all files in this jday folder
    files = input_fs.ls(jday_path, detail=False)
    station_groups = defaultdict(list)

    for key in files:
        fname = key.split("/")[-1]
        if not fname.endswith(".ms"):
            continue

        try:
            info = parse_fname(fname)
        except Exception as e:
            print(f"Failed to parse filename: {fname}, error: {e}")
            continue

        if info["instrument"] in valid_instruments and info["component"] in valid_components:
            station_key = f"{info['network']}.{info['station']}.{info['location']}.{info['instrument']}"
            station_groups[station_key].append(f"{input_protocol}://{key}")

    if not station_groups:
        return 0, None, None

    # Save to local file
    local_dir = f"mseed/{year}"
    os.makedirs(local_dir, exist_ok=True)
    local_path = f"{local_dir}/{jday}.txt"
    entries = ["|".join(sorted(files)) for files in station_groups.values()]
    with open(local_path, "w") as f:
        f.write("\n".join(sorted(entries)))

    remote_path = f"{output_bucket}/{output_folder}/{year}/{jday}.txt"
    return len(entries), local_path, remote_path


# %%
if __name__ == "__main__":
    # Step 1: Discover all jday folders (fast ls-based listing)
    jday_folders = discover_jday_folders()

    # Step 2: Process all jday folders in parallel
    print(f"\nProcessing {len(jday_folders)} jday folders...")
    with ThreadPoolExecutor(max_workers=64) as executor:
        results = list(tqdm(executor.map(process_jday_folder, jday_folders), total=len(jday_folders), desc="Processing"))

    # Collect upload tasks
    upload_tasks = [(local, remote) for _, local, remote in results if local]
    total_groups = sum(r[0] for r in results)

    # Step 3: Upload to GCS in parallel
    output_fs = fsspec.filesystem(output_protocol, token=output_token)

    def upload_file(paths):
        local, remote = paths
        output_fs.put(local, remote)

    print(f"\nUploading {len(upload_tasks)} files to GCS...")
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(tqdm(executor.map(upload_file, upload_tasks), total=len(upload_tasks), desc="Uploading"))

    print(f"Saved {total_groups} station-day groups across {len(upload_tasks)} jdays")

# %%

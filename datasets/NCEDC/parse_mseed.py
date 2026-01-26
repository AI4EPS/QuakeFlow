# %%
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import fsspec
from tqdm import tqdm

# %%
input_protocol = "s3"
input_bucket = "ncedc-pds"
input_folder = "continuous_waveforms"

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset"
output_folder = "NCEDC/mseed"

valid_instruments = {"BH", "HH", "EH", "HN", "DP", "SH", "EP"}
valid_components = {"3", "2", "1", "E", "N", "Z"}

input_fs = fsspec.filesystem(input_protocol, anon=True)


# %%
def parse_fname(fname):
    """Parse NCEDC mseed filename.

    Format: {station}.{network}.{channel}.{location}.{quality}.{year}.{jday}
    Example: ABL.CI.HHE..D.2026.001
    """
    parts = fname.split(".")
    return {
        "network": parts[1],
        "station": parts[0],
        "location": parts[3],
        "instrument": parts[2][:2],
        "component": parts[2][-1],
        "year": parts[5],
        "jday": parts[6],
    }


def discover_jday_folders():
    """Discover all jday folders: networks -> years -> jdays."""
    print("Discovering folder structure...")

    # Step 1: List networks
    networks = input_fs.ls(f"{input_bucket}/{input_folder}", detail=False)
    networks = [n for n in networks if len(n.split("/")[-1]) == 2]
    print(f"  Found {len(networks)} networks")

    # Step 2: List years per network (parallel)
    def list_years(network_path):
        try:
            return input_fs.ls(network_path, detail=False)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(list_years, networks))

    year_folders = [y for result in results for y in result if y.split("/")[-1].isdigit()]
    print(f"  Found {len(year_folders)} network-year combinations")

    # Step 3: List jdays per year (parallel)
    def list_jdays(year_path):
        try:
            return input_fs.ls(year_path, detail=False)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(list_jdays, year_folders))

    # Filter to only valid jday folders: {year}.{3-digit-jday}
    def is_valid_jday(path):
        name = path.split("/")[-1]
        if "." not in name or len(name) != 8:
            return False
        parts = name.split(".")
        return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()

    jday_folders = [f for result in results for f in result if is_valid_jday(f)]
    print(f"  Found {len(jday_folders)} jday folders")
    return jday_folders


def process_jday_folder(jday_path):
    """List files in jday folder, group by station, return results."""
    # Parse year/jday from path: .../network/year/year.jday
    year_jday = jday_path.split("/")[-1]  # e.g., "2024.001"
    year, jday = year_jday.split(".")

    # List all files in this jday folder
    try:
        files = input_fs.ls(jday_path, detail=False)
    except Exception:
        return []

    results = []
    for key in files:
        fname = key.split("/")[-1]
        try:
            info = parse_fname(fname)
        except Exception as e:
            print(f"Failed to parse filename: {fname}, error: {e}")
            continue

        if info["instrument"] in valid_instruments and info["component"] in valid_components:
            station_key = f"{info['network']}.{info['station']}.{info['location']}.{info['instrument']}"
            results.append((year, jday, station_key, f"{input_protocol}://{key}"))

    return results


# %%
if __name__ == "__main__":
    # Step 1: Discover all jday folders (fast ls-based listing)
    jday_folders = discover_jday_folders()

    # Step 2: Process all jday folders in parallel
    print(f"\nProcessing {len(jday_folders)} jday folders...")
    with ThreadPoolExecutor(max_workers=128) as executor:
        results = list(tqdm(executor.map(process_jday_folder, jday_folders), total=len(jday_folders), desc="Processing"))

    # Step 3: Merge results by (year, jday)
    print("\nGrouping results...")
    jday_groups = defaultdict(lambda: defaultdict(list))
    for folder_results in results:
        for year, jday, station_key, file_path in folder_results:
            jday_groups[(year, jday)][station_key].append(file_path)

    # Step 4: Save to local files
    print(f"Saving {len(jday_groups)} jday files...")
    upload_tasks = []
    total_groups = 0
    for (year, jday), station_groups in sorted(jday_groups.items()):
        local_dir = f"mseed/{year}"
        os.makedirs(local_dir, exist_ok=True)
        local_path = f"{local_dir}/{jday}.txt"
        entries = ["|".join(sorted(files)) for files in station_groups.values()]
        with open(local_path, "w") as f:
            f.write("\n".join(sorted(entries)))
        upload_tasks.append((local_path, f"{output_bucket}/{output_folder}/{year}/{jday}.txt"))
        total_groups += len(entries)

    # Step 5: Upload to GCS in parallel
    output_fs = fsspec.filesystem(output_protocol, token=output_token)

    def upload_file(paths):
        local, remote = paths
        output_fs.put(local, remote)

    print(f"\nUploading {len(upload_tasks)} files to GCS...")
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(tqdm(executor.map(upload_file, upload_tasks), total=len(upload_tasks), desc="Uploading"))

    print(f"Saved {total_groups} station-day groups across {len(upload_tasks)} jdays")

# %%

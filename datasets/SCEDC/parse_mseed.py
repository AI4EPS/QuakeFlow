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


# %%
def parse_fname(mseed):
    """Parse SCEDC mseed filename.

    Format: {network}{station}{location}{channel}___{year}{jday}.ms
    Example: AZCRY__BHE___2026001.ms
    """
    fname = mseed.split("/")[-1]
    return {
        "network": fname[:2],
        "station": fname[2:7].rstrip("_"),
        "location": fname[10:12].rstrip("_"),
        "instrument": fname[7:9],
        "component": fname[9],
        "jday": fname[17:20],
    }


def scan_jday(args):
    """Scan a single jday folder and return grouped mseed files."""
    jday_path, input_fs = args
    folder_name = jday_path.split("/")[-1]
    year, jday = folder_name.split("_")

    # Use ls instead of glob - faster for listing directory contents
    try:
        files = input_fs.ls(jday_path, detail=False)
    except Exception:
        return jday, []

    suffix = f"_{year}{jday}.ms"
    groups = defaultdict(list)
    for mseed in files:
        fname = mseed.split("/")[-1]
        # Quick filter before parsing
        if not fname.endswith(suffix):
            continue

        try:
            info = parse_fname(mseed)
        except Exception:
            continue

        if info["instrument"] in valid_instruments and info["component"] in valid_components:
            key = f"{info['network']}.{info['station']}.{info['location']}.{info['instrument']}"
            groups[key].append(f"{input_protocol}://{mseed}")

    result = ["|".join(sorted(files)) for files in groups.values()]
    return jday, result


def collect_mseeds(year, max_workers=64):
    """Collect and group mseed files by station for a given year."""
    input_fs = fsspec.filesystem(input_protocol, anon=True)
    output_fs = fsspec.filesystem(output_protocol, token=output_token)

    # Find all jday folders for this year using ls
    try:
        year_path = f"{input_bucket}/{input_folder}/{year}"
        folders = input_fs.ls(year_path, detail=False)
        jdays = [f for f in folders if f.split("/")[-1].startswith(f"{year}_")]
    except FileNotFoundError:
        print(f"No day folders found for year {year}")
        return 0

    if not jdays:
        print(f"No day folders found for year {year}")
        return 0

    print(f"Found {len(jdays)} day folders for year {year}")

    # Process jdays in parallel using threads (I/O-bound)
    jday_groups = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(scan_jday, [(jday, input_fs) for jday in jdays])
        for jd, entries in tqdm(futures, total=len(jdays), desc=f"Scanning {year}"):
            jday_groups[jd].extend(entries)

    # Save each jday to separate file
    local_dir = f"mseed/{year}"
    total_groups = 0
    local_files = []
    for jday, entries in sorted(jday_groups.items()):
        if not entries:
            continue
        os.makedirs(local_dir, exist_ok=True)
        local_path = f"{local_dir}/{jday}.txt"
        with open(local_path, "w") as f:
            f.write("\n".join(sorted(entries)))
        local_files.append((local_path, f"{output_bucket}/{output_folder}/{year}/{jday}.txt"))
        total_groups += len(entries)

    # Upload all files in parallel
    def upload_file(paths):
        local, remote = paths
        output_fs.put(local, remote)

    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(upload_file, local_files))

    print(f"Saved {total_groups} station-day groups across {len(local_files)} files for year {year}")
    return total_groups


# %%
if __name__ == "__main__":
    input_fs = fsspec.filesystem(input_protocol, anon=True)

    # Find all available years using ls
    try:
        year_folders = input_fs.ls(f"{input_bucket}/{input_folder}", detail=False)
        years = sorted([int(yf.split("/")[-1]) for yf in year_folders if yf.split("/")[-1].isdigit()])
    except Exception as e:
        print(f"Error listing years: {e}")
        years = []

    print(f"Found years: {years}")

    # Process years sequentially (each year already parallelized internally)
    for year in years:
        print(f"\nProcessing year {year}")
        collect_mseeds(year)

# %%

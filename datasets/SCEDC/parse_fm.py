# %%
# https://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_focal/sc2024_hash_ABCD_so.focmec.scedc
# https://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_focal/SouthernCalifornia_1981-2011_focalmec_Format.pdf

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import pandas as pd
from tqdm import tqdm

input_url = "https://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_focal"
input_files = ["YSH_2010.hash"] + [f"sc{x}_hash_ABCD_so.focmec.scedc" for x in range(2011, 2025)]

output_protocol = "gs"
output_bucket = "quakeflow_dataset"
output_folder = "SCEDC/catalog"

result_path = "catalog"
os.makedirs(result_path, exist_ok=True)


# %%
def parse_focal_mechanism_line(line):
    """Parse a single line of SCEDC focal mechanism data."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    tokens = line.split()
    if len(tokens) != 21:
        print(f"Unexpected number of tokens ({len(tokens)} != 21) in line: {line}")
        return None

    try:
        year = int(tokens[0])
        month = int(tokens[1])
        day = int(tokens[2])
        hour = int(tokens[3])
        minute = int(tokens[4])
        second = float(tokens[5])
        event_id = tokens[6]
        latitude = float(tokens[7])
        longitude = float(tokens[8])
        depth = float(tokens[9])
        magnitude = float(tokens[10])
        strike = float(tokens[11])
        dip = float(tokens[12])
        rake = float(tokens[13])
        fault_plane_uncertainty = float(tokens[14])
        aux_fault_plane_uncertainty = float(tokens[15])
        num_first_motions = int(tokens[16])
        first_motion_misfit = float(tokens[17])
        num_sp_ratios = int(tokens[18])
        sp_ratio_misfit = float(tokens[19])
        quality = tokens[20]

        # Validate time components
        if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second < 60):
            print(f"Invalid time components in line: {line}")
            return None

        time_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:06.3f}"

        return {
            'event_id': f"ci{event_id}",
            'time': time_str,
            'latitude': latitude,
            'longitude': longitude,
            'depth_km': depth,
            'magnitude': magnitude,
            'strike': strike,
            'dip': dip,
            'rake': rake,
            'plane1_uncertainty': fault_plane_uncertainty,
            'plane2_uncertainty': aux_fault_plane_uncertainty,
            'num_first_motions': num_first_motions,
            'first_motion_misfit': first_motion_misfit,
            'num_sp_ratios': num_sp_ratios,
            'sp_ratio_misfit': sp_ratio_misfit,
            'quality': quality,
        }
    except (ValueError, IndexError) as e:
        print(f"Error parsing line: {line}")
        print(f"Error: {e}")
        return None


def parse_focal_mechanism_file(file_path):
    """Parse a focal mechanism file from URL and return a DataFrame."""
    records = []

    fs = fsspec.filesystem("https")
    with fs.open(file_path, 'r') as f:
        for line in f:
            record = parse_focal_mechanism_line(line)
            if record:
                records.append(record)

    return pd.DataFrame(records)


# %%
columns_to_keep = [
    'event_id',
    'time',
    'latitude',
    'longitude',
    'depth_km',
    'magnitude',
    'strike',
    'dip',
    'rake',
    'num_first_motions',
    'first_motion_misfit',
    'num_sp_ratios',
    'sp_ratio_misfit',
    'plane1_uncertainty',
    'plane2_uncertainty',
    'quality',
]


def process_focal_mechanism_file(input_file):
    """Process a single focal mechanism file and return list of (year, jday, df) tuples."""
    file_path = f"{input_url}/{input_file}"
    df = parse_focal_mechanism_file(file_path)
    if len(df) == 0:
        return []

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])  # Drop rows with invalid time
    if len(df) == 0:
        return []

    df["year"] = df["time"].dt.strftime("%Y")
    df["jday"] = df["time"].dt.strftime("%j")
    df['time'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

    results = []
    for (year, jday), group_df in df.groupby(["year", "jday"]):
        if len(group_df) > 0:
            results.append((year, jday, group_df[columns_to_keep]))
    return results


# %%
if __name__ == "__main__":
    output_fs = fsspec.filesystem(output_protocol, token=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))

    print(f"Processing {len(input_files)} focal mechanism files")

    # Process files in parallel
    all_results = {}  # (year, jday) -> list of dataframes
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_focal_mechanism_file, f): f for f in input_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing"):
            for year, jday, df in future.result():
                key = (year, jday)
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append(df)

    # Merge and save locally
    local_files = []
    for (year, jday), dfs in tqdm(all_results.items(), desc="Saving"):
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['event_id'])
        local_path = f"{result_path}/{year}/{jday}"
        os.makedirs(local_path, exist_ok=True)
        csv_path = f"{local_path}/focal_mechanisms.csv"
        merged.to_csv(csv_path, index=False)
        local_files.append((csv_path, f"{output_bucket}/{output_folder}/{year}/{jday}/focal_mechanisms.csv"))

    # Parallel upload
    def upload_file(args):
        local, remote = args
        output_fs.put(local, remote)

    print(f"Uploading {len(local_files)} files...")
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(tqdm(executor.map(upload_file, local_files), total=len(local_files), desc="Uploading"))

# %%
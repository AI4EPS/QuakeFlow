# %%
import os
import sys

import fsspec
import pandas as pd
from tqdm import tqdm

input_protocol = "s3"
input_bucket = "ncedc-pds"
input_folder = "mechanism"

output_protocol = "gs"
output_bucket = "quakeflow_dataset"
output_folder = "NC/catalog"

result_path = "dataset"
os.makedirs(result_path, exist_ok=True)

# %%
# NCSN.MECH format description (Y2K compliant HYPO71 + fault plane solution)
# See: Reasenberg & Oppenheimer, 1985, FPFIT, FPPLOT, and FPPAGE
# https://ncedc.org/pub/doc/cat5/ncsn.mech.txt

def safe_float(s):
    """Convert string to float, return None if empty or invalid."""
    s = s.strip()
    return float(s) if s else None


def safe_int(s):
    """Convert string to int, return None if empty or invalid."""
    s = s.strip()
    return int(s) if s else None


def parse_focal_mechanism_line(line):
    """Parse a single line of focal mechanism data according to NCSN.MECH format."""
    line = line.ljust(142)  # Pad line to expected length

    try:
        # Time and location (HYPO71 summary card format)
        year = int(line[0:4])
        month = int(line[4:6])
        day = int(line[6:8])
        hour = int(line[9:11])
        minute = int(line[11:13])
        second = float(line[13:19])

        lat_deg = float(line[19:22])
        lat_hem = line[22:23].strip()
        lat_min = float(line[24:28])

        lon_deg = float(line[28:32])
        lon_hem = line[32:33].strip()
        lon_min = float(line[33:38])

        depth = float(line[38:45])
        magnitude = float(line[47:52])
        num_ps_times = safe_int(line[52:55])
        azimuthal_gap = safe_float(line[55:59])
        nearest_station_dist = safe_float(line[59:64])
        rms_residual = safe_float(line[64:69])
        horizontal_error = safe_float(line[69:74])
        vertical_error = safe_float(line[74:79])
        magnitude_type = line[81:82].strip() or None

        # Fault plane solution parameters
        dip_direction = safe_float(line[83:86])
        dip_angle = safe_float(line[87:89])
        rake = safe_float(line[89:93])
        first_motion_misfit = safe_float(line[95:99])
        num_first_motions = safe_int(line[100:103])
        misfit_uncertainty = safe_float(line[104:109])
        station_distribution_ratio = safe_float(line[110:114])
        strike_uncertainty = safe_float(line[121:123])
        dip_uncertainty = safe_float(line[124:126])
        rake_uncertainty = safe_float(line[127:129])
        convergence_flag = line[129:130].strip() or None
        multiple_solution_flag = line[130:131].strip() or None
        event_id = line[131:141].strip() or None

        # Convert to decimal degrees
        latitude = round(lat_deg + lat_min / 60.0, 4)
        if lat_hem == 'S':
            latitude = -latitude

        longitude = round(lon_deg + lon_min / 60.0, 4)
        if lon_hem != 'E':  # Blank means West
            longitude = -longitude

        time_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:05.2f}"

        # Calculate strike from dip direction (strike = dip_direction - 90)
        strike = (dip_direction - 90) % 360 if dip_direction is not None else None

        return {
            'event_id': f"nc{event_id}" if event_id else None,
            'time': time_str,
            'latitude': latitude,
            'longitude': longitude,
            'depth_km': depth,
            'magnitude': magnitude,
            'magnitude_type': magnitude_type,
            'num_ps_times': num_ps_times,
            'azimuthal_gap': azimuthal_gap,
            'nearest_station_km': nearest_station_dist,
            'rms_residual': rms_residual,
            'horizontal_error_km': horizontal_error,
            'vertical_error_km': vertical_error,
            'strike': strike,
            'dip': dip_angle,
            'rake': rake,
            'dip_direction': dip_direction,
            'first_motion_misfit': first_motion_misfit,
            'num_first_motions': num_first_motions,
            'misfit_uncertainty': misfit_uncertainty,
            'station_distribution_ratio': station_distribution_ratio,
            'strike_uncertainty': strike_uncertainty,
            'dip_uncertainty': dip_uncertainty,
            'rake_uncertainty': rake_uncertainty,
            'convergence_flag': convergence_flag,
            'multiple_solution_flag': multiple_solution_flag,
        }
    except (ValueError, IndexError) as e:
        print(f"Error parsing line: {line}")
        print(f"Error: {e}")
        return None


def parse_mech_file(fs, file_path):
    """Parse a focal mechanism file and return a DataFrame."""
    records = []

    with fs.open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            record = parse_focal_mechanism_line(line)
            if record:
                records.append(record)

    return pd.DataFrame(records)


# %%
input_fs = fsspec.filesystem(input_protocol, anon=True)
# mech_files = sorted(input_fs.glob(f"{input_bucket}/{input_folder}/????/????.??.mech"), reverse=True)
## FIXME: HARD CODED FOR TESTING
mech_files = [f"{input_bucket}/{input_folder}/2023/2023.01.mech"]
output_fs = fsspec.filesystem(output_protocol, token=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))

print(f"Found {len(mech_files)} focal mechanism files")

# %%
columns_to_keep = [
    'event_id',
    'time',
    'latitude',
    'longitude',
    'depth_km',
    'magnitude',
    'magnitude_type',
    'strike',
    'dip',
    'rake',
    'first_motion_misfit',
    'num_first_motions',
    'strike_uncertainty',
    'dip_uncertainty',
    'rake_uncertainty',
]

for mech_file in tqdm(mech_files):
    print(f"Processing {mech_file}")

    df = parse_mech_file(input_fs, mech_file)

    if len(df) == 0:
        continue

    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.strftime("%Y")
    df["jday"] = df["time"].dt.strftime("%j")
    df['time'] = df['time'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%f'))

    for (year, jday), group_df in tqdm(df.groupby(["year", "jday"]), leave=False):
        if len(group_df) == 0:
            continue
        os.makedirs(f"{result_path}/{year}/{jday}", exist_ok=True)

        group_df = group_df[columns_to_keep]
        group_df.to_csv(f"{result_path}/{year}/{jday}/focal_mechanisms.csv", index=False)
        output_fs.put(
            f"{result_path}/{year}/{jday}/focal_mechanisms.csv",
            f"{output_bucket}/{output_folder}/{year}/{jday}/focal_mechanisms.csv",
        )

# %%

# %%
import os

import fsspec
import pandas as pd
from tqdm import tqdm

input_protocol = "s3"
input_bucket = "scedc-pds"
input_folder = "earthquake_catalogs/SCEC_DC"

output_protocol = "gs"
output_bucket = "quakeflow_dataset"
output_folder = "SCEDC/catalog"

result_path = "dataset"
os.makedirs(result_path, exist_ok=True)

# Create filesystem objects once
input_fs = fsspec.filesystem(input_protocol, anon=True)
output_fs = fsspec.filesystem(output_protocol, token=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))


# %%
# Parse fixed-width fields according to SCEDC format
# date = line[0:10].strip()           # YYYY/MM/DD
# time = line[11:22].strip()          # HH:mm:SS.ss
# event_type = line[23:25].strip()    # event type (eq, qb, sn, nt, uk)
# geo_type = line[27:28].strip()      # geographical type (l, r, t)
# magnitude = line[29:33].strip()     # magnitude
# mag_type = line[34:35].strip()      # magnitude type
# latitude = line[39:45].strip()      # latitude
# longitude = line[46:54].strip()     # longitude
# depth = line[55:60].strip()         # depth
# quality = line[61:62].strip()       # location quality
# event_id = line[63:71].strip()      # event ID
# nph = line[73:76].strip()           # number of picked phases
# ngrams = line[77:81].strip()        # number of grams

# %%
# Standardized location quality mapping
LOCATION_QUALITY_MAP = {
    'A': 'excellent',
    'B': 'good',
    'C': 'fair',
    'D': 'poor',
}

# Standardized geographical type mapping
GEOGRAPHICAL_TYPE_MAP = {
    'l': 'local',
    'r': 'regional',
    't': 'teleseismic',
}

# Standardized magnitude type mapping (consistent with NCEDC)
MAGNITUDE_TYPE_MAP = {
    'a': 'amplitude',
    'b': 'body_wave',
    'd': 'duration',
    'dl': 'low_gain_amplitude',
    'e': 'energy',
    'h': 'human_assigned',
    'l': 'local',
    'n': 'none',
    'un': 'unknown',
    'w': 'moment',
}

# Standardized event type mapping (consistent with NCEDC)
EVENT_TYPE_MAP = {
    'bc': 'building_collapse',
    'eq': 'earthquake',
    'ex': 'chemical_explosion',
    'lp': 'long_period',
    'ls': 'landslide',
    'mi': 'meteor_impact',
    'nt': 'nuclear_test',
    'ot': 'other',
    'qb': 'quarry_blast',
    'rs': 'rockslide',
    'sh': 'survey_shot',
    'sn': 'sonic_boom',
    'st': 'subnet_trigger',
    'th': 'thunder',
    'uk': 'unknown',
}


def parse_scedc_catalog_line(line):
    """Parse a single line from SCEDC catalog."""
    if line.startswith('#') or len(line.strip()) == 0:
        return None

    tokens = line.strip().split()
    if len(tokens) != 13:
        print(f"Unexpected number of tokens ({len(tokens)} != 13) in line: {line.strip()}")
    date, time, event_type, geo_type, magnitude, mag_type, latitude, longitude, depth, quality, event_id, nph, ngrams = tokens

    datetime_str = f"{date} {time}"

    return {
        'event_id': event_id,
        'time': datetime_str,
        'latitude': float(latitude) if latitude else None,
        'longitude': float(longitude) if longitude else None,
        'depth_km': float(depth) if depth else None,
        'magnitude': float(magnitude) if magnitude else None,
        'magnitude_type': mag_type,
        'event_type': event_type,
        'geographical_type': geo_type,
        'location_quality': quality,
        'num_phases': int(nph) if nph else None,
        'num_grams': int(ngrams) if ngrams else None,
    }


def map_column_names(df):
    """Map SCEDC column names to standard format and normalize values."""
    # Standardize categorical values to descriptive names
    if 'location_quality' in df.columns:
        df['location_quality'] = df['location_quality'].map(LOCATION_QUALITY_MAP).fillna(df['location_quality'])
    if 'geographical_type' in df.columns:
        df['geographical_type'] = df['geographical_type'].map(GEOGRAPHICAL_TYPE_MAP).fillna(df['geographical_type'])
    if 'magnitude_type' in df.columns:
        df['magnitude_type'] = df['magnitude_type'].map(MAGNITUDE_TYPE_MAP).fillna(df['magnitude_type'])
    if 'event_type' in df.columns:
        df['event_type'] = df['event_type'].map(EVENT_TYPE_MAP).fillna(df['event_type'])

    return df

# %%
catalog_files = sorted(input_fs.glob(f"{input_bucket}/{input_folder}/*.catalog"), reverse=True)

# %%
columns_to_keep = [
    # Core fields (consistent with NCEDC)
    'event_id',
    'time',
    'latitude',
    'longitude',
    'depth_km',
    'magnitude',
    'magnitude_type',
    'event_type',
    # SCEDC-specific fields
    'location_quality',
    'geographical_type',
    'num_phases',
    'num_grams',
]

def process_catalog_file(catalog_file):
    """Process a single catalog file and save events by year/jday (local only)."""
    print(f"Processing {catalog_file}")

    with input_fs.open(f"{catalog_file}", 'r') as f:
        lines = f.readlines()

    parsed_data = [parse_scedc_catalog_line(line) for line in lines]
    parsed_data = [p for p in parsed_data if p is not None]

    if not parsed_data:
        print(f"No valid data found in {catalog_file}")
        return None

    df = pd.DataFrame(parsed_data)
    df = map_column_names(df)

    # Vectorized datetime operations
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.strftime("%Y")
    df["jday"] = df["time"].dt.strftime("%j")
    df["time"] = df["time"].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    df['event_id'] = "ci" + df['event_id'].astype(str)

    for (year, jday), group_df in df.groupby(["year", "jday"]):
        if len(group_df) == 0:
            continue
        os.makedirs(f"{result_path}/{year}/{jday}", exist_ok=True)
        group_df[columns_to_keep].to_csv(f"{result_path}/{year}/{jday}/events.csv", index=False)
    return catalog_file


if __name__ == "__main__":
    # Process all files locally first
    for catalog_file in tqdm(catalog_files, desc="Processing catalogs"):
        process_catalog_file(catalog_file)

    # Upload year directories to GCS
    print("Uploading to GCS...")
    years = sorted(os.listdir(result_path))
    for year in tqdm(years, desc="Uploading"):
        year_path = f"{result_path}/{year}"
        if os.path.isdir(year_path):
            output_fs.put(year_path, f"{output_bucket}/{output_folder}/{year}", recursive=True)
    print("Done!")

# %%


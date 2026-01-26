# %%
import os

import fsspec
import pandas as pd
from tqdm import tqdm

input_protocol = "s3"
input_bucket = "ncedc-pds"
input_folder = "earthquake_catalogs/NCEDC"

output_protocol = "gs"
output_bucket = "quakeflow_dataset"
output_folder = "NCEDC/catalog"

result_path = "dataset"
os.makedirs(result_path, exist_ok=True)

# Create filesystem objects once
input_fs = fsspec.filesystem(input_protocol, anon=True)
output_fs = fsspec.filesystem(output_protocol, token=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))

# %%
# status: (Event status)
#       A: Automatic
#       F: Finalized
#       H: Human Reviewed
#       I: Intermediate

# magType: (Magnitude Type)
#       a : Primary amplitude magnitude (Jerry Eaton's XMAG)
#       b : Body-wave magnitude
#       d : Duration magnitude
#       dl: Low-gain initial P-wave amplitude magnitude
#       e : Energy magnitude
#       h : Human assigned magnitude
#       l : Local magnitude
#       n : No magnitude
#       un: Unknown magnitude type
#       w : Moment magnitude

# type: (EventType)
#       bc: Building collapse/demolition
#       eq: Earthquake
#       ex: Generic chemical blast
#       lp: Long period volcanic earthquake
#       ls: Landslide
#       mi: Meteor/comet impact
#       nt: Nuclear test
#       ot: Other miscellaneous
#       qb: Quarry blast
#       rs: Rockslide
#       sh: Refraction/reflection survey shot
#       sn: Sonic shockwave
#       st: Subnet trigger
#       th: Thunder
#       uk: Unknown type

# %%
# Standardized review status mapping
REVIEW_STATUS_MAP = {
    'A': 'automatic',
    'F': 'finalized',
    'H': 'human_reviewed',
    'I': 'intermediate',
}

# Standardized magnitude type mapping (consistent with SCEDC)
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

# Standardized event type mapping (consistent with SCEDC)
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

def map_column_names(df):
    column_mapping = {
        'id': 'event_id',
        'time': 'time',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'depth': 'depth_km',
        'mag': 'magnitude',
        'magType': 'magnitude_type',
        'type': 'event_type',
        'gap': 'azimuthal_gap_deg',
        'dmin': 'min_station_distance_deg',
        'rms': 'rms_residual_sec',
        'horizontalError': 'horizontal_error_km',
        'depthError': 'depth_error_km',
        'status': 'review_status',
        'nst': 'num_stations',
        'net': 'network',
        'updated': 'updated_time',
        'place': 'place',
        'magError': 'magnitude_error',
        'magNst': 'num_magnitude_stations',
        'locationSource': 'location_source',
        'magSource': 'magnitude_source'
    }

    existing_columns = {col: column_mapping[col] for col in df.columns if col in column_mapping}
    df = df.rename(columns=existing_columns)

    # Standardize categorical values to descriptive names
    if 'review_status' in df.columns:
        df['review_status'] = df['review_status'].map(REVIEW_STATUS_MAP).fillna(df['review_status'])
    if 'magnitude_type' in df.columns:
        df['magnitude_type'] = df['magnitude_type'].map(MAGNITUDE_TYPE_MAP).fillna(df['magnitude_type'])
    if 'event_type' in df.columns:
        df['event_type'] = df['event_type'].map(EVENT_TYPE_MAP).fillna(df['event_type'])

    return df

# %%
csv_files = sorted(input_fs.glob(f"{input_bucket}/{input_folder}/*.ehpcsv"), reverse=True)

# %%
columns_to_keep = [
    # Core fields (consistent with SCEDC)
    'event_id',
    'time',
    'latitude',
    'longitude',
    'depth_km',
    'magnitude',
    'magnitude_type',
    'event_type',
    # NCEDC-specific fields
    'review_status',
    'num_stations',
    'azimuthal_gap_deg',
    'min_station_distance_deg',
    'rms_residual_sec',
    'horizontal_error_km',
    'depth_error_km',
]

def process_catalog_file(csv_file):
    """Process a single catalog file and save events by year/jday (local only)."""
    print(f"Processing {csv_file}")

    df = pd.read_csv(f"{input_protocol}://{csv_file}", dtype=str, encoding='latin-1', storage_options={"anon": True})
    df = map_column_names(df)

    # Vectorized datetime operations
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.strftime("%Y")
    df["jday"] = df["time"].dt.strftime("%j")
    df['time'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    df['event_id'] = "nc" + df['event_id'].astype(str)

    for (year, jday), group_df in df.groupby(["year", "jday"]):
        if len(group_df) == 0:
            continue
        os.makedirs(f"{result_path}/{year}/{jday}", exist_ok=True)
        group_df[columns_to_keep].to_csv(f"{result_path}/{year}/{jday}/events.csv", index=False)
    return csv_file


if __name__ == "__main__":
    # Process all files locally first
    for csv_file in tqdm(csv_files, desc="Processing catalogs"):
        process_catalog_file(csv_file)

    # Upload year directories to GCS
    print("Uploading to GCS...")
    years = sorted(os.listdir(result_path))
    for year in tqdm(years, desc="Uploading"):
        year_path = f"{result_path}/{year}"
        if os.path.isdir(year_path):
            output_fs.put(year_path, f"{output_bucket}/{output_folder}/{year}", recursive=True)
    print("Done!")

    
# %%


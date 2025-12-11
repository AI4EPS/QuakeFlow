# %%
import os
import sys

import fsspec
import pandas as pd
from tqdm import tqdm

input_protocol = "s3"
input_bucket = "ncedc-pds"
input_folder = "earthquake_catalogs/NCEDC"

output_protocol = "gs"
output_bucket = "quakeflow_dataset"
output_folder = "NC/catalog"

result_path = "dataset"
os.makedirs(result_path, exist_ok=True)

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
        'gap': 'azimuthal_gap',
        'dmin': 'minimum_distance_km',
        'rms': 'time_residual',
        'horizontalError': 'horizontal_error_km',
        'depthError': 'depth_error_km', 
        'status': 'review_status',
        'nst': 'num_stations',
        'net': 'network',
        'updated': 'updated_time',
        'place': 'place',
        'magError': 'magnitude_error',
        'magNst': 'magnitude_num_stations',
        'locationSource': 'location_source',
        'magSource': 'magnitude_source'
    }
    
    # Rename columns that exist in the dataframe
    existing_columns = {col: column_mapping[col] for col in df.columns if col in column_mapping}
    df = df.rename(columns=existing_columns)
    
    return df

# %%
input_fs = fsspec.filesystem(input_protocol, anon=True)
csv_files = sorted(input_fs.glob(f"{input_bucket}/{input_folder}/*.ehpcsv"), reverse=True)
output_fs = fsspec.filesystem(output_protocol, token=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))

# %%
columns_to_keep = [
        'event_id',
        'time',
        'latitude', 
        'longitude',
        'depth_km',
        'magnitude',
        'magnitude_type',
        'event_type',
        'azimuthal_gap',
        'minimum_distance_km',
        'time_residual',
        'horizontal_error_km',
        'depth_error_km', 
        'review_status',
]

## FIXME: HARD CODED FOR TESTING
csv_files = ["ncedc-pds/earthquake_catalogs/NCEDC/2023.ehpcsv"]

for csv_file in tqdm(csv_files):
    print(f"Processing {csv_file}")

    df = pd.read_csv(f"{input_protocol}://{csv_file}", dtype=str, encoding='latin-1')
    df = map_column_names(df)

    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.strftime("%Y")
    df["jday"] = df["time"].dt.strftime("%j")
    df['time'] = df['time'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%f'))
    df['event_id'] = df['event_id'].apply(lambda x: "nc" + x)


    for (year, jday), group_df in tqdm(df.groupby(["year", "jday"])):
        if len(group_df) == 0:
            continue
        os.makedirs(f"{result_path}/{year}/{jday}", exist_ok=True)

        group_df = group_df[columns_to_keep]
        group_df.to_csv(f"{result_path}/{year}/{jday}/events.csv", index=False)
        output_fs.put(
            f"{result_path}/{year}/{jday}/events.csv",
            f"{output_bucket}/{output_folder}/{year}/{jday}/events.csv",
        )

    
# %%


# %%
import fsspec   
import pandas as pd
from tqdm import tqdm
import os

input_protocol = "s3"
input_bucket = "scedc-pds"
input_folder = "earthquake_catalogs/SCEC_DC"

output_protocol = "gs"
output_bucket = "quakeflow_dataset"
output_folder = "SC/catalog"

result_path = "dataset"
os.makedirs(result_path, exist_ok=True)

# %%
# Parse SCEDC fixed-width catalog format
def parse_scedc_catalog_line(line):
    """
    Parse a single line from SCEDC catalog in fixed-width format
    """
    if line.startswith('#') or len(line.strip()) == 0:
        return None
    
    # try:
    # Parse fixed-width fields according to SCEDC format
    date = line[0:10].strip()           # YYYY/MM/DD
    time = line[11:22].strip()          # HH:mm:SS.ss
    event_type = line[23:25].strip()    # event type (eq, qb, sn, nt, uk)
    geo_type = line[27:28].strip()      # geographical type (l, r, t)
    magnitude = line[29:33].strip()     # magnitude
    mag_type = line[34:35].strip()      # magnitude type
    latitude = line[39:45].strip()      # latitude
    longitude = line[46:54].strip()     # longitude
    depth = line[55:60].strip()         # depth
    quality = line[61:62].strip()       # location quality
    event_id = line[63:71].strip()      # event ID
    nph = line[73:76].strip()           # number of picked phases
    ngrams = line[77:81].strip()        # number of grams
    
    # Convert to proper data types
    magnitude = float(magnitude) if magnitude else None
    latitude = float(latitude) if latitude else None
    longitude = float(longitude) if longitude else None
    depth = float(depth) if depth else None
    nph = int(nph) if nph else None
    ngrams = int(ngrams) if ngrams else None
    
    # Combine date and time
    datetime_str = f"{date} {time}"
        
    return {
        'event_id': event_id,
        'time': datetime_str,
        'latitude': latitude,
        'longitude': longitude,
        'depth_km': depth,
        'magnitude': magnitude,
        'magnitude_type': mag_type,
        'event_type': event_type,
        'geographical_type': geo_type,
        'location_quality': quality,
        'num_phases': nph,
        'num_grams': ngrams
    }
    # except Exception as e:
    #     print(f"Error parsing line: {line.strip()}, Error: {e}")
    #     return None

def map_column_names(df):
    """
    Map SCEDC column names to standard format
    """
    column_mapping = {
        'event_id': 'event_id',
        'time': 'time',
        'latitude': 'latitude', 
        'longitude': 'longitude',
        'depth_km': 'depth_km',
        'magnitude': 'magnitude',
        'magnitude_type': 'magnitude_type',
        'event_type': 'event_type',
        'geographical_type': 'geographical_type',
        'location_quality': 'location_quality',
        'num_phases': 'num_phases',
        'num_grams': 'num_grams'
    }
    
    # Rename columns that exist in the dataframe
    existing_columns = {col: column_mapping[col] for col in df.columns if col in column_mapping}
    df = df.rename(columns=existing_columns)
    
    return df

# %%
input_fs = fsspec.filesystem(input_protocol, anon=True)
output_fs = fsspec.filesystem(output_protocol, token=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))
catalog_files = sorted(input_fs.glob(f"{input_bucket}/{input_folder}/*.catalog"), reverse=True)

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
    'geographical_type',
    'location_quality',
    'num_phases',
    'num_grams'
]

for catalog_file in tqdm(catalog_files):

    with input_fs.open(f"{catalog_file}", 'r') as f:
        lines = f.readlines()

    parsed_data = []
    for line in lines:
        parsed_line = parse_scedc_catalog_line(line)
        if parsed_line:
            parsed_data.append(parsed_line)
    
    if not parsed_data:
        print(f"No valid data found in {catalog_file}")
        continue
    
    df = pd.DataFrame(parsed_data)
    df = map_column_names(df)

    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.strftime("%Y")
    df["jday"] = df["time"].dt.strftime("%j")
    df["time"] = df["time"].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.%f'))
    df['event_id'] = df['event_id'].apply(lambda x: "sc" + x)

    for (year, jday), df in df.groupby(["year", "jday"]):
        if len(df) == 0:
            continue
        os.makedirs(f"{result_path}/{year}/{jday}", exist_ok=True)

        df = df[columns_to_keep]
        df.to_csv(f"{result_path}/{year}/{jday}/events.csv", index=False)
        output_fs.put(
            f"{result_path}/{year}/{jday}/events.csv",
            f"{output_bucket}/{output_folder}/{year}/{jday}/events.csv",
        )

    if year <= "2024":
        break

# %%


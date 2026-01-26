# %%
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
input_protocol = "s3"
input_bucket = "scedc-pds"
input_folder = "event_phases"
input_fs = fsspec.filesystem(input_protocol, anon=True)

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset"
output_folder = "SCEDC/catalog"
output_fs = fsspec.filesystem(output_protocol, token=output_token)


result_path = "catalog"
os.makedirs(result_path, exist_ok=True)

# %%
## https://scedc.caltech.edu/data/stp/STP_Manual_v1.01.pdf
# Defining a function to parse event location information
parse_event_time = lambda x: (
    datetime.strptime(":".join(x.split(":")[:-1]), "%Y/%m/%d,%H:%M") + timedelta(seconds=float(x.split(":")[-1]))
)

def parse_event_line(line):
    fields = line.split()
    try:
        event_info = {
            "event_id": "ci" + fields[0],
            "event_type": fields[1],
            # "event_time": parse_event_time(fields[3]),
            "time": parse_event_time(fields[3]),
            "latitude": float(fields[4]),
            "longitude": float(fields[5]),
            "depth_km": float(fields[6]),
            "magnitude": float(fields[7]),
            "magnitude_type": fields[8],
            "quality": float(fields[9]),
        }
    except:
        event_info = {
            "event_id": fields[0],
            "event_type": fields[1],
            # "date": fields[2],
            # "event_time": parse_event_time(fields[2]),
            "time": parse_event_time(fields[2]),
            "latitude": float(fields[3]),
            "longitude": float(fields[4]),
            "depth_km": float(fields[5]),
            "magnitude": float(fields[6]),
            "magnitude_type": fields[7],
            "quality": float(fields[8]),
        }

    return event_info


def calc_azimuth_vectorized(lat1, lon1, lat2, lon2):
    """Calculate azimuth from point 1 to point 2 in degrees (vectorized)."""
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    az = np.degrees(np.arctan2(x, y))
    return np.round((az + 360) % 360, 2)


def parse_phase_line(line, event_id, event_time, event_lat, event_lon):
    fields = line.split()
    polarity = fields[8][0]
    if polarity == ".":
        polarity = ""
    elif polarity in ("c", "+", "u"):
        polarity = "U"
    elif polarity in ("d", "-", "r"):
        polarity = "D"
    else:
        polarity = ""
    return {
        "network": fields[0],
        "station": fields[1],
        "channel": fields[2],
        "instrument": fields[2][:-1],
        "component": fields[2][-1],
        "location": fields[3] if fields[3] != "--" else "",
        "sta_lat": float(fields[4]),
        "sta_lon": float(fields[5]),
        "event_lat": event_lat,
        "event_lon": event_lon,
        "elevation_m": float(fields[6]),
        "depth_km": -round(float(fields[6]) / 1000, 3),
        "phase_type": fields[7],
        "phase_polarity": polarity,
        "phase_remark": fields[9],
        "phase_score": float(fields[10]),
        "distance_km": float(fields[11]),
        "phase_time": event_time + timedelta(seconds=float(fields[12])),
        "event_id": event_id,
    }


def read_phase_file(file, input_fs):
    """Read and parse a single phase file. Returns (event, phases_list) or None."""
    try:
        with input_fs.open(file, "r") as fp:
            lines = fp.readlines()

        event_line = lines[0].strip()
        if event_line.startswith("0       1970/01/01,00:00:00.000"):
            return None

        event = parse_event_line(event_line)
        phases_ = [
            parse_phase_line(line.strip(), event["event_id"], event["time"], event["latitude"], event["longitude"])
            for line in lines[1:]
        ]
        if len(phases_) == 0:
            return None

        return (event, phases_)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return None


# %%
def process(jday):

    input_fs = fsspec.filesystem(input_protocol, anon=True)
    output_fs = fsspec.filesystem(output_protocol, token=output_token)

    # Get list of phase files
    phase_files = list(input_fs.glob(f"{jday}/*.phase"))

    # Read files in parallel using ThreadPoolExecutor
    events = []
    all_phases = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(lambda f: read_phase_file(f, input_fs), phase_files))

    for result in results:
        if result is not None:
            event, phases_ = result
            events.append(event)
            all_phases.extend(phases_)

    # %% save all events
    phase_columns = [
        "event_id",
        "network",
        "station",
        "location",
        "instrument",
        "component",
        "phase_type",
        "phase_time",
        "phase_score",
        "phase_polarity",
        "phase_remark",
        "distance_km",
        "azimuth",
        "back_azimuth",
    ]
    event_columns = [
        "event_id",
        "time",
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
        "magnitude_type",
        "event_type",
        "quality",
    ]

    if len(events) == 0:
        return None
    
    events = pd.DataFrame(events)
    events = events[event_columns]
    events["time"] = pd.to_datetime(events["time"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    phases = pd.DataFrame(all_phases)
    # Vectorized azimuth calculation
    phases["azimuth"] = calc_azimuth_vectorized(
        phases["event_lat"].values, phases["event_lon"].values,
        phases["sta_lat"].values, phases["sta_lon"].values
    )
    phases["back_azimuth"] = calc_azimuth_vectorized(
        phases["sta_lat"].values, phases["sta_lon"].values,
        phases["event_lat"].values, phases["event_lon"].values
    )
    phases = phases[phase_columns]
    phases["phase_time"] = pd.to_datetime(phases["phase_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    phases["phase_remark"] = phases["phase_remark"].replace(".", "")

    year, jday = jday.split('/')[-1].split('_')

    os.makedirs(f"{result_path}/{year}/{jday}", exist_ok=True)

    # events.to_csv(f"{result_path}/event/{year}/{jday}.csv", index=False)
    # output_fs.put(
    #     f"{result_path}/event/{year}/{jday}.csv",
    #     f"{output_bucket}/{output_folder}/event/{year}/{jday}.csv",
    # )
    phases.to_csv(f"{result_path}/{year}/{jday}/phases.csv", index=False)
    output_fs.put(
        f"{result_path}/{year}/{jday}/phases.csv",
        f"{output_bucket}/{output_folder}/{year}/{jday}/phases.csv",
    )

    # %% save picks with P/S pairs - keep best score per event/station/phase_type
    phases_best = phases.loc[phases.groupby(["event_id", "network", "station", "phase_type"])["phase_score"].idxmax()]

    # Find groups that have both P and S
    group_cols = ["event_id", "network", "station"]
    phase_counts = phases_best.groupby(group_cols + ["phase_type"]).size().unstack(fill_value=0)
    has_both = phase_counts.index[(phase_counts.get("P", 0) > 0) & (phase_counts.get("S", 0) > 0)]

    if len(has_both) == 0:
        return None

    phases_ps = phases_best.set_index(group_cols).loc[has_both].reset_index()
    phases_ps = phases_ps[phase_columns]

    phases_ps.to_csv(f"{result_path}/{year}/{jday}/phases_ps.csv", index=False)
    output_fs.put(
        f"{result_path}/{year}/{jday}/phases_ps.csv",
        f"{output_bucket}/{output_folder}/{year}/{jday}/phases_ps.csv",
    )


# %%
if __name__ == "__main__":
    file_list = []
    for year in tqdm(sorted(input_fs.glob(f"{input_bucket}/{input_folder}/????"), reverse=True), desc="Scanning phase files"):
        if year.endswith("done"):
            continue

        for jday in sorted(input_fs.glob(f"{year}/????_???"), reverse=False):
            file_list.append(jday)

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process, jday) for jday in file_list]
        for future in tqdm(as_completed(futures), total=len(file_list), desc="Processing phase files"):
            result = future.result()
            if result is not None:
                print(result)
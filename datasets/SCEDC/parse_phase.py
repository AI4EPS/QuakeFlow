# %%
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import fsspec
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
output_folder = "SC/catalog"
output_fs = fsspec.filesystem(output_protocol, token=output_token)


result_path = "dataset"
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


def parse_phase_line(line, event_id, event_time):
    fields = line.split()
    phase_pick = {
        "network": fields[0],
        "station": fields[1],
        "channel": fields[2],
        "instrument": fields[2][:-1],
        "component": fields[2][-1],
        "location": fields[3] if fields[3] != "--" else "",
        "latitude": float(fields[4]),
        "longitude": float(fields[5]),
        "elevation_m": float(fields[6]),
        "depth_km": -round(float(fields[6]) / 1000, 3),
        "phase_type": fields[7],
        "phase_polarity": fields[8],
        # "signal onset quality": fields[9],
        "phase_remark": fields[9],
        "phase_score": float(fields[10]),
        "distance_km": float(fields[11]),
        "phase_time": (event_time + timedelta(seconds=float(fields[12]))),
        "event_id": event_id,
    }
    if phase_pick["phase_polarity"][0] == ".":
        phase_pick["phase_polarity"] = ""
    elif phase_pick["phase_polarity"][0] in ["c", "+", "u"]:
        phase_pick["phase_polarity"] = "U"
    elif phase_pick["phase_polarity"][0] in ["d", "-", "r"]:
        phase_pick["phase_polarity"] = "D"
    else:
        print(f"Unknown polarity: {phase_pick['phase_polarity']}")
        phase_pick["phase_polarity"] = ""
    return phase_pick


# %% 
def process(jday):

    input_fs = fsspec.filesystem(input_protocol, anon=True)
    output_fs = fsspec.filesystem(output_protocol, token=output_token)

    events = []
    phases = []

    for file in input_fs.glob(f"{jday}/*.phase"):

        with input_fs.open(file, "r") as fp:
            lines = fp.readlines()
        
            event_line = lines[0].strip()
            nan_case = "0       1970/01/01,00:00:00.000"
            if event_line.startswith(nan_case):
                continue

            event = parse_event_line(event_line)
            phases_ = [parse_phase_line(line.strip(), event["event_id"], event["time"]) for line in lines[1:]]
            if len(phases_) == 0:
                continue
        
        events.append(event)
        phases_ = pd.DataFrame(phases_)
        phases.append(phases_)

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
    events["time"] = pd.to_datetime(events["time"])
    # events["time"] = events["time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00")
    events["time"] = events["time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f"))


    phases = pd.concat(phases)
    phases = phases.reset_index(drop=True)
    phases = phases[phase_columns]
    phases["phase_time"] = pd.to_datetime(phases["phase_time"])
    phases["phase_time"] = phases["phase_time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    # phases["phase_time"] = phases["phase_time"].apply(lambda x: x + "+00:00")
    phases["phase_remark"] = phases["phase_remark"].apply(lambda x: "" if x == "." else x)

    year, jday = jday.split('/')[-1].split('_')

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

    # %% save picks with P/S pairs
    phases_ps = []
    phases = phases.loc[phases.groupby(["event_id", "network", "station", "phase_type"])["phase_score"].idxmax()]
    for (event_id, network, station), picks in phases.groupby(["event_id", "network", "station"]):
        if len(picks) >= 2:
            phase_type = picks["phase_type"].unique()
            if ("P" in phase_type) and ("S" in phase_type):
                phases_ps.append(picks)
        if len(picks) >= 3:
            print(event_id, network, station, len(picks))

    if len(phases_ps) == 0:
        return None

    phases_ps = pd.concat(phases_ps)
    phases_ps = phases_ps[phase_columns]

    phases_ps.to_csv(f"{result_path}/{year}/{jday}/phases_ps.csv", index=False)
    output_fs.put(
        f"{result_path}/{year}/{jday}/phases_ps.csv",
        f"{output_bucket}/{output_folder}/{year}/{jday}/phases_ps.csv",
    )


# %%
if __name__ == "__main__":
    file_list = []
    for year in tqdm(sorted(input_fs.glob(f"{input_bucket}/{input_folder}/????"), reverse=True)):
        if year.endswith("done"):
            continue

        for jday in sorted(input_fs.glob(f"{year}/????_???"), reverse=False):
            file_list.append(jday)

    ## FIXME: HARD CODED FOR TESTING
    file_list = ["scedc-pds/event_phases/2024/2024_001"]
    for jday in file_list:
        print(f"Processing {jday}")
        process(jday)
        sys.exit(0)

    ncpu = mp.cpu_count() - 1
    with ProcessPoolExecutor(max_workers=ncpu) as executor:
        futures = [executor.submit(process, file) for file in file_list]
        
        for future in tqdm(as_completed(futures), total=len(file_list)):
            result = future.result()
            if result is not None:
                print(result)
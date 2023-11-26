# %%
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", nargs="?", type=str, default="local", help="root path")
    parser.add_argument("region", nargs="?", type=str, default="demo", help="region")
    parser.add_argument("--dt_cc", action="store_true", help="run convert_dtcc.py")
    return parser.parse_args()


args = parse_args()

# %%
root_path = args.root_path
region = args.region

result_path = f"{region}/hypodd"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")

# %%
############################################# Station Format ######################################################
station_json = f"{region}/obspy/stations.json"
stations = pd.read_json(f"{root_path}/{station_json}", orient="index")

shift_topo = stations["elevation_m"].max() / 1e3
# shift_topo = stations["elevation_m"].max()/1e3 + 3.0
# shift_topo = 0.0 ## prevent air quakes
# shift_topo = 3.0

converted_hypoinverse = []
converted_hypodd = {}

for sta, row in stations.iterrows():
    network_code, station_code, comp_code, channel_code = sta.split(".")
    station_weight = " "
    lat_degree = int(row["latitude"])
    lat_minute = (row["latitude"] - lat_degree) * 60
    north = "N" if lat_degree >= 0 else "S"
    lng_degree = int(row["longitude"])
    lng_minute = (row["longitude"] - lng_degree) * 60
    west = "W" if lng_degree <= 0 else "E"
    elevation = row["elevation_m"]
    line_hypoinverse = f"{station_code:<5} {network_code:<2} {comp_code[:-1]:<1}{channel_code:<3} {station_weight}{abs(lat_degree):2.0f} {abs(lat_minute):7.4f}{north}{abs(lng_degree):3.0f} {abs(lng_minute):7.4f}{west}{elevation:4.0f}\n"
    converted_hypoinverse.append(line_hypoinverse)

    # tmp_code = f"{station_code}{channel_code}"
    tmp_code = f"{station_code}"
    converted_hypodd[tmp_code] = f"{tmp_code:<8s} {row['latitude']:.3f} {row['longitude']:.3f}\n"


with open(f"{root_path}/{result_path}/stations.dat", "w") as f:
    for k, v in converted_hypodd.items():
        f.write(v)


# %%
############################################# Picks Format ######################################################
picks_csv = f"{region}/gamma/gamma_picks.csv"
events_csv = f"{region}/gamma/gamma_events.csv"

picks = pd.read_csv(f"{root_path}/{picks_csv}", parse_dates=["phase_time"])
events = pd.read_csv(f"{root_path}/{events_csv}", parse_dates=["time"])

# events.sort_values("time", inplace=True)
picks = picks.loc[picks["event_index"].isin(events["event_index"])]

lines = []
picks_by_event = picks.groupby("event_index").groups
for i, event in tqdm(events.iterrows(), desc="Convert gamma catalog", total=len(events)):
    # event_time = datetime.strptime(event["time"], "%Y-%m-%dT%H:%M:%S.%f")
    event_time = event["time"]
    lat = event["latitude"]
    lng = event["longitude"]
    # dep = event["depth(m)"] / 1e3 + shift_topo
    dep = event["depth_km"] + shift_topo
    mag = event["magnitude"]
    EH = 0
    EZ = 0
    RMS = event["sigma_time"]

    year, month, day, hour, min, sec = (
        event_time.year,
        event_time.month,
        event_time.day,
        event_time.hour,
        event_time.minute,
        float(event_time.strftime("%S.%f")),
    )
    event_line = f"# {year:4d} {month:2d} {day:2d} {hour:2d} {min:2d} {sec:5.2f}  {lat:7.4f} {lng:9.4f}   {dep:5.2f} {mag:5.2f} {EH:5.2f} {EZ:5.2f} {RMS:5.2f} {event['event_index']:9d}\n"

    lines.append(event_line)

    picks_idx = picks_by_event[event["event_index"]]
    for j in picks_idx:
        # pick = picks.iloc[j]
        pick = picks.loc[j]
        network_code, station_code, comp_code, channel_code = pick["station_id"].split(".")
        phase_type = pick["phase_type"].upper()
        phase_score = pick["phase_score"]
        # pick_time = (datetime.strptime(pick["phase_time"], "%Y-%m-%dT%H:%M:%S.%f") - event_time).total_seconds()
        pick_time = (pick["phase_time"] - event_time).total_seconds()
        tmp_code = f"{station_code}"
        pick_line = f"{tmp_code:<7s}   {pick_time:6.3f}   {phase_score:5.4f}   {phase_type}\n"
        lines.append(pick_line)

with open(f"{root_path}/{result_path}/phase.txt", "w") as fp:
    fp.writelines(lines)

# %%
if args.dt_cc:
    os.system("python convert_dtcc.py")
    os.system("cp templates/dt.cc relocation/hypodd/")

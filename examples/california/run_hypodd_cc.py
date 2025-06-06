# %%
# from args import parse_args
##
import argparse
import json
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gamma on NCEDC/SCEDC data")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--root_path", type=str, default="local")
    parser.add_argument("--region", type=str, default="Cal")
    parser.add_argument("--bucket", type=str, default="quakeflow_catalog")
    return parser.parse_args()


# %%
args = parse_args()
root_path = args.root_path
region = args.region

# with open(f"{root_path}/{region}/config.json", "r") as fp:
#     config = json.load(fp)
config = json.load(open("config.json", "r"))

# %%
data_path = f"{region}/cctorch"
result_path = f"{region}/hypodd"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")

# %%
stations = pd.read_csv(f"{root_path}/{data_path}/cctorch_stations.csv")

station_lines = {}
for i, row in stations.iterrows():
    station_id = row["station_id"]
    network_code, station_code, comp_code, channel_code = station_id.split(".")
    # tmp_code = f"{station_code}{channel_code}"
    tmp_code = f"{station_code}"
    station_lines[tmp_code] = f"{tmp_code:<8s} {row['latitude']:.3f} {row['longitude']:.3f}\n"


with open(f"{root_path}/{result_path}/stations.dat", "w") as f:
    for line in sorted(station_lines.values()):
        f.write(line)

# %%
events = pd.read_csv(f"{root_path}/{data_path}/cctorch_events.csv")
events["time"] = pd.to_datetime(events["event_time"], format="mixed")

event_lines = []

for i, row in events.iterrows():
    event_index = row["event_index"]
    origin = row["time"]
    magnitude = row["magnitude"]
    x_err = 0.0
    z_err = 0.0
    time_err = 0.0
    dx, dy, dz = 0.0, 0.0, 0.0
    # dx = np.random.uniform(-0.01, 0.01)
    # dy = np.random.uniform(-0.01, 0.01)
    # dz = np.random.uniform(0, 10)
    # dz = 0
    event_lines.append(
        f"{origin.year:4d}{origin.month:02d}{origin.day:02d}  "
        f"{origin.hour:2d}{origin.minute:02d}{origin.second:02d}{round(origin.microsecond / 1e4):02d}  "
        # f"{row['latitude']:8.4f}  {row['longitude']:9.4f}   {row['depth_km']:8.4f}  "
        f"{row['latitude'] + dy:8.4f}  {row['longitude']+ dx:9.4f}   {row['depth_km']+dz:8.4f}  "
        f"{magnitude:5.2f}  {x_err:5.2f}  {z_err:5.2f}  {time_err:5.2f}  {event_index:9d}\n"
    )

with open(f"{root_path}/{result_path}/events.dat", "w") as f:
    f.writelines(event_lines)

# %%
os.system(f"bash run_hypodd_cc.sh {root_path} {region}")

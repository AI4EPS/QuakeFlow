# %%
import json
import os
import pickle

import numpy as np
import pandas as pd
from args import parse_args
from tqdm import tqdm

# %%
args = parse_args()
root_path = args.root_path
region = args.region

with open(f"{root_path}/{region}/config.json", "r") as fp:
    config = json.load(fp)

# %%
data_path = f"{region}/cctorch"
result_path = f"{region}/adloc_dd"
if not os.path.exists(f"{result_path}"):
    os.makedirs(f"{result_path}")

# %%
stations = pd.read_csv(f"{root_path}/{data_path}/cctorch_stations.csv")
stations["station_id"] = stations["station"]
stations = stations.groupby("station_id").first().reset_index()

# %%
events = pd.read_csv(f"{root_path}/{data_path}/cctorch_events.csv", dtype={"event_index": str})
events["time"] = pd.to_datetime(events["event_time"], format="mixed")

# %%
stations["idx_sta"] = np.arange(len(stations))  # reindex in case the index does not start from 0 or is not continuous
events["idx_eve"] = np.arange(len(events))  # reindex in case the index does not start from 0 or is not continuous
mapping_phase_type_int = {"P": 0, "S": 1}

# %%
with open(f"{root_path}/{data_path}/dt.cc", "r") as f:
    lines = f.readlines()

# %%
event_index1 = []
event_index2 = []
station_index = []
phase_type = []
phase_score = []
phase_dtime = []

stations.set_index("station_id", inplace=True)
events.set_index("event_index", inplace=True)

for line in tqdm(lines):
    if line[0] == "#":
        evid1, evid2, _ = line[1:].split()
    else:
        stid, dt, weight, phase = line.split()
        event_index1.append(events.loc[evid1, "idx_eve"])
        event_index2.append(events.loc[evid2, "idx_eve"])
        station_index.append(stations.loc[stid, "idx_sta"])
        phase_type.append(mapping_phase_type_int[phase])
        phase_score.append(weight)
        phase_dtime.append(dt)


dtypes = np.dtype(
    [
        ("idx_eve1", np.int32),
        ("idx_eve2", np.int32),
        ("idx_sta", np.int32),
        ("phase_type", np.int32),
        ("phase_score", np.float32),
        ("phase_dtime", np.float32),
    ]
)
pairs_array = np.memmap(
    f"{root_path}/{result_path}/pair_dt.dat",
    mode="w+",
    shape=(len(phase_dtime),),
    dtype=dtypes,
)
pairs_array["idx_eve1"] = event_index1
pairs_array["idx_eve2"] = event_index2
pairs_array["idx_sta"] = station_index
pairs_array["phase_type"] = phase_type
pairs_array["phase_score"] = phase_score
pairs_array["phase_dtime"] = phase_dtime
with open(f"{root_path}/{result_path}/pair_dtypes.pkl", "wb") as f:
    pickle.dump(dtypes, f)


# %%
events.to_csv(f"{root_path}/{result_path}/pair_events.csv", index=True, index_label="event_index")
stations.to_csv(f"{root_path}/{result_path}/pair_stations.csv", index=True, index_label="station_id")

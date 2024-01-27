# %%
import argparse
import json
import os
from datetime import datetime
from glob import glob
from itertools import product

import numpy as np
import pandas as pd
import torch


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", nargs="?", type=str, default="local", help="root path")
    parser.add_argument("region", nargs="?", type=str, default="demo", help="region")


args = parse_args()
# %%
root_path = args.root_path
region = args.region

output_path = f"{region}/qtm"
if not os.path.exists(f"{root_path}/{output_path}"):
    os.makedirs(f"{root_path}/{output_path}")

# %%
with open(f"{root_path}/{region}/config.json", "r") as fp:
    config = json.load(fp)

# %% Get mseed list
mseed_list = sorted(glob(f"{root_path}/{region}/waveforms/????-???/??/*.mseed"))
folder_level = 3  # year-jday/hour/station_id.mseed
mseeds = pd.DataFrame(mseed_list, columns=["fname"])
mseeds["mseed_station_id"] = mseeds["fname"].apply(
    lambda x: "/".join(x.replace(".mseed", "").split("/")[-folder_level:])[:-1]
)
mseeds["station_id"] = mseeds["fname"].apply(lambda x: x.replace(".mseed", "").split("/")[-1][:-1])
mseeds["begin_time"] = mseeds["fname"].apply(
    lambda x: datetime.strptime("T".join(x.split("/")[-folder_level : -folder_level + 2]), "%Y-%jT%H").strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )
)
mseeds = (
    mseeds.groupby("mseed_station_id")
    .agg(
        {
            "station_id": lambda x: ",".join(x.unique()),
            "begin_time": lambda x: ",".join(x.unique()),
            "fname": lambda x: ",".join(sorted(x)),
        }
    )
    .reset_index()
)
mseeds.to_csv(f"{root_path}/{region}/qtm/mseed_list.csv", index=False)
with open(f"{root_path}/{region}/qtm/mseed_list.txt", "w") as fp:
    fp.write("\n".join(mseeds["fname"]))

# %%
with open(f"{root_path}/{region}/qtm/event_phase_station_id.txt", "r") as fp:
    event_phase_station_id = fp.read().splitlines()

event_phase_station_id = pd.DataFrame(event_phase_station_id, columns=["event_phase_station_id"])
event_phase_station_id[["event_index", "phase_type", "station_id"]] = event_phase_station_id[
    "event_phase_station_id"
].str.split("/", expand=True)

# %% Generate event mseed pairs
pairs = []
unique_station_ids = np.intersect1d(mseeds["station_id"].unique(), event_phase_station_id["station_id"].unique())

for station_id in unique_station_ids:
    mseed_index = mseeds.loc[mseeds["station_id"] == station_id].index
    event_phase_station_index = event_phase_station_id.loc[event_phase_station_id["station_id"] == station_id].index
    pairs.extend(product(mseed_index, event_phase_station_index))

# %%
with open(f"{root_path}/{region}/qtm/pairs.txt", "w") as fp:
    fp.write("\n".join([f"{x[0]},{x[1]}" for x in pairs]))

# %%
num_gpu = torch.cuda.device_count()
base_cmd = f"python ../CCTorch/run.py --mode=TM --pair_list={root_path}/{region}/qtm/pairs.txt --data_list1={root_path}/{region}/qtm/mseed_list.txt --data_format1=mseed --data_path2={root_path}/{region}/qtm/template.dat --data_format2=memmap --config={root_path}/{region}/qtm/config.json --batch_size=128 --block_size1=1 --block_size2=128 --normalize --reduce_c  --result_path={root_path}/{region}/qtm/ccpairs"
if num_gpu == 0:
    os.system(f"{base_cmd} --device=cpu")
    # os.system(f"{base_cmd} --device=mps")
else:
    os.system(base_cmd)

# %%

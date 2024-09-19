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
from tqdm import tqdm


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", nargs="?", type=str, default="local", help="root path")
    parser.add_argument("region", nargs="?", type=str, default="demo", help="region")
    return parser.parse_args()


args = parse_args()
# %%
root_path = args.root_path
region = args.region

result_path = f"{region}/qtm"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")

# %%
with open(f"{root_path}/{region}/config.json", "r") as fp:
    config = json.load(fp)

# %% Get mseed list
mseed_list = sorted(glob(f"{root_path}/{region}/waveforms/????/???/??/*.mseed"))
subdir = 4  # year/jday/hour/station_id.mseed
mseeds = pd.DataFrame(mseed_list, columns=["fname"])
mseeds["mseed_id"] = mseeds["fname"].apply(lambda x: "/".join(x.replace(".mseed", "").split("/")[-subdir:])[:-1])
mseeds["station_id"] = mseeds["fname"].apply(lambda x: x.replace(".mseed", "").split("/")[-1][:-1])
mseeds["begin_time"] = mseeds["fname"].apply(
    lambda x: datetime.strptime(
        f"{x.split('/')[-subdir]}-{x.split('/')[-subdir+1]}T{x.split('/')[-subdir+2]}", "%Y-%jT%H"
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")
)
mseeds = (
    mseeds.groupby("mseed_id")
    .agg(
        {
            "station_id": lambda x: ",".join(x.unique()),
            "begin_time": lambda x: ",".join(x.unique()),
            "fname": lambda x: ",".join(sorted(x)),
        }
    )
    .reset_index()
)
mseeds["idx_mseed"] = np.arange(len(mseeds))
mseeds.to_csv(f"{root_path}/{region}/qtm/mseed_list.csv", index=False)
with open(f"{root_path}/{region}/qtm/mseed_list.txt", "w") as fp:
    fp.write("\n".join(mseeds["fname"]))

# %%
# with open(f"{root_path}/{region}/qtm/event_phase_station_id.txt", "r") as fp:
#     event_phase_station_id = fp.read().splitlines()
picks = pd.read_csv(f"{root_path}/{region}/cctorch/cctorch_picks.csv")
stations = pd.read_csv(f"{root_path}/{region}/cctorch/cctorch_stations.csv")
picks = picks.merge(stations[["idx_sta", "station_id"]], on="idx_sta")
print(picks.iloc[:10])

# %% Generate event mseed pairs
pairs = []
unique_station_ids = np.intersect1d(mseeds["station_id"].unique(), picks["station_id"].unique())
print(f"Number of unique station ids: {len(unique_station_ids)}")

# %%
with open(f"{root_path}/{region}/qtm/pairs.txt", "w") as fp:
    mseeds = mseeds.set_index("idx_mseed")
    picks = picks.groupby("station_id")
    for idx_mseed, row in tqdm(mseeds.iterrows(), total=len(mseeds), desc="Writing pairs"):
        station_id = row["station_id"]
        for idx_pick in picks.get_group(station_id)["idx_pick"].values:
            fp.write(f"{idx_mseed},{idx_pick}\n")

## based on GPU memory
batch = 16
block_size1 = 1
block_size2 = 100_000  # ~7GB

# %%
base_cmd = (
    f"../CCTorch/run.py --mode=TM --pair_list={root_path}/{region}/qtm/pairs.txt "
    f"--data_list1={root_path}/{region}/qtm/mseed_list.txt --data_format1=mseed "
    f"--data_list2={root_path}/{region}/cctorch/cctorch_picks.csv --data_path2={root_path}/{region}/cctorch/template.dat --data_format2=memmap "
    f"--config={root_path}/{region}/cctorch/config.json --batch_size={batch} --block_size1={block_size1} --block_size2={block_size2} --normalize --reduce_c --result_path={root_path}/{region}/qtm/ccpairs"
)

# %%
num_gpu = torch.cuda.device_count()
if num_gpu == 0:
    if os.uname().sysname == "Darwin":
        cmd = f"python {base_cmd} --device=mps"
    else:
        cmd = f"python {base_cmd} --device=cpu"
elif num_gpu == 1:
    cmd = f"python {base_cmd}"
else:
    cmd = f"torchrun --standalone --nproc_per_node {num_gpu} {base_cmd}"

# %%
print(cmd)
os.system(cmd)

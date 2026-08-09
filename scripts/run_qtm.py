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
    parser.add_argument("--batch_size", type=int, default=4,
                        help="templates correlated per GPU step. Each pair holds a full copy of the "
                             "continuous trace (~600 MB for 24 h at 100 Hz, 3 comp), so this is bounded "
                             "by GPU memory: 16 needs ~16 GB of a 24 GB card")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"],
                        help="precision of the normalized correlation. float32 tracks float64 to "
                             "~1.5e-6 and runs ~2.5x faster; float64 is exact but sits at the FP64 "
                             "roofline. See tests/qtm/test_dtype.py.")
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
# mseed_list = sorted(glob(f"{root_path}/{region}/waveforms/????/???/??/*.mseed"))
# subdir = 4  # year/jday/hour/station_id.mseed
mseed_list = sorted(glob(f"{root_path}/{region}/waveforms/????/???/*.mseed"))
subdir = 3  # year/jday/station_id.mseed
mseeds = pd.DataFrame(mseed_list, columns=["file_name"])
mseeds["mseed_id"] = mseeds["file_name"].apply(lambda x: "/".join(x.replace(".mseed", "").split("/")[-subdir:])[:-1])
mseeds["station_id"] = mseeds["file_name"].apply(lambda x: x.replace(".mseed", "").split("/")[-1][:-1])
# mseeds["begin_time"] = mseeds["fname"].apply(
#     lambda x: datetime.strptime(
#         f"{x.split('/')[-subdir]}-{x.split('/')[-subdir+1]}T{x.split('/')[-subdir+2]}", "%Y-%jT%H"
#     ).strftime("%Y-%m-%dT%H:%M:%S.%f")
# )
mseeds["begin_time"] = mseeds["file_name"].apply(
    lambda x: datetime.strptime(f"{x.split('/')[-subdir]}-{x.split('/')[-subdir+1]}", "%Y-%j").strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )
)

## Every waveform day on disk is scanned unless bounded. That is rarely what is wanted:
## the tests scan the single day their templates come from, and a production archive may
## hold years. scan_start/scan_end sit beside event_start/event_end in config["cctorch"],
## so one place decides which days are cut from and which are searched.
cctorch_config = config.get("cctorch", {})
for bound, op in (("scan_start", "ge"), ("scan_end", "lt")):
    if cctorch_config.get(bound):
        n_before = len(mseeds)
        keep = getattr(pd.to_datetime(mseeds["begin_time"]), op)(pd.Timestamp(cctorch_config[bound]))
        mseeds = mseeds[keep]
        print(f"{bound} {cctorch_config[bound]}: {len(mseeds)} of {n_before} waveform files kept")
if len(mseeds) == 0:
    raise ValueError(
        f"no waveforms between {cctorch_config.get('scan_start')} and {cctorch_config.get('scan_end')}"
    )

mseeds = (
    mseeds.groupby("mseed_id")
    .agg(
        {
            "station_id": lambda x: "|".join(x.unique()),
            "begin_time": lambda x: "|".join(x.unique()),
            "file_name": lambda x: "|".join(sorted(x)),
        }
    )
    .reset_index()
)
mseeds["idx_mseed"] = np.arange(len(mseeds))
print(mseeds.head())

mseeds.to_csv(f"{root_path}/{region}/qtm/mseed_list.csv", index=False)
with open(f"{root_path}/{region}/qtm/mseed_list.txt", "w") as fp:
    fp.write("\n".join(mseeds["file_name"]))

# %%
# with open(f"{root_path}/{region}/qtm/event_phase_station_id.txt", "r") as fp:
#     event_phase_station_id = fp.read().splitlines()
picks = pd.read_csv(f"{root_path}/{region}/qtm/template_picks.csv")
stations = pd.read_csv(f"{root_path}/{region}/qtm/template_stations.csv")
if "station_id" not in picks.columns:
    picks = picks.merge(stations[["idx_sta", "station_id"]], on="idx_sta")
print(picks.head())

# %% Generate event mseed pairs
## A station can have waveforms but no templates, or templates but no waveforms for the day
## being scanned; only their intersection can be correlated. The intersection was already
## computed here and then ignored, so a station with waveforms and no templates raised a
## KeyError instead of being skipped.
pairs = mseeds[["idx_mseed", "station_id"]].merge(picks[["idx_pick", "station_id"]], on="station_id")
pairs = pairs.sort_values(["idx_mseed", "idx_pick"])

n_paired = pairs["station_id"].nunique()
print(f"Number of unique station ids: {n_paired}")
print(f"  skipping {mseeds['station_id'].nunique() - n_paired} station(s) with waveforms but no "
      f"templates, {picks['station_id'].nunique() - n_paired} with templates but no waveforms")
print(f"Number of pairs: {len(pairs):,}")

pairs[["idx_mseed", "idx_pick"]].to_csv(
    f"{root_path}/{region}/qtm/pairs.txt", index=False, header=False
)

## based on GPU memory
batch_size = args.batch_size
block_size1 = 1
block_size2 = 100_000  # ~7GB

# %%
base_cmd = (
    f"../CCTorch/run.py --mode=TM --pair_list={root_path}/{region}/qtm/pairs.txt "
    f"--data_list1={root_path}/{region}/qtm/mseed_list.csv --data_format1=mseed "
    f"--data_list2={root_path}/{region}/qtm/template_picks.csv --data_path2={root_path}/{region}/qtm/template.dat --data_format2=memmap "
    f"--picks_csv={root_path}/{region}/qtm/template_picks.csv --events_csv={root_path}/{region}/qtm/template_events.csv --stations_csv={root_path}/{region}/qtm/template_stations.csv "
    f"--config={root_path}/{region}/qtm/config.json --batch_size={batch_size} --block_size1={block_size1} --block_size2={block_size2} --normalize --reduce_c --dtype={args.dtype} --result_path={root_path}/{region}/qtm/ccpairs "
    # f"--device=cuda --dataset_cpu"
    f"--device=cuda"
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
## os.system returns the wait status, not the exit code, and torchrun dying (OOM, for one)
## otherwise passes silently and the previous run's ccpairs get scored as if they were new
status = os.system(cmd)
if status != 0:
    raise SystemExit(f"CCTorch failed (exit status {status >> 8 or status}); see the log above")

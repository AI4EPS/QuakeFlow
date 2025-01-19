# %%
import json
import os
import sys
from collections import defaultdict
from glob import glob
from typing import Dict, List

import fsspec
import numpy as np
import pandas as pd
import torch
from args import parse_args
from run_event_association import associate


def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    node_rank: int = 0,
    num_nodes: int = 1,
    data_type: str = "continuous",
    overwrite: bool = False,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:

    # %%
    if data_type == "continuous":
        # subdir = 3
        subdir = 2
    elif data_type == "event":
        subdir = 1

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    result_path = f"{region}/phasenet_plus"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}", exist_ok=True)

    # %%
    waveform_dir = f"{region}/waveforms"
    # if not os.path.exists(f"{root_path}/{waveform_dir}"):
    #     if protocol != "file":
    #         fs.get(f"{bucket}/{waveform_dir}/", f"{root_path}/{waveform_dir}/", recursive=True)

    if data_type == "continuous":
        # mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????/???/??/*.mseed"))
        mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????/???/*.mseed"))
    elif data_type == "event":
        mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/*.mseed"))
    else:
        raise ValueError("data_type must be either continuous or event")

    # %%
    mseed_3c = defaultdict(list)
    for mseed in mseed_list:
        # key = mseed.replace(f"{root_path}/{waveform_dir}/", "").replace(".mseed", "")
        key = "/".join(mseed.replace(".mseed", "").split("/")[-subdir - 1 :])
        if data_type == "continuous":
            key = key[:-1]
        mseed_3c[key].append(mseed)
    print(f"Number of mseed files: {len(mseed_3c)}")

    # %% skip processed files
    if not overwrite:
        # processed = sorted(glob(f"{root_path}/{result_path}/picks_phasenet_plus/????/???/??/*.csv"))
        processed = sorted(glob(f"{root_path}/{result_path}/picks_phasenet_plus/????/???/*.csv"))
        processed = ["/".join(f.replace(".csv", "").split("/")[-subdir - 1 :]) for f in processed]
        processed = [p[:-1] for p in processed]  ## remove the channel suffix
        print(f"Number of processed files: {len(processed)}")
        keys = sorted(list(set(mseed_3c.keys()) - set(processed)))
    else:
        keys = list(mseed_3c.keys())

    print(f"Number of unprocessed files: {len(keys)}")
    keys = list(np.array_split(keys, num_nodes)[node_rank])
    print(f"Node {node_rank:03d}/{num_nodes:03d}: processing {len(keys)} files")

    if len(keys) == 0:
        return 0

    mseed_3c = [",".join(sorted(mseed_3c[k])) for k in keys]

    # %%
    mseed_file = f"{root_path}/{result_path}/mseed_list_{node_rank:03d}_{num_nodes:03d}.csv"
    with open(mseed_file, "w") as fp:
        fp.write("\n".join(mseed_3c))

    # %%
    response_path = f"{region}/obspy/inventory"

    num_gpu = torch.cuda.device_count()
    print(f"num_gpu = {num_gpu}")
    base_cmd = f"../EQNet/predict.py --model phasenet_plus --add_polarity --add_event --format mseed --data_list={root_path}/{result_path}/mseed_list_{node_rank:03d}_{num_nodes:03d}.csv --response_path={root_path}/{response_path} --result_path={root_path}/{result_path} --batch_size 1 --workers 1 --subdir_level {subdir}"
    # base_cmd += " --resume ../../QuakeFlow/EQNet/model_phasenet_plus_0630/model_99.pth"
    if num_gpu == 0:
        cmd = f"python {base_cmd} --device=cpu"
    elif num_gpu == 1:
        cmd = f"python {base_cmd}"
    else:
        cmd = f"torchrun --standalone --nproc_per_node {num_gpu} {base_cmd}"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region
    num_nodes = args.num_nodes
    node_rank = args.node_rank

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    run_phasenet(root_path=root_path, region=region, config=config, overwrite=args.overwrite)

    if num_nodes == 1:
        os.system(f"python merge_phasenet_plus_picks.py --region {region}")

    if num_nodes == 1:
        config.update({"VPVS_RATIO": 1.73, "VP": 6.0})
        stations = pd.read_json(f"{root_path}/{region}/obspy/stations.json", orient="index")
        stations["station_id"] = stations.index
        events = pd.read_csv(
            f"{root_path}/{region}/phasenet_plus/events_phasenet_plus.csv", parse_dates=["center_time", "event_time"]
        )
        picks = pd.read_csv(f"{root_path}/{region}/phasenet_plus/picks_phasenet_plus.csv", parse_dates=["phase_time"])
        events, picks = associate(picks, events, stations, config)
        events.to_csv(f"{root_path}/{region}/phasenet_plus/phasenet_plus_events_associated.csv", index=False)
        picks.to_csv(f"{root_path}/{region}/phasenet_plus/phasenet_plus_picks_associated.csv", index=False)

# %%

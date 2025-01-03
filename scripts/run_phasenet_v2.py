# %%
import json
import os
import sys
from collections import defaultdict
from glob import glob
from typing import Dict, List

import fsspec
import numpy as np
from args import parse_args


def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    node_rank: int = 0,
    num_nodes: int = 1,
    overwrite: bool = False,
    model_path: str = "../PhaseNet/",
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    result_path = f"{region}/phasenet"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    waveform_dir = f"{region}/waveforms"
    # mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????/???/??/*.mseed"))
    # subdir = 3
    mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????/???/*.mseed"))
    subdir = 2

    # %%
    mseed_3c = defaultdict(list)
    for mseed in mseed_list:
        key = "/".join(mseed.replace(".mseed", "").split("/")[-subdir - 1 :])
        key = key[:-1]  ## remove the channel suffix
        mseed_3c[key].append(mseed)
    print(f"Number of mseed files: {len(mseed_3c)}")

    # %%
    if not overwrite:
        # processed = sorted(glob(f"{root_path}/{result_path}/picks/????/???/??/*.csv"))
        processed = sorted(glob(f"{root_path}/{result_path}/picks/????/???/*.csv"))
        processed = ["/".join(f.replace(".csv", "").split("/")[-subdir - 1 :]) for f in processed]
        processed = [p[:-1] for p in processed]  ## remove the channel suffix
        print(f"Number of processed files: {len(processed)}")

    keys = sorted(list(set(mseed_3c.keys()) - set(processed)))
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
    inventory_path = f"{root_path}/{region}/obspy/inventory"

    # %%
    os.system(
        f"python {model_path}/phasenet/predict.py --model={model_path}/model/190703-214543 --data_dir=./ --data_list={mseed_file} --response_xml={inventory_path} --format=mseed --amplitude --highpass_filter=1.0 --result_dir={root_path}/{result_path} --result_fname=phasenet_picks_{node_rank:03d}_{num_nodes:03d} --batch_size=1 --subdir_level={subdir}"
    )


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region
    num_nodes = args.num_nodes
    node_rank = args.node_rank

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    os.system("cd ../PhaseNet && git checkout quakeflow && git pull")
    run_phasenet(root_path=root_path, region=region, config=config)

    if num_nodes == 1:
        os.system(f"python merge_phasenet_picks.py --region {region}")

# %%

# %%
from typing import Dict, List
from args import parse_args
import os
from collections import defaultdict
from glob import glob
import fsspec
import numpy as np
import json
import os
import sys


def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    node_rank: int = 0,
    num_nodes: int = 1,
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
    mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????/???/??/*.mseed"))

    # %%
    processed = sorted(glob(f"{root_path}/{result_path}/picks/????/???/??/*.csv"))
    processed = [f.replace(f"{root_path}/{result_path}/picks/", "").replace(".csv", "")[:-1] for f in processed]
    print(f"Processed: {len(processed)}")

    # %%
    mseed_3c = defaultdict(list)
    for mseed in mseed_list:
        key = mseed.replace(f"{root_path}/{waveform_dir}/", "").replace(".mseed", "")[:-1]
        if key in processed:
            continue
        mseed_3c[key].append(mseed)
    mseed_3c = [",".join(sorted(v)) for k, v in mseed_3c.items()]
    print(f"Unprocessed: {len(mseed_3c)}")
    mseed_3c = list(np.array_split(mseed_3c, num_nodes)[node_rank])

    # %%
    mseed_file = f"{root_path}/{result_path}/mseed_list_{node_rank:03d}_{num_nodes:03d}.csv"
    with open(mseed_file, "w") as fp:
        fp.write("\n".join(mseed_3c))

    # %%
    inventory_path = f"{root_path}/{region}/obspy/inventory"

    # %%
    os.system(
        f"python {model_path}/phasenet/predict.py --model={model_path}/model/190703-214543 --data_dir=./ --data_list={mseed_file} --response_xml={inventory_path} --format=mseed --amplitude --highpass_filter=1.0 --result_dir={root_path}/{result_path} --result_fname=phasenet_picks_{node_rank:03d}_{num_nodes:03d} --batch_size=1"
    )


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    run_phasenet(root_path=root_path, region=region, config=config)

# %%

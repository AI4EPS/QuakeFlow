# %%
import json
import os
from glob import glob
from typing import Dict, List

import fsspec
from args import parse_args


def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    node_rank: int = 0,
    num_nodes: int = 1,
    model_path: str = "../PhaseNet/",
    mseed_list: List = None,
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
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        if protocol != "file":
            fs.get(f"{bucket}/{waveform_dir}/", f"{root_path}/{waveform_dir}/", recursive=True)

    if mseed_list is None:
        mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????/???/??/*.mseed"))
    else:
        with open(f"{root_path}/{region}/{mseed_list}", "r") as fp:
            mseed_list = fp.read().split("\n")

    # %% group channels into stations
    mseed_list = sorted(list(set([x.split(".mseed")[0][:-1] + "*.mseed" for x in mseed_list])))

    # %%
    ## filter out processed events
    # processed_list = sorted(list(fs.glob(f"{result_path}/????-???/??/*.csv")))
    # processed_event_id = [f.name.split("/")[-1].replace(".csv", ".mseed") for f in processed_list]
    # file_list = [f for f in file_list if f.split("/")[-1] not in processed_event_id]
    # mseed_list = sorted(list(set(mseed_list)))

    # %%
    if protocol != "file":
        fs.get(
            f"{bucket}/{region}/results/network/inventory.xml", f"{root_path}/{region}/results/network/inventory.xml"
        )

    # %%
    with open(f"{root_path}/{result_path}/mseed_list_{node_rank:03d}_{num_nodes:03d}.csv", "w") as fp:
        fp.write("fname\n")
        fp.write("\n".join(mseed_list))

    # %%
    cmd = f"python {model_path}/phasenet/predict.py --model={model_path}/model/190703-214543 --data_dir=./ --data_list={root_path}/{result_path}/mseed_list_{node_rank:03d}_{num_nodes:03d}.csv --response_xml={root_path}/{region}/results/network/inventory.xml --format=mseed --amplitude --highpass_filter=1.0 --result_dir={root_path}/{result_path} --result_fname=phasenet_picks_{node_rank:03d}_{num_nodes:03d} --batch_size=1"
    # cmd += " --sampling_rate 100"
    os.system(cmd)

    if protocol != "file":
        fs.put(
            f"{root_path}/{result_path}/phasenet_picks_{node_rank:03d}_{num_nodes:03d}.csv",
            f"{bucket}/{result_path}/phasenet_picks_{node_rank:03d}_{num_nodes:03d}.csv",
        )

    # copy to results/picking
    if not os.path.exists(f"{root_path}/{region}/results/picking"):
        os.makedirs(f"{root_path}/{region}/results/picking")
    os.system(
        f"cp {root_path}/{result_path}/phasenet_picks_{node_rank:03d}_{num_nodes:03d}.csv {root_path}/{region}/results/picking/phase_picks_{node_rank:03d}_{num_nodes:03d}.csv"
    )
    if protocol != "file":
        fs.put(
            f"{root_path}/{result_path}/phasenet_picks_{node_rank:03d}_{num_nodes:03d}.csv",
            f"{bucket}/{region}/results/picking/phase_picks_{node_rank:03d}_{num_nodes:03d}.csv",
        )

    return f"{result_path}/phasenet_picks_{node_rank:03d}_{num_nodes:03d}.csv"


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    run_phasenet(root_path=root_path, region=region, config=config)

# %%

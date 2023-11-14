# %%
from typing import Dict, List

from kfp import dsl


@dsl.component()
def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    rank: int = 0,
    model_path: str = "../PhaseNet/",
    data_type: str = "continuous",
    mseed_list: List = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:
    import os
    from glob import glob

    import fsspec
    import torch

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    # result_path = f"{region}/phasenet/{rank:03d}"
    result_path = f"{region}/phasenet"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}", exist_ok=True)

    # %%
    waveform_dir = f"{region}/waveforms"
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        if protocol != "file":
            fs.get(f"{bucket}/{waveform_dir}/", f"{root_path}/{waveform_dir}/", recursive=True)

    if mseed_list is None:
        if data_type == "continuous":
            mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????-???/??/*.mseed"))
        elif data_type == "event":
            mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/*.mseed"))
        else:
            raise ValueError("data_type must be either continuous or event")

    # %% group channels into stations
    if data_type == "continuous":
        mseed_list = list(set([x.split(".mseed")[0][:-1] + "*.mseed" for x in mseed_list]))
    mseed_list = sorted(mseed_list)

    # %%
    if protocol != "file":
        fs.get(f"{bucket}/{region}/obspy/inventory.xml", f"{root_path}/{region}/obspy/inventory.xml")

    # %%
    with open(f"{root_path}/{result_path}/mseed_list_{rank:03d}.csv", "w") as fp:
        fp.write("\n".join(mseed_list))

    # %%
    if data_type == "continuous":
        folder_depth = 3
    elif data_type == "event":
        folder_depth = 1
    num_gpu = torch.cuda.device_count()
    print(f"num_gpu = {num_gpu}")
    base_cmd = f"../EQNet/predict.py --model phasenet --add_polarity --add_event --format mseed --data_list={root_path}/{result_path}/mseed_list_{rank:03d}.csv --response_xml={root_path}/{region}/obspy/inventory.xml --result_path={root_path}/{result_path} --batch_size 1 --workers 1 --folder_depth {folder_depth}"
    if num_gpu == 0:
        cmd = f"python {base_cmd} --device=cpu"
    elif num_gpu == 1:
        cmd = f"python {base_cmd}"
    else:
        cmd = f"torchrun --standalone --nproc_per_node {num_gpu} {base_cmd}"
    print(cmd)
    os.system(cmd)

    os.system(
        f"cp {root_path}/{result_path}/picks_phasenet.csv {root_path}/{result_path}/phasenet_picks_{rank:03d}.csv"
    )
    os.system(
        f"cp {root_path}/{result_path}/events_phasenet.csv {root_path}/{result_path}/phasenet_events_{rank:03d}.csv",
    )

    if protocol != "file":
        fs.put(f"{root_path}/{result_path}/", f"{bucket}/{result_path}/", recursive=True)

    return f"{result_path}/phasenet_picks_{rank:03d}.csv"


if __name__ == "__main__":
    import json
    import os
    import sys

    root_path = "local"
    region = "demo"
    data_type = "continuous"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]
    if len(sys.argv) > 3:
        data_type = sys.argv[3]

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    run_phasenet.python_func(root_path, region=region, config=config, data_type=data_type)

    if config["num_nodes"] == 1:
        os.system(f"mv {root_path}/{region}/phasenet/mseed_list_000.csv {root_path}/{region}/phasenet/mseed_list.csv")
        os.system(
            f"mv {root_path}/{region}/phasenet/phasenet_picks_000.csv {root_path}/{region}/phasenet/phasenet_picks.csv"
        )
        os.system(
            f"mv {root_path}/{region}/phasenet/phasenet_events_000.csv {root_path}/{region}/phasenet/phasenet_events.csv"
        )

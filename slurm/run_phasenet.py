# %%
from typing import Dict, List

from kfp import dsl


@dsl.component(base_image="zhuwq0/phasenet-api:latest")
def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    rank: int = 0,
    model_path: str = "../PhaseNet/",
    mseed_list: List = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:
    import os
    from glob import glob

    import fsspec

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
        mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????-???/??/*.mseed"))

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
        fs.get(f"{bucket}/{region}/obspy/inventory.xml", f"{root_path}/{region}/obspy/inventory.xml")

    # %%
    with open(f"{root_path}/{result_path}/mseed_list_{rank:03d}.csv", "w") as fp:
        fp.write("fname\n")
        fp.write("\n".join(mseed_list))

    # %%
    os.system(
        f"python {model_path}/phasenet/predict.py --model={model_path}/model/190703-214543 --data_dir=./ --data_list={root_path}/{result_path}/mseed_list_{rank:03d}.csv --response_xml={root_path}/{region}/obspy/inventory.xml --format=mseed --amplitude --highpass_filter=1.0 --result_dir={root_path}/{result_path} --result_fname=phasenet_picks_{rank:03d} --batch_size=1"
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
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    run_phasenet.python_func(root_path, region=region, config=config)

    if config["num_nodes"] == 1:
        os.system(f"mv {root_path}/{region}/phasenet/mseed_list_000.csv {root_path}/{region}/phasenet/mseed_list.csv")
        os.system(
            f"mv {root_path}/{region}/phasenet/phasenet_picks_000.csv {root_path}/{region}/phasenet/phasenet_picks.csv"
        )

# %%

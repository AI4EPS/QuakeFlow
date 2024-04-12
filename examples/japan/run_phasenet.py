# %%
from typing import Dict, List

from kfp import dsl

PLATFORM = "cloud"  # "cloud" or "local"


def null_decorator(func):
    return func


def quakeflow_component(platform, decorator=dsl.component, **kwargs):
    if platform != "local":
        return decorator(**kwargs)
    else:
        return null_decorator


# @dsl.component(base_image="zhuwq0/phasenet-api:latest")
@quakeflow_component(platform=PLATFORM, base_image="zhuwq0/phasenet-api:latest")
def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    rank: int = 0,
    world_size: int = 1,
    model_path: str = "../../PhaseNet",
    mseed_list: List = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:
    # %%
    import os
    from collections import defaultdict
    from glob import glob

    import fsspec
    import numpy as np

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    result_path = f"{region}/phasenet"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    waveform_dir = f"{region}/waveforms"

    if mseed_list is None:
        mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????-???/??/*.sac"))
    else:
        with open(f"{root_path}/{region}/{mseed_list}", "r") as fp:
            mseed_list = fp.read().split("\n")

    # %% group channels
    mseed_3c = defaultdict(list)
    for mseed in mseed_list:
        # B.sac is for station N.WJMF.EB
        mseed_3c[mseed.replace("B.sac", "").replace(".sac", "")[:-1]].append(mseed)
    mseed_3c = [",".join(sorted(v)) for k, v in mseed_3c.items()]
    mseed_3c = list(np.array_split(mseed_3c, world_size)[rank])

    # %%
    with open(f"{root_path}/{result_path}/mseed_list_{rank:03d}.csv", "w") as fp:
        fp.write("\n".join(mseed_3c))

    # %%
    os.system(
        f"python {model_path}/phasenet/predict.py --model={model_path}/model/190703-214543 --data_dir=./ --data_list={root_path}/{result_path}/mseed_list_{rank:03d}.csv --format=mseed_3c --amplitude --highpass_filter=1.0 --result_dir={root_path}/{result_path} --result_fname=phasenet_picks_{rank:03d} --batch_size=1"
    )

    if protocol != "file":
        fs.put(
            f"{root_path}/{result_path}/phasenet_picks_{rank:03d}.csv",
            f"{bucket}/{result_path}/phasenet_picks_{rank:03d}.csv",
        )

    # copy to results/phase_picking
    if not os.path.exists(f"{root_path}/{region}/results/phase_picking"):
        os.makedirs(f"{root_path}/{region}/results/phase_picking")
    os.system(
        f"cp {root_path}/{result_path}/phasenet_picks_{rank:03d}.csv {root_path}/{region}/results/phase_picking/phase_picks_{rank:03d}.csv"
    )
    if protocol != "file":
        fs.put(
            f"{root_path}/{result_path}/phasenet_picks_{rank:03d}.csv",
            f"{bucket}/{region}/results/phase_picking/phase_picks_{rank:03d}.csv",
        )

    return f"{result_path}/phasenet_picks_{rank:03d}.csv"


if __name__ == "__main__":
    import json
    import multiprocessing as mp
    import os
    import sys

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    root_path = "local"
    region = "hinet"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    # run_phasenet.execute(root_path=root_path, region=region, config=config)
    ncpu = 16
    with mp.get_context("spawn").Pool(ncpu) as pool:
        results = pool.starmap(
            run_phasenet,
            [(root_path, region, config, i, ncpu) for i in range(ncpu)],
        )


# %%

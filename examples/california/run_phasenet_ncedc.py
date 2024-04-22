# %%
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from kfp import compiler, dsl
from kfp.client import Client


@dsl.component(packages_to_install=["fsspec", "gcsfs", "s3fs", "tqdm"])
def get_mseed_list(
    root_path: str,
    region: str,
    config: Dict,
    rank: int = 0,
    year: int = 2023,
    stations: List = None,
    protocol: str = "s3",
    bucket: str = "",
    token: Dict = None,
) -> str:
    # %%
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    import fsspec
    from tqdm import tqdm

    # %%
    protocol = "s3"
    bucket = "ncedc-pds"
    folder = "continuous_waveforms"
    fs = fsspec.filesystem(protocol=protocol, anon=True)

    # # %%
    # valid_channels = ["3", "2", "1", "E", "N", "Z"]
    # valid_instruments = ["BH", "HH", "EH", "HN", "DP"]

    # # %%
    # mseed_dir = "mseed_list"
    # if not os.path.exists(f"{mseed_dir}"):
    #     os.makedirs(f"{mseed_dir}")

    # # for year in range(1999, 2024)[::-1]:
    # def process(year):
    #     networks = fs.glob(f"{bucket}/{folder}/??")
    #     pbar = tqdm(networks, desc=f"{year}")
    #     for i, network in enumerate(pbar):
    #         pbar.set_description(f"{year} {network.split('/')[-1]}")
    #         # mseed_list = []
    #         # jdays = fs.glob(f"{network}/{year}/????.???")
    #         # for jday in jdays:
    #         #     mseeds = fs.glob(f"{jday}/*.{jday.split('/')[-1]}")
    #         #     mseed_list.extend(mseeds)
    #         mseed_list = fs.glob(f"{network}/{year}/{year}.???/*.{year}.???")
    #         mseed_list = sorted([f"{protocol}://{mseed}" for mseed in mseed_list])
    #         if len(mseed_list) > 0:
    #             with open(f"{mseed_dir}/{year}_{network.split('/')[-1]}.txt", "w") as fp:
    #                 fp.write("\n".join(mseed_list))

    #         groups = defaultdict(list)
    #         for mseed in mseed_list:
    #             tmp = mseed.split(".")
    #             if (tmp[3][-1] in valid_channels) and (tmp[3][:2] in valid_instruments):
    #                 key = ".".join(tmp[:3]) + "." + tmp[3][:-1] + "." + ".".join(tmp[4:])
    #                 groups[key].append(mseed)
    #             # else:
    #             #     print(f"Invalid channel: {mseed}")

    #         if len(groups) > 0:
    #             with open(f"{mseed_dir}/{year}_{network.split('/')[-1]}_3c.txt", "w") as fp:
    #                 keys = sorted(groups.keys())
    #                 for key in keys:
    #                     fp.write(",".join(sorted(groups[key])) + "\n")

    # num_cores = 32
    # with ThreadPoolExecutor(max_workers=num_cores) as executor:
    #     futures = [executor.submit(process, year) for year in range(1999, 2024)]

    # raise

    # %%
    fs = fsspec.filesystem("gs", token=token)
    bucket = "quakeflow_catalog"
    folder = "NC"
    networks = fs.glob(f"{bucket}/{folder}/mseed_list/{year}_??_3c.txt")
    networks = [x.split("/")[-1].split("_")[1] for x in networks]
    # networks = ["NC", "BK"]
    mseed_list = []
    for network in networks:
        if fs.exists(f"{bucket}/{folder}/mseed_list/{year}_{network}_3c.txt"):
            with fs.open(f"{bucket}/{folder}/mseed_list/{year}_{network}_3c.txt", "r") as fp:
                mseed_list.extend(fp.read().splitlines())

    # %%
    if stations is not None:
        mseed_list_filt = []
        for mseed in mseed_list:
            station_id = ".".join(mseed.split(",")[0].split("/")[-1].split(".")[:3])
            if station_id[:-1] in stations:
                mseed_list_filt.append(mseed)
        print(f"Number of selected mseed files: {len(mseed_list_filt)} / {len(mseed_list)}")
        mseed_list = mseed_list_filt

    # %%
    mseed_list = sorted(list(set(mseed_list)))
    print(f"Total number of mseed files: {len(mseed_list)}")

    folder = "NC/phasenet"
    # processed = set(fs.glob(f"{bucket}/{folder}/**/*.csv"))
    processed = []
    for network in tqdm(networks, desc="Count processed"):
        processed.extend(fs.glob(f"{bucket}/{folder}/{network}/{year}/????.???/*.csv"))
    processed = set(processed)
    mseed_csv_set = set()
    mapping_dit = {}
    for mseed in tqdm(mseed_list, desc="Filter processed"):
        tmp = mseed.split(",")[0].lstrip("s3://").split("/")
        parant_dir = "/".join(tmp[2:-1])
        fname = tmp[-1].rstrip(".mseed") + ".csv"
        tmp_name = f"{bucket}/{folder}/{parant_dir}/{fname}"
        mseed_csv_set.add(tmp_name)
        mapping_dit[tmp_name] = mseed
    mseed_csv_set = list(mseed_csv_set - processed)
    unprocess = sorted([mapping_dit[x] for x in mseed_csv_set], reverse=True, key=lambda x: "/".join(x.split("/")[-2:]))
    print(f"Unprocessed sample {len(unprocess)}")

    # %%
    # raise

    with fs.open(f"{bucket}/{folder}/mseed_list/{year}_3c.txt", "w") as fp:
        fp.write("\n".join(unprocess))

    return f"{bucket}/{folder}/mseed_list/{year}_3c.txt"


@dsl.component(base_image="zhuwq0/phasenet-ncedc:v1.2", packages_to_install=["fsspec", "gcsfs", "s3fs"])
def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    rank: int = 0,
    model_path: str = "./",
    mseed_list: str = "mseed_list.txt",
    protocol: str = "s3",
    bucket: str = "",
    token: Dict = None,
) -> int:
    # %%
    import json
    import os
    from collections import defaultdict
    from glob import glob

    import fsspec
    import obspy
    from tqdm import tqdm

    # %%
    fs = fsspec.filesystem("gs", token=token)

    with fs.open(mseed_list, "r") as fp:
        mseed_list = fp.read().splitlines()

    with open("mseed_list.txt", "w") as fp:
        fp.write("\n".join(mseed_list[rank :: config["world_size"]]))

    if len(mseed_list[rank :: config["world_size"]]) == 0:
        return 0

    # %%
    os.system(
        f"python {model_path}/phasenet/predict.py --model={model_path}/model/190703-214543 --data_list=mseed_list.txt  --format=mseed --amplitude --batch_size=1 --sampling_rate=100 --highpass_filter=1.0 --result_dir={root_path}/picks/"
    )

    return 0


if __name__ == "__main__":
    # %%
    parser = argparse.ArgumentParser(description="Run PhaseNet on NCEDC data")
    parser.add_argument("--year", type=int, default=2023, help="Year to process")
    parser.add_argument("--world_size", type=int, default=64, help="Number of workers")
    args = parser.parse_args()

    token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)

    # %%
    stations = pd.read_csv("stations_ncedc.csv")
    stations["station_id"] = stations["station"] + "." + stations["network"] + "." + stations["instrument"]
    stations = stations["station_id"].unique().tolist()

    # year = 2012
    # year = 2013
    # year = 2014
    # year = 2015
    # year = 2022
    # year = 2023
    year = args.year
    print(f"{year = }")

    # mseed_list = get_mseed_list.execute(root_path="./", region="NC", config={}, year=year, token=token)
    # print(f"{mseed_list = }")
    # run_phasenet.execute(
    #     root_path="./",
    #     region="NC",
    #     config={"world_size": 1},
    #     rank=0,
    #     token=token,
    #     model_path="../PhaseNet/",
    #     mseed_list=mseed_list,
    # )
    # raise

    bucket = "quakeflow_share"
    protocol = "gs"
    token = None
    token_file = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    if os.path.exists(token_file):
        with open(token_file, "r") as fp:
            token = json.load(fp)

    yaml_path = Path("yaml")
    if not yaml_path.exists():
        yaml_path.mkdir()

    world_size = args.world_size

    @dsl.pipeline
    def run_pipeline(root_path: str, region: str, config: Dict, token: Dict = None):
        mseed_op = get_mseed_list(
            root_path=root_path, region=region, config=config, year=year, stations=stations, token=token
        )
        mseed_op.set_cpu_request("2100m")
        mseed_op.set_memory_request("12000Mi")
        with dsl.ParallelFor(items=[i for i in range(world_size)], parallelism=world_size) as item:
            # for item in [0]:
            phasenet_op = run_phasenet(
                root_path=root_path,
                region=region,
                config=config,
                mseed_list=mseed_op.output,
                rank=item,
                token=token,
            )
            phasenet_op.set_cpu_request("2100m")
            phasenet_op.set_memory_request("12000Mi")
            # phasenet_op.set_retry(3)

    client = Client("https://4fedc9c19a233c34-dot-us-west1.pipelines.googleusercontent.com")
    run = client.create_run_from_pipeline_func(
        run_pipeline,
        arguments={"token": token, "region": "NC", "root_path": "./", "config": {"world_size": world_size}},
        run_name=f"phasenet-{year}",
        enable_caching=False,
    )
# %%

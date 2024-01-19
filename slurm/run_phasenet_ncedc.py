# %%
from typing import Dict, List

from kfp import dsl


@dsl.component(base_image="zhuwq0/phasenet-ncedc:v1.1", packages_to_install=["fsspec", "gcsfs", "s3fs"])
def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    rank: int = 0,
    model_path: str = "./",
    mseed_list: List = None,
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
    protocol = "s3"
    bucket = "ncedc-pds"
    folder = "continuous_waveforms"
    fs = fsspec.filesystem(protocol=protocol, anon=True)

    # %%
    valid_channels = ["3", "2", "1", "E", "N", "Z"]
    valid_instruments = ["BH", "HH", "EH", "HN", "DP"]

    # %%
    # mseed_dir = "mseed_list"
    # if not os.path.exists(f"{mseed_dir}"):
    #     os.makedirs(f"{mseed_dir}")

    # for year in range(2023, 2024):
    #     networks = fs.glob(f"{bucket}/{folder}/*")
    #     for i, network in enumerate(tqdm(networks)):
    #         mseed_list = []
    #         # years = fs.glob(f"{network}/????")
    #         # for year in years:
    #         jdays = fs.glob(f"{network}/{year}/????.???")
    #         for jday in jdays:
    #             mseeds = fs.glob(f"{jday}/*.{jday.split('/')[-1]}")
    #             mseed_list.extend(mseeds)

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

    # %%
    world_size = config["world_size"]

    fs = fsspec.filesystem("gs", token=token)
    bucket = "quakeflow_catalog"
    folder = "NC"
    networks = ["NC", "BK"]
    mseed_list = []
    for network in networks:
        with fs.open(f"{bucket}/{folder}/mseed_list/2023_{network}_3c.txt", "r") as fp:
            mseed_list.extend(fp.read().splitlines())
    mseed_list = sorted(list(set(mseed_list)))

    with open("mseed_list.txt", "w") as fp:
        fp.write("\n".join(mseed_list[rank::world_size]))

    print(
        f"Total number of mseed files: {len(mseed_list[rank::world_size])}/{len(mseed_list)}, rank: {rank}, world_size: {world_size}"
    )

    # %%
    os.system(
        f"python {model_path}/phasenet/predict.py --model={model_path}/model/190703-214543 --data_list=mseed_list.txt  --format=mseed --amplitude --batch_size=1 --sampling_rate=100 --highpass_filter=1.0 --result_dir={root_path}/picks/"
    )

    return 0


if __name__ == "__main__":
    import json
    import os
    from pathlib import Path

    from kfp import compiler
    from kfp.client import Client

    token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)

    # run_phasenet.execute(
    #     root_path="./", region="NC", config={"world_size": 1}, rank=0, token=token, model_path="../PhaseNet/"
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

    world_size = 128

    @dsl.pipeline
    def run_pipeline(root_path: str, region: str, config: Dict, token: Dict = None):
        with dsl.ParallelFor(items=[i for i in range(world_size)], parallelism=world_size) as item:
            # for item in [0]:
            op = run_phasenet(
                root_path=root_path,
                region=region,
                config=config,
                rank=item,
                token=token,
            )
            op.set_cpu_request("1100m")
            op.set_memory_request("12000Mi")
            op.set_retry(3)

    client = Client("https://60381a1245c5a95b-dot-us-west1.pipelines.googleusercontent.com")
    run = client.create_run_from_pipeline_func(
        run_pipeline,
        arguments={"token": token, "region": "NC", "root_path": "./", "config": {"world_size": world_size}},
    )
# %%

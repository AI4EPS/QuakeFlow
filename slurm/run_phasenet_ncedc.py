# %%
from typing import Dict, List

from kfp import dsl


@dsl.component(base_image="zhuwq0/phasenet-ncedc:v1.4", packages_to_install=["fsspec", "gcsfs", "s3fs"])
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
    # fs = fsspec.filesystem(protocol="s3", anon=True)
    # bucket = "ncedc-pds"
    # folder = "continuous_waveforms"
    # # %%
    # mseed_dir = "mseed_list"
    # if not os.path.exists(f"{mseed_dir}"):
    #     os.makedirs(f"{mseed_dir}")
    # networks = fs.glob(f"{bucket}/{folder}/*")
    # for i, network in enumerate(networks):
    #     mseed_list = []
    #     years = fs.glob(f"{network}/????")
    #     for year in tqdm(years, desc=network):
    #         jdays = fs.glob(f"{year}/????.???")
    #         for jday in jdays:
    #             mseeds = fs.glob(f"{jday}/*.{jday.split('/')[-1]}")
    #             mseed_list.extend(mseeds)
    #     with open(f"{mseed_dir}/mseed_list_{network.split('/')[-1]}.txt", "w") as fp:
    #         fp.write("\n".join(mseed_list))
    #     groups = defaultdict(list)
    #     for mseed in tqdm(mseed_list):
    #         tmp = mseed.split(".")
    #         key = ".".join(tmp[:3]) + "." + tmp[3][:-1] + "." + ".".join(tmp[4:])
    #         groups[key].append(mseed)
    #     with open(f"{mseed_dir}/mseed_list_{network.split('/')[-1]}_3c.txt", "w") as fp:
    #         for key, value in tqdm(groups.items()):
    #             fp.write(",".join(sorted(value)) + "\n")
    # %%

    world_size = config["world_size"]

    fs = fsspec.filesystem("gs", token=token)
    bucket = "quakeflow_catalog"
    folder = "NC"
    network = region
    with fs.open(f"{bucket}/{folder}/mseed_list/mseed_list_{network}.txt", "r") as fp:
        mseed_list = fp.read().split("\n")

    valid_channels = ["3", "2", "1", "E", "N", "Z", "U", "V"]
    valid_instruments = ["BH", "HH", "EH", "HN", "EL"]
    groups = defaultdict(list)
    for mseed in tqdm(mseed_list):
        tmp = mseed.split(".")
        if (tmp[3][-1] in valid_channels) and (tmp[3][:2] in valid_instruments):
            key = ".".join(tmp[:3]) + "." + tmp[3][:-1] + "." + ".".join(tmp[4:])
            groups[key].append(mseed)
        # else:
        #     print(f"Invalid channel: {mseed}")

    with open("mseed_list.txt", "w") as fp:
        # for key, value in tqdm(groups.items()):
        #     fp.write(",".join(sorted(value)) + "\n")
        keys = sorted(groups.keys())[rank::world_size]
        for key in tqdm(keys):
            fp.write(",".join(sorted(groups[key])) + "\n")

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

    # run_phasenet.python_func(
    #     root_path="./", region="NC", config={"world_size": 1}, rank=0, token=token, model_path="../PhaseNet/"
    # )
    # raise

    bucket = "quakeflow_share"
    protocol = "gs"
    token = None
    token_file = "/Users/weiqiang/.config/gcloud/application_default_credentials.json"
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
            op.set_memory_limit("12000Mi")
            op.set_retry(3)

    client = Client("https://3824c3562c113c3e-dot-us-central1.pipelines.googleusercontent.com")
    run = client.create_run_from_pipeline_func(
        run_pipeline,
        arguments={"token": token, "region": "NC", "root_path": "./", "config": {"world_size": world_size}},
    )
# %%

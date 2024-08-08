# %%
from typing import Dict

from kfp import dsl


@dsl.component(base_image="zhuwq0/quakeflow:latest")
def set_config(root_path: str, region: str, config: Dict, protocol: str, bucket: str, token: Dict) -> Dict:
    import json
    import os

    import fsspec

    fs = fsspec.filesystem(protocol, token=token)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    data_dir = f"{region}"
    if not os.path.exists(f"{root_path}/{data_dir}"):
        os.makedirs(f"{root_path}/{data_dir}", exist_ok=True)
    for subfolder in ["obspy", "waveforms", "results", "kubeflow"]:
        if not os.path.exists(f"{root_path}/{data_dir}/{subfolder}"):
            os.makedirs(f"{root_path}/{data_dir}/{subfolder}", exist_ok=True)
    for subfolder in [
        "data",
        "phase_picking",
        "phase_association",
        "earthquake_location",
        "earthquake_relocation",
        "earthquake_mechanism",
    ]:
        if not os.path.exists(f"{root_path}/{data_dir}/results/{subfolder}"):
            os.makedirs(f"{root_path}/{data_dir}/results/{subfolder}", exist_ok=True)

    config_region = {}
    if "default" in config:
        config_region.update(config["default"])
    if "obspy" in config:
        config_region.update(config["obspy"])
    if "skypilot" in config:
        config_region["skypilot"] = config["skypilot"]
    if "kubeflow" in config:
        config_region["kubeflow"] = config["kubeflow"]
    if "phasenet" in config:
        config_region["phasenet"] = config["phasenet"]
    if "gamma" in config:
        config_region["gamma"] = config["gamma"]
    if "cctorch" in config:
        config_region["cctorch"] = config["cctorch"]
    if "region" in config:
        if region in config["region"]:
            config_region.update(config["region"][region])

    with open(f"{root_path}/{data_dir}/config.json", "w") as fp:
        json.dump(config_region, fp, indent=4)
    if protocol != "file":
        fs.put(f"{root_path}/{data_dir}/config.json", f"{bucket}/{data_dir}/config.json")
    print(json.dumps(config_region, indent=4))

    return config_region


if __name__ == "__main__":
    import json
    import os
    import sys

    root_path = "local"
    region = "demo"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    with open("config.json", "r") as fp:
        config = json.load(fp)

    num_nodes = len(os.environ["SKYPILOT_NODE_IPS"].split("\n")) if "SKYPILOT_NODE_IPS" in os.environ else 1
    if "skypilot" in config:
        config["skypilot"]["num_nodes"] = num_nodes
    else:
        config["skypilot"] = {"num_nodes": num_nodes}

    set_config.execute(root_path=root_path, region=region, config=config, protocol="file", bucket="", token=None)

    # # %%
    # import os
    # from pathlib import Path

    # from kfp import compiler
    # from kfp.client import Client

    # bucket = "quakeflow_share"
    # protocol = "gs"
    # token = None
    # token_file = "/Users/weiqiang/.config/gcloud/application_default_credentials.json"
    # if os.path.exists(token_file):
    #     with open(token_file, "r") as fp:
    #         token = json.load(fp)

    # yaml_path = Path("yaml")
    # if not yaml_path.exists():
    #     yaml_path.mkdir()

    # @dsl.pipeline
    # def test_set_config():
    #     set_config(root_path=root_path, region=region, config=config, protocol=protocol, bucket=bucket, token=token)

    # client = Client("3a1395ae1e4ad10-dot-us-west1.pipelines.googleusercontent.com")
    # run = client.create_run_from_pipeline_func(
    #     test_set_config,
    #     arguments={},
    # )

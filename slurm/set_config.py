# %%
from typing import Dict

from kfp import dsl


@dsl.component(base_image="zhuwq0/quakeflow:latest")
def set_config(root_path: str, region: str, config: Dict, protocol: str, bucket: str, token: Dict) -> Dict:
    import json
    import os
    from pathlib import Path

    import fsspec

    fs = fsspec.filesystem(protocol, token=token)
    root_path = Path(f"{root_path}/{region}")
    if not root_path.exists():
        root_path.mkdir()
    result_path = root_path / "obspy"
    if not result_path.exists():
        result_path.mkdir()

    config_region = {}
    if "default" in config:
        config_region.update(config["default"])
    if "obspy" in config:
        config_region.update(config["obspy"])
    if "region" in config:
        if region in config["region"]:
            config_region.update(config["region"][region])

    with open(root_path / "config.json", "w") as fp:
        json.dump(config_region, fp, indent=4)
    if protocol != "file":
        fs.put(str(root_path / "config.json"), f"{bucket}/{region}/config.json")
    print(json.dumps(config_region, indent=4))

    return config_region


if __name__ == "__main__":
    import json

    root_path = "./"
    region = "demo"
    with open("config.json", "r") as fp:
        config = json.load(fp)

    set_config.python_func(root_path=root_path, region=region, config=config, protocol="file", bucket="", token=None)

    # %%
    # import os
    # from kfp import compiler
    # from kfp.client import Client
    # from pathlib import Path

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

    # compiler.Compiler().compile(set_config, str(yaml_path / "set_config.yaml"))

    # @dsl.pipeline
    # def test_set_config():
    #     set_config(root_path=root_path, region=region, config=config, protocol=protocol, bucket=bucket, token=token)

    # client = Client("3a1395ae1e4ad10-dot-us-west1.pipelines.googleusercontent.com")
    # run = client.create_run_from_pipeline_func(
    #     test_set_config,
    #     arguments={},
    # )

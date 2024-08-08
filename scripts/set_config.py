# %%
from typing import Dict


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
    for subfolder in [
        "network",
        "waveforms",
        "picking",
        "association",
        "location",
        "relocation",
        "mechanism",
        "results",
    ]:
        if not os.path.exists(f"{root_path}/{data_dir}/results/{subfolder}"):
            os.makedirs(f"{root_path}/{data_dir}/results/{subfolder}", exist_ok=True)

    config_region = {}
    ## default values
    config_region["num_nodes"] = 1
    ## submodules config
    if "default" in config:
        config_region.update(config["default"])
    if "obspy" in config:
        config_region.update(config["obspy"])
    if "phasenet" in config:
        config_region["phasenet"] = config["phasenet"]
    if "gamma" in config:
        config_region["gamma"] = config["gamma"]
    if "adloc" in config:
        config_region["adloc"] = config["adloc"]
    if "cctorch" in config:
        config_region["cctorch"] = config["cctorch"]
    if "adtomo" in config:
        config_region["adtomo"] = config["adtomo"]
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

    set_config(root_path=root_path, region=region, config=config, protocol="file", bucket="", token=None)

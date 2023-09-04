# %%
import json
import os

from kfp import compiler, components, dsl, kubernetes
from kfp.client import Client

if __name__ == "__main__":
    from download_catalog import download_catalog
    from download_station import download_station
    from set_config import set_config

    bucket = "quakeflow_share"
    protocol = "gs"
    token = None
    token_file = "/Users/weiqiang/.config/gcloud/application_default_credentials.json"
    if os.path.exists(token_file):
        with open(token_file, "r") as fp:
            token = json.load(fp)

    yaml_path = "yaml"
    root_path = "/data"
    region = "demo"
    with open(f"config.json", "r") as fp:
        config = json.load(fp)

    compiler.Compiler().compile(set_config, f"{yaml_path}/set_config.yaml")
    compiler.Compiler().compile(download_catalog, f"{yaml_path}/download_catalog.yaml")
    compiler.Compiler().compile(download_station, f"{yaml_path}/download_station.yaml")

    set_config = components.load_component_from_file(f"{yaml_path}/set_config.yaml")
    download_catalog = components.load_component_from_file(f"{yaml_path}/download_catalog.yaml")
    download_station = components.load_component_from_file(f"{yaml_path}/download_station.yaml")

    @dsl.pipeline
    def quakeflow(region: str, config: dict):
        pvc = kubernetes.CreatePVC(
            pvc_name_suffix="-quakeflow",
            # pvc_name="quakeflow",
            access_modes=["ReadWriteOnce"],
            size="5Gi",
            storage_class_name="standard",
        ).set_display_name("Create Storage")
        config = set_config(
            root_path=root_path, region=region, config=config, protocol=protocol, bucket=bucket, token=token
        )
        kubernetes.mount_pvc(config, pvc_name=pvc.outputs["name"], mount_path=root_path)
        catalog = download_catalog(
            root_path=root_path, region=region, config=config.output, protocol=protocol, bucket=bucket, token=token
        )
        kubernetes.mount_pvc(catalog, pvc_name=pvc.outputs["name"], mount_path=root_path)
        station = download_station(
            root_path=root_path, region=region, config=config.output, protocol=protocol, bucket=bucket, token=token
        )
        kubernetes.mount_pvc(station, pvc_name=pvc.outputs["name"], mount_path=root_path)
        delete_pvc = kubernetes.DeletePVC(pvc_name=pvc.outputs["name"]).after(catalog).after(station)

    compiler.Compiler().compile(quakeflow, f"{yaml_path}/quakeflow.yaml")
    client = Client("https://3a1395ae1e4ad10-dot-us-west1.pipelines.googleusercontent.com")
    run = client.create_run_from_pipeline_func(
        quakeflow,
        arguments={"region": region, "config": config},
    )

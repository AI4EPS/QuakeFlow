# %%
import json
import os

from kfp import compiler, components, dsl, kubernetes
from kfp.client import Client

if __name__ == "__main__":
    from download_catalog import download_catalog
    from download_station import download_station
    from download_waveform import download_waveform
    from download_waveform_event import download_waveform_event

    # from download_waveform_v2 import download_waveform
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
    num_nodes = 16  # NCEDC 8
    download_mode = "event"  # "event" or "waveform

    with open(f"config.json", "r") as fp:
        config = json.load(fp)
    config["kubeflow"]["num_nodes"] = num_nodes

    compiler.Compiler().compile(set_config, f"{yaml_path}/set_config.yaml")
    compiler.Compiler().compile(download_catalog, f"{yaml_path}/download_catalog.yaml")
    compiler.Compiler().compile(download_station, f"{yaml_path}/download_station.yaml")
    compiler.Compiler().compile(download_waveform_event, f"{yaml_path}/download_waveform_event.yaml")
    compiler.Compiler().compile(download_waveform, f"{yaml_path}/download_waveform.yaml")

    set_config = components.load_component_from_file(f"{yaml_path}/set_config.yaml")
    download_catalog = components.load_component_from_file(f"{yaml_path}/download_catalog.yaml")
    download_station = components.load_component_from_file(f"{yaml_path}/download_station.yaml")
    download_waveform_event = components.load_component_from_file(f"{yaml_path}/download_waveform_event.yaml")
    download_waveform = components.load_component_from_file(f"{yaml_path}/download_waveform.yaml")

    @dsl.pipeline
    def quakeflow(region: str, config: dict):
        pvc = kubernetes.CreatePVC(
            # pvc_name_suffix="-quakeflow",
            pvc_name="quakeflow",
            access_modes=["ReadWriteOnce"],
            size="5Gi",
            storage_class_name="standard",
        )
        config = set_config(
            root_path=root_path, region=region, config=config, protocol=protocol, bucket=bucket, token=token
        )
        # config.set_caching_options(enable_caching=False)
        kubernetes.mount_pvc(config, pvc_name=pvc.outputs["name"], mount_path=root_path)
        catalog = download_catalog(
            root_path=root_path, region=region, config=config.output, protocol=protocol, bucket=bucket, token=token
        )
        kubernetes.mount_pvc(catalog, pvc_name=pvc.outputs["name"], mount_path=root_path)
        station = download_station(
            root_path=root_path, region=region, config=config.output, protocol=protocol, bucket=bucket, token=token
        )
        kubernetes.mount_pvc(station, pvc_name=pvc.outputs["name"], mount_path=root_path)

        with dsl.ParallelFor(items=list(range(num_nodes)), parallelism=num_nodes) as index:
            pvc_node = kubernetes.CreatePVC(
                pvc_name_suffix="-quakeflow",
                access_modes=["ReadWriteOnce"],
                size="100Gi",
                storage_class_name="standard",
            )

            if download_mode == "event":
                ## Download event waveform
                waveform = download_waveform_event(
                    root_path=root_path,
                    region=region,
                    config=config.output,
                    index=index,
                    protocol=protocol,
                    bucket=bucket,
                    token=token,
                )

            elif download_mode == "waveform":
                ## Download continuous waveform
                waveform = download_waveform(
                    root_path=root_path,
                    region=region,
                    config=config.output,
                    index=index,
                    protocol=protocol,
                    bucket=bucket,
                    token=token,
                )
            else:
                raise ValueError("download_mode must be either 'event' or 'waveform'")
            waveform.set_cpu_request("1100m")
            waveform.set_retry(3)
            waveform.after(catalog).after(station)
            kubernetes.mount_pvc(waveform, pvc_name=pvc_node.outputs["name"], mount_path=root_path)
            kubernetes.DeletePVC(pvc_name=pvc_node.outputs["name"]).after(waveform)

    compiler.Compiler().compile(quakeflow, f"{yaml_path}/quakeflow.yaml")
    client = Client("https://44fd9d51c7dd4225-dot-us-west1.pipelines.googleusercontent.com")
    run = client.create_run_from_pipeline_func(
        quakeflow,
        arguments={"region": region, "config": config},
    )

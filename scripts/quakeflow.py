# %%
import argparse
import json
import os
from typing import Dict, List

from google.cloud import aiplatform
from kfp import compiler, components, dsl, kubernetes
from kfp.client import Client


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default="kubeflow")
    parser.add_argument("--download_mode", type=str, default="waveform")
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--region", type=str, default="demo")
    parser.add_argument("--root_path", type=str, default="/data")
    parser.add_argument("--bucket", type=str, default="quakeflow_share")
    parser.add_argument("--protocol", type=str, default="gs")
    parser.add_argument("--yaml_path", type=str, default="yaml")
    parser.add_argument("--token_file", type=str, default="application_default_credentials.json")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from download_catalog import download_catalog
    from download_station import download_station

    # from download_waveform import download_waveform
    from download_waveform_event import download_waveform_event
    from download_waveform_v2 import download_waveform
    from run_gamma import run_gamma
    from run_phasenet import run_phasenet
    from set_config import set_config

    args = parse_args()
    print(args)
    bucket = args.bucket
    protocol = args.protocol
    token_file = args.token_file
    root_path = args.root_path
    region = args.region
    num_nodes = args.num_nodes
    download_mode = args.download_mode
    platform = args.platform
    yaml_path = args.yaml_path
    if not os.path.exists(yaml_path):
        os.makedirs(yaml_path)

    token = None
    if os.path.exists(token_file):
        with open(token_file, "r") as fp:
            token = json.load(fp)

    with open(f"config.json", "r") as fp:
        config = json.load(fp)
    config["kubeflow"]["num_nodes"] = num_nodes

    # compiler.Compiler().compile(set_config, f"{yaml_path}/set_config.yaml")
    # compiler.Compiler().compile(download_catalog, f"{yaml_path}/download_catalog.yaml")
    # compiler.Compiler().compile(download_station, f"{yaml_path}/download_station.yaml")
    # compiler.Compiler().compile(download_waveform_event, f"{yaml_path}/download_waveform_event.yaml")
    # compiler.Compiler().compile(download_waveform, f"{yaml_path}/download_waveform.yaml")
    # compiler.Compiler().compile(run_phasenet, f"{yaml_path}/run_phasenet.yaml")
    # compiler.Compiler().compile(run_gamma, f"{yaml_path}/run_gamma.yaml")

    # set_config = components.load_component_from_file(f"{yaml_path}/set_config.yaml")
    # download_catalog = components.load_component_from_file(f"{yaml_path}/download_catalog.yaml")
    # download_station = components.load_component_from_file(f"{yaml_path}/download_station.yaml")
    # download_waveform_event = components.load_component_from_file(f"{yaml_path}/download_waveform_event.yaml")
    # download_waveform = components.load_component_from_file(f"{yaml_path}/download_waveform.yaml")
    # run_phasenet = components.load_component_from_file(f"{yaml_path}/run_phasenet.yaml")

    @dsl.pipeline
    def quakeflow(token: Dict):
        if platform == "kubeflow":
            pvc = kubernetes.CreatePVC(
                # pvc_name_suffix="-quakeflow",
                pvc_name="quakeflow",
                access_modes=["ReadWriteOnce"],
                size="5Gi",
                storage_class_name="standard",
            )
        config_ = set_config(
            root_path=root_path, region=region, config=config, protocol=protocol, bucket=bucket, token=token
        )
        # config.set_caching_options(enable_caching=False)
        if platform == "kubeflow":
            kubernetes.mount_pvc(config_, pvc_name=pvc.outputs["name"], mount_path=root_path)
        catalog_ = download_catalog(
            root_path=root_path, region=region, config=config_.output, protocol=protocol, bucket=bucket, token=token
        )
        if platform == "kubeflow":
            kubernetes.mount_pvc(catalog_, pvc_name=pvc.outputs["name"], mount_path=root_path)
        station_ = download_station(
            root_path=root_path, region=region, config=config_.output, protocol=protocol, bucket=bucket, token=token
        )
        if platform == "kubeflow":
            kubernetes.mount_pvc(station_, pvc_name=pvc.outputs["name"], mount_path=root_path)

        with dsl.ParallelFor(items=list(range(num_nodes)), parallelism=num_nodes) as rank:
            if platform == "kubeflow":
                pvc_node = kubernetes.CreatePVC(
                    pvc_name_suffix="-quakeflow",
                    # pvc_name=rank.name,
                    access_modes=["ReadWriteOnce"],
                    size="100Gi",
                    storage_class_name="standard",
                )

            if download_mode == "event":
                ## Download event waveform
                waveform_ = download_waveform_event(
                    root_path=root_path,
                    region=region,
                    config=config_.output,
                    rank=rank,
                    protocol=protocol,
                    bucket=bucket,
                    token=token,
                )

            elif download_mode == "waveform":
                ## Download continuous waveform
                waveform_ = download_waveform(
                    root_path=root_path,
                    region=region,
                    config=config_.output,
                    rank=rank,
                    protocol=protocol,
                    bucket=bucket,
                    token=token,
                )
            else:
                raise ValueError("download_mode must be either 'event' or 'waveform'")
            waveform_.set_cpu_request("1100m")
            waveform_.set_retry(3)
            waveform_.after(catalog_).after(station_)
            if platform == "kubeflow":
                kubernetes.mount_pvc(waveform_, pvc_name=pvc_node.outputs["name"], mount_path=root_path)

            phasenet_ = run_phasenet(
                root_path=root_path,
                region=region,
                config=config_.output,
                node_rank=rank,
                model_path="./",
                protocol=protocol,
                bucket=bucket,
                token=token,
            )
            phasenet_.set_cpu_request("1100m")
            phasenet_.set_retry(3)
            phasenet_.after(waveform_)
            if platform == "kubeflow":
                kubernetes.mount_pvc(phasenet_, pvc_name=pvc_node.outputs["name"], mount_path=root_path)

            gamma_ = run_gamma(
                root_path=root_path,
                region=region,
                config=config_.output,
                node_rank=rank,
                protocol=protocol,
                bucket=bucket,
                token=token,
            )
            gamma_.set_cpu_request("1100m")
            gamma_.set_retry(3)
            gamma_.after(phasenet_)
            # gamma_.after(catalog_).after(station_)
            if platform == "kubeflow":
                kubernetes.mount_pvc(gamma_, pvc_name=pvc_node.outputs["name"], mount_path=root_path)

            if platform == "kubeflow":
                kubernetes.DeletePVC(pvc_name=pvc_node.outputs["name"]).after(gamma_)

        # merge_phasenet_picks_ = merge_op(
        #     folder="results/phase_picking",
        #     fname="phase_picks",
        #     region=region,
        #     config=config_.output,
        #     protocol=protocol,
        #     bucket=bucket,
        #     token=token,
        # )
        # merge_phasenet_picks_.set_display_name("merge_phasenet_picks")
        # merge_phasenet_picks_.after(gamma_)
        # merge_gamma_picks_ = merge_op(
        #     folder="results/phase_association",
        #     fname="gamma_picks",
        #     region=region,
        #     config=config_.output,
        #     protocol=protocol,
        #     bucket=bucket,
        #     token=token,
        # )
        # merge_gamma_picks_.set_display_name("merge_gamma_picks")
        # merge_gamma_picks_.after(gamma_)
        # merge_gamma_events_ = merge_op(
        #     folder="results/phase_association",
        #     fname="gamma_events",
        #     region=region,
        #     config=config_.output,
        #     protocol=protocol,
        #     bucket=bucket,
        #     token=token,
        # )
        # merge_gamma_events_.set_display_name("merge_gamma_events")
        # merge_gamma_events_.after(gamma_)

    compiler.Compiler().compile(quakeflow, f"{yaml_path}/quakeflow.yaml")

    if platform == "kubeflow":
        client = Client("https://707d41eacf1b063c-dot-us-west1.pipelines.googleusercontent.com")
        run = client.create_run_from_pipeline_func(
            quakeflow,
            arguments={"token": token},
            enable_caching=True,
        )
        # run = client.create_run_from_pipeline_package(
        #     f"{yaml_path}/quakeflow.yaml",
        #     arguments={
        #         "token": token,
        #     },
        # )

    elif platform == "vertex":
        aiplatform.init(
            project="quakeflow-385822",
            location="us-west1",
        )

        job = aiplatform.PipelineJob(
            display_name="quakeflow",
            template_path=f"{yaml_path}/quakeflow.yaml",
            parameter_values={"token": token},
        )

        job.submit()

    else:
        raise ValueError("platform must be either 'kubeflow' or 'vertex'")

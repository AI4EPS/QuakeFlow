# %%
import os
import textwrap
from typing import List
import json

from google.cloud import aiplatform
from kfp import compiler, dsl
from kfp.client import Client
from kfp.dsl import Dataset, Input, Output


@dsl.component
def double(num: int, out_dataset: Output[Dataset]):
    with open(out_dataset.path, "w") as f:
        f.write(str(2 * num))


@dsl.component
def add(in_datasets: Input[List[Dataset]], out_dataset: Output[Dataset]):
    nums = []
    for dataset in in_datasets:
        with open(dataset.path) as f:
            nums.append(int(f.read()))
    with open(out_dataset.path, "w") as f:
        f.write(str(sum(nums)))


@dsl.container_component
def add_container(in_datasets: Input[List[Dataset]], out_dataset: Output[Dataset]):
    return dsl.ContainerSpec(
        image="python:3.7",
        command=["python", "-c"],
        args=[
            textwrap.dedent(
                """
            import argparse
            import json
            import os

            def main(in_datasets, out_dataset_uri):
                in_dicts = json.loads(in_datasets)
                uris = [d['uri'] for d in in_dicts]
                total = 0
                for uri in uris:
                    with open(uri.replace('gs://', '/gcs/')) as f:
                        total += int(f.read())

                outpath = out_dataset_uri.replace('gs://', '/gcs/')
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                with open(outpath, 'w') as f:
                    f.write(str(total))

            parser = argparse.ArgumentParser()
            parser.add_argument('in_datasets')
            parser.add_argument('out_dataset_uri')
            args = parser.parse_args()

            main(args.in_datasets, args.out_dataset_uri)
            """
            ),
            in_datasets,
            out_dataset.uri,
        ],
    )


@dsl.pipeline
def math_pipeline() -> List[Dataset]:
    with dsl.ParallelFor([1, 2, 3]) as x:
        t = double(num=x)
    add(in_datasets=dsl.Collected(t.outputs["out_dataset"]))
    add_container(in_datasets=dsl.Collected(t.outputs["out_dataset"]))
    return dsl.Collected(t.outputs["out_dataset"])


if __name__ == "__main__":
    yaml_path = "yaml"

    # compiler.Compiler().compile(pipeline_func=math_pipeline, package_path=f"{yaml_path}/test_fanin.yaml")

    # client = Client("https://44fd9d51c7dd4225-dot-us-west1.pipelines.googleusercontent.com")
    # run = client.create_run_from_pipeline_package(f"{yaml_path}/test_fanin.yaml")

    # aiplatform.init(
    #     project="quakeflow-385822",
    #     location="us-west1",
    # )

    # job = aiplatform.PipelineJob(
    #     display_name="test_fanin",
    #     template_path=f"{yaml_path}/test_fanin.yaml",
    # )

    # job.submit()

    ## TEST QuakeFlow

    root_path = "local"
    region = "demo"
    with open("config.json", "r") as fp:
        config = json.load(fp)
    bucket = "quakeflow_share"
    protocol = "gs"
    token = None
    token_file = "/Users/weiqiang/.config/gcloud/application_default_credentials.json"
    if os.path.exists(token_file):
        with open(token_file, "r") as fp:
            token = json.load(fp)

    aiplatform.init(
        project="quakeflow-385822",
        location="us-west1",
    )

    job = aiplatform.PipelineJob(
        display_name="quakeflow",
        template_path=f"{yaml_path}/quakeflow.yaml",
        # template_path=f"{yaml_path}/set_config.yaml",
        parameter_values={
            "bucket": bucket,
            "config": config,
            "protocol": protocol,
            "region": region,
            "root_path": "/data",
            "token": token,
        },
    )

    job.submit()

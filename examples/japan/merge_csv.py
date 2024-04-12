from typing import Dict, List

from kfp import dsl


@dsl.component(base_image="zhuwq0/quakeflow:latest")
def merge_op(
    root_path: str,
    region: str,
    config: Dict,
    folder: str,
    fname: str,
    file_list: List[str] = None,
    protocol: str = "file",
    bucket: str = "quakeflow_share",
    token: Dict = None,
):
    import fsspec
    import pandas as pd

    fs = fsspec.filesystem(protocol, token=token)

    num_nodes = config["kubeflow"]["num_nodes"] if "kubeflow" in config else 1

    if file_list is None:
        file_list = [f"{folder}/{fname}_{rank:03d}.csv" for rank in range(num_nodes)]

    df_list = []
    for file in file_list:
        if protocol != "file":
            if fs.exists(f"{bucket}/{region}/{file}"):
                print(f"Merging {bucket}/{region}/{file}")
                with fs.open(f"{bucket}/{region}/{file}", "r") as fp:
                    df = pd.read_csv(fp)
                    df["rank"] = file.split("_")[-1].split(".")[0]
                df_list.append(df)
            else:
                print(f"{bucket}/{region}/{file} does not exist")
        else:
            if os.path.exists(f"{root_path}/{region}/{file}"):
                print(f"Merging {root_path}/{region}/{file}")
                df = pd.read_csv(f"{root_path}/{region}/{file}")
                df["rank"] = file.split("_")[-1].split(".")[0]
                df_list.append(df)
            else:
                print(f"{root_path}/{region}/{file} does not exist")

    if len(df_list) > 0:
        df_list = pd.concat(df_list, ignore_index=True)
        print(f"Saving {region}/{folder}/{fname}.csv: {len(df_list)} rows")
        if not os.path.exists(f"{root_path}/{region}/{folder}"):
            os.makedirs(f"{root_path}/{region}/{folder}")
        df_list.to_csv(f"{root_path}/{region}/{folder}/{fname}.csv", index=False)
        if protocol != "file":
            fs.put(f"{root_path}/{region}/{folder}/{fname}.csv", f"{bucket}/{region}/{folder}/{fname}.csv")


@dsl.component(base_image="zhuwq0/quakeflow:latest")
def reindex(
    root_path: str,
    region: str,
    config: Dict,
    picks_csv: str = None,
    events_csv: str = None,
    protocol: str = "file",
    bucket: str = "quakeflow_share",
    token: Dict = None,
):
    import os

    import fsspec
    import pandas as pd

    fs = fsspec.filesystem(protocol, token=token)

    if picks_csv is None:
        picks_csv = f"results/phase_picking/picks.csv"
    if events_csv is None:
        events_csv = f"results/phase_association/events.csv"

    print(f"Reindexing {region}/{picks_csv} and {region}/{events_csv}")
    if protocol != "file":
        if fs.exists(f"{bucket}/{region}/{picks_csv}"):
            with fs.open(f"{bucket}/{region}/{picks_csv}", "r") as fp:
                picks = pd.read_csv(fp)
        else:
            print(f"{bucket}/{region}/{picks_csv} does not exist")
        if fs.exists(f"{bucket}/{region}/{events_csv}"):
            with fs.open(f"{bucket}/{region}/{events_csv}", "r") as fp:
                events = pd.read_csv(fp)
        else:
            print(f"{bucket}/{region}/{events_csv} does not exist")
    else:
        if os.path.exists(f"{root_path}/{region}/{picks_csv}"):
            picks = pd.read_csv(f"{root_path}/{region}/{picks_csv}")
        else:
            print(f"{root_path}/{region}/{picks_csv} does not exist")
        if os.path.exists(f"{root_path}/{region}/{events_csv}"):
            events = pd.read_csv(f"{root_path}/{region}/{events_csv}")
        else:
            print(f"{root_path}/{region}/{events_csv} does not exist")

    events = events.sort_values(["rank", "event_index"])
    events["rank_event_index"] = events["rank"].astype(str) + "_" + events["event_index"].astype(str)
    events.reset_index(inplace=True, drop=True)
    events["event_index"] = events.index
    mapping = dict(zip(events["rank_event_index"], events.index))

    picks["rank_event_index"] = picks["rank"].astype(str) + "_" + picks["event_index"].astype(str)
    picks["event_index"] = picks["rank_event_index"].map(mapping)
    picks["event_index"] = picks["event_index"].fillna(-1)

    events.drop(columns=["rank_event_index"], inplace=True)
    picks.drop(columns=["rank_event_index"], inplace=True)

    os.system(f"mv {root_path}/{region}/{events_csv} {root_path}/{region}/{events_csv}.bak")
    os.system(f"mv {root_path}/{region}/{picks_csv} {root_path}/{region}/{picks_csv}.bak")
    if protocol != "file":
        fs.put(f"{root_path}/{region}/{events_csv}.bak", f"{bucket}/{region}/{events_csv}.bak")
        fs.put(f"{root_path}/{region}/{picks_csv}.bak", f"{bucket}/{region}/{picks_csv}.bak")
    events.to_csv(f"{root_path}/{region}/{events_csv}", index=False)
    picks.to_csv(f"{root_path}/{region}/{picks_csv}", index=False)
    if protocol != "file":
        fs.put(f"{root_path}/{region}/{events_csv}", f"{bucket}/{region}/{events_csv}")
        fs.put(f"{root_path}/{region}/{picks_csv}", f"{bucket}/{region}/{picks_csv}")


if __name__ == "__main__":
    import argparse
    import json
    import os

    import fsspec

    def parse_args():
        parser = argparse.ArgumentParser()
        # parser.add_argument("--root_path", type=str, default="local", help="root path")
        # parser.add_argument("--region", type=str, default="demo", help="region")
        parser.add_argument("root_path", nargs="?", type=str, default="local", help="root path")
        parser.add_argument("region", nargs="?", type=str, default="demo", help="region")
        parser.add_argument("--num_nodes", type=int, default=1)
        parser.add_argument("--bucket", type=str, default="quakeflow_share")
        parser.add_argument("--protocol", type=str, default="file")
        parser.add_argument("--token_file", type=str, default="application_default_credentials.json")
        args = parser.parse_args()
        return args

    args = parse_args()
    root_path = args.root_path
    region = args.region

    if os.path.exists(args.token_file):
        with open(args.token_file, "r") as fp:
            token = json.load(fp)
    else:
        token = None

    if args.protocol == "file":
        with open(f"{root_path}/{region}/config.json", "r") as fp:
            config = json.load(fp)
    else:
        with fsspec.open(f"{args.protocol}://{args.bucket}/{region}/config.json", "r", token=token) as fp:
            config = json.load(fp)

    if "kubeflow" in config:
        config["kubeflow"]["num_nodes"] = args.num_nodes
    else:
        config["kubeflow"] = {"num_nodes": args.num_nodes}

    folders = ["results/phase_picking", "results/phase_association", "results/phase_association"]
    fnames = ["phase_picks", "picks", "events"]
    for folder, fname in zip(folders, fnames):
        merge_op.execute(
            root_path=root_path,
            region=region,
            config=config,
            folder=folder,
            fname=fname,
            protocol=args.protocol,
            token=token,
        )

    reindex.execute(
        root_path=root_path,
        region=region,
        config=config,
        picks_csv="results/phase_association/picks.csv",
        events_csv="results/phase_association/events.csv",
        protocol=args.protocol,
        token=token,
    )

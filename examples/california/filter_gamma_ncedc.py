# %%
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, NamedTuple

import pandas as pd
from kfp import compiler, dsl
from kfp.client import Client


@dsl.component(packages_to_install=["fsspec", "gcsfs", "s3fs", "tqdm", "numpy", "pandas", "scikit-learn"])
def filt_gamma(
    root_path: str,
    region: str,
    config: Dict,
    year: int = 2023,
    jday: int = 0,
    picks_csv: str = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> NamedTuple("outputs", events=str, picks=str):
    import os

    import fsspec
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    from tqdm import tqdm

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    result_path = f"{region}/gamma/{year:04d}"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    station_json = f"{region}/results/network/stations.json"
    gamma_events_csv = f"{result_path}/gamma_events_{jday:03d}.csv"
    gamma_picks_csv = f"{result_path}/gamma_picks_{jday:03d}.csv"

    gamma_events_filt_csv = f"{result_path}/gamma_events_filt_{jday:03d}.csv"
    gamma_picks_filt_csv = f"{result_path}/gamma_picks_filt_{jday:03d}.csv"

    # %%
    if protocol == "file":
        if not os.path.exists(f"{root_path}/{gamma_events_csv}"):
            print(f"{root_path}/{gamma_events_csv} does not exist")
            return NamedTuple("outputs", events=str, picks=str)(
                events=gamma_events_filt_csv, picks=gamma_picks_filt_csv
            )
    else:
        if not fs.exists(f"{bucket}/{gamma_events_csv}"):
            print(f"{bucket}/{gamma_events_csv} does not exist")
            return NamedTuple("outputs", events=str, picks=str)(
                events=gamma_events_filt_csv, picks=gamma_picks_filt_csv
            )

    # %%
    if protocol == "file":
        events = pd.read_csv(f"{root_path}/{gamma_events_csv}", parse_dates=["time"])
        picks = pd.read_csv(f"{root_path}/{gamma_picks_csv}", parse_dates=["phase_time"])
        stations = pd.read_json(f"{root_path}/{station_json}", orient="index")
    else:
        with fs.open(f"{bucket}/{gamma_events_csv}", "rb") as fp:
            events = pd.read_csv(fp, parse_dates=["time"])
        with fs.open(f"{bucket}/{gamma_picks_csv}", "rb") as fp:
            picks = pd.read_csv(fp, parse_dates=["phase_time"])
        with fs.open(f"{bucket}/{station_json}", "rb") as fp:
            stations = pd.read_json(fp, orient="index")
    stations["station_id"] = stations.index

    # $$
    MIN_NEAREST_STATION_RATIO = 0.2
    MIN_PICKS = 5
    MIN_P_PICKS = 3
    MIN_S_PICKS = 3

    # %%
    stations = stations[stations["station_id"].isin(picks["station_id"].unique())]

    neigh = NearestNeighbors(n_neighbors=min(len(stations), 10))
    neigh.fit(stations[["longitude", "latitude"]].values)

    # %%
    picks = picks.merge(events[["event_index", "longitude", "latitude"]], on="event_index", suffixes=("", "_event"))
    picks = picks.merge(stations[["station_id", "longitude", "latitude"]], on="station_id", suffixes=("", "_station"))

    # %%
    filtered_events = []
    for i, event in tqdm(events.iterrows(), total=len(events), desc=f"Filtering {jday:03d}"):
        # sid = neigh.kneighbors([[event["longitude"], event["latitude"]]])[1][0]
        picks_ = picks[picks["event_index"] == event["event_index"]]

        if len(picks_) < MIN_PICKS:
            continue
        if len(picks_[picks_["phase_type"] == "P"]) < MIN_P_PICKS:
            continue
        if len(picks_[picks_["phase_type"] == "S"]) < MIN_S_PICKS:
            continue

        longitude, latitude = picks_[["longitude", "latitude"]].mean().values
        sid = neigh.kneighbors([[longitude, latitude]])[1][0]
        stations_neigh = stations.iloc[sid]["station_id"].values
        picks_neigh = picks_[picks_["station_id"].isin(stations_neigh)]
        stations_with_picks = picks_neigh["station_id"].unique()
        if len(stations_with_picks) / len(stations_neigh) > MIN_NEAREST_STATION_RATIO:
            filtered_events.append(event)

    # %%
    print(f"Events before filtering: {len(events)}")
    print(f"Events after filtering: {len(filtered_events)}")
    filtered_events = pd.DataFrame(filtered_events)
    filtered_events.to_csv(f"{root_path}/{gamma_events_filt_csv}", index=False)

    # filtered_picks = picks[picks["event_index"].isin(filtered_events["event_index"])]
    filtered_picks = picks.copy()
    filtered_picks.loc[~filtered_picks["event_index"].isin(filtered_events["event_index"]), "event_index"] = -1
    filtered_picks.to_csv(f"{root_path}/{gamma_picks_filt_csv}", index=False)

    if protocol != "file":
        fs.put(
            f"{root_path}/{gamma_events_filt_csv}",
            f"{bucket}/{gamma_events_filt_csv}",
        )
        print(f"Uploaded {root_path}/{gamma_events_filt_csv} to {bucket}/{gamma_events_filt_csv}")
        fs.put(
            f"{root_path}/{gamma_picks_filt_csv}",
            f"{bucket}/{gamma_picks_filt_csv}",
        )
        print(f"Uploaded {root_path}/{gamma_events_filt_csv} to {bucket}/{gamma_events_filt_csv}")

    # %% copy to results/phase_association
    if not os.path.exists(f"{root_path}/{region}/results/phase_association"):
        os.makedirs(f"{root_path}/{region}/results/phase_association")
    os.system(
        f"cp {root_path}/{gamma_events_filt_csv} {root_path}/{region}/results/phase_association/events_{jday:03d}.csv"
    )
    os.system(
        f"cp {root_path}/{gamma_picks_filt_csv} {root_path}/{region}/results/phase_association/picks_{jday:03d}.csv"
    )

    if protocol != "file":
        fs.put(
            f"{root_path}/{gamma_events_filt_csv}",
            f"{bucket}/{region}/results/phase_association/events_{jday:03d}.csv",
        )
        print(
            f"Uploaded {root_path}/{gamma_events_filt_csv} to {bucket}/{region}/results/phase_association/events_{jday:03d}.csv"
        )
        fs.put(
            f"{root_path}/{gamma_picks_filt_csv}",
            f"{bucket}/{region}/results/phase_association/picks_{jday:03d}.csv",
        )
        print(
            f"Uploaded {root_path}/{gamma_picks_filt_csv} to {bucket}/{region}/results/phase_association/picks_{jday:03d}.csv"
        )

    outputs = NamedTuple("outputs", events=str, picks=str)
    return outputs(events=gamma_events_filt_csv, picks=gamma_picks_filt_csv)


if __name__ == "__main__":
    import json
    import os
    import sys

    import fsspec

    protocol = "gs"
    token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)
    fs = fsspec.filesystem(protocol=protocol, token=token)

    root_path = "local"
    bucket = "quakeflow_catalog"
    region = "NC"
    with fs.open(f"{bucket}/{region}/config.json", "rb") as fp:
        config = json.load(fp)
    # if len(sys.argv) > 1:
    #     root_path = sys.argv[1]
    #     region = sys.argv[2]
    # with open(f"{root_path}/{region}/config.json", "r") as fp:
    #     config = json.load(fp)

    year = 2023

    ### Local
    # for jday in range(1, 366):
    # # for jday in [200]:
    #     filt_gamma.execute(root_path=root_path, region=region, config=config, year=year, jday=jday, protocol=protocol, bucket="quakeflow_catalog")
    # raise

    ### GCP
    world_size = 1
    jdays = [i for i in range(366)]
    processed = fs.glob(f"{bucket}/{region}/gamma/2023/gamma_events_filt_*.csv")
    processed = [int(p.split("_")[-1].split(".")[0]) for p in processed]
    jdays = [j for j in jdays if j not in processed]
    print(f"{len(jdays) = }")
    # jdays = [201]

    config["world_size"] = world_size

    @dsl.pipeline
    def run_pipeline(root_path: str, region: str, config: Dict, bucket: str, protocol: str, token: Dict = None):
        with dsl.ParallelFor(items=jdays, parallelism=world_size) as item:
            gamma_op = filt_gamma(
                root_path=root_path,
                region=region,
                config=config,
                year=year,
                jday=item,
                bucket=bucket,
                protocol=protocol,
                token=token,
            )
            gamma_op.set_cpu_request("1100m")
            gamma_op.set_memory_request("5000Mi")


    client = Client("https://36ca05fb3e7bbc04-dot-us-west1.pipelines.googleusercontent.com")
    run = client.create_run_from_pipeline_func(
        run_pipeline,
        arguments={
            "region": region,
            "root_path": "./",
            "bucket": "quakeflow_catalog",
            "protocol": protocol,
            "token": token,
            "config": config,
        },
        run_name=f"filter-gamma-{year}",
        enable_caching=False,
    )

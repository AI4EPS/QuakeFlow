# %%
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, NamedTuple

import pandas as pd
from kfp import compiler, dsl
from kfp.client import Client


@dsl.component(packages_to_install=["fsspec", "gcsfs", "s3fs", "tqdm", "numpy", "pyproj", "pandas", "gmma"])
def run_gamma(
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
    import json
    import os

    import fsspec
    import numpy as np
    import pandas as pd
    from gamma.utils import association, estimate_eps
    from pyproj import Proj

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    result_path = f"{region}/gamma/{year:04d}"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    # station_csv = data_path / "stations.csv"
    station_json = f"{region}/results/network/stations.json"
    if picks_csv is None:
        picks_csv = f"{region}/results/phase_picking/{year:04d}/phase_picks_{jday:03d}.csv"
    gamma_events_csv = f"{result_path}/gamma_events_{jday:03d}.csv"
    gamma_picks_csv = f"{result_path}/gamma_picks_{jday:03d}.csv"

    # %%
    ## read picks
    try:
        if protocol == "file":
            picks = pd.read_csv(f"{root_path}/{picks_csv}", parse_dates=["phase_time"])
        else:
            # picks = pd.read_csv(f"{protocol}://{bucket}/{picks_csv}",  parse_dates=["phase_time"])
            with fs.open(f"{bucket}/{picks_csv}", "r") as fp:
                picks = pd.read_csv(fp, parse_dates=["phase_time"])
    except Exception as e:
        print(f"Error reading {picks_csv}: {e}")
        return NamedTuple("outputs", events=str, picks=str)(events=gamma_events_csv, picks=gamma_picks_csv)
    picks.rename(columns={"station_id": "id", "phase_time": "timestamp", "phase_type": "type", "phase_score": "prob", "phase_amplitude": "amp"}, inplace=True)
    # picks["id"] = picks["id"].apply(lambda x: ".".join(x.split(".")[:2])) # remove channel

    ## read stations
    if protocol == "file":
        stations = pd.read_json(f"{root_path}/{station_json}", orient="index")
    else:
        with fs.open(f"{bucket}/{station_json}", "r") as fp:
            stations = pd.read_json(fp, orient="index")
    stations["id"] = stations.index
    # stations["id"] = stations["id"].apply(lambda x: ".".join(x.split(".")[:2])) # remove channel
    # stations = stations.groupby("id").first().reset_index()

    if "longitude0" not in config:
        config["longitude0"] = (config["minlongitude"] + config["maxlongitude"]) / 2
    if "latitude0" not in config:
        config["latitude0"] = (config["minlatitude"] + config["maxlatitude"]) / 2
    proj = Proj(f"+proj=sterea +lon_0={config['longitude0']} +lat_0={config['latitude0']} +units=km")
    stations[["x(km)", "y(km)"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x / 1e3)

    ### setting GMMA configs
    config["use_dbscan"] = True
    config["use_amplitude"] = True
    config["method"] = "BGMM"
    if config["method"] == "BGMM":  ## BayesianGaussianMixture
        config["oversample_factor"] = 5
    if config["method"] == "GMM":  ## GaussianMixture
        config["oversample_factor"] = 1

    # earthquake location
    config["vel"] = {"p": 6.0, "s": 6.0 / 1.75}
    config["dims"] = ["x(km)", "y(km)", "z(km)"]
    config["x(km)"] = (
        np.array([config["minlongitude"] - config["longitude0"], config["maxlongitude"] - config["longitude0"]])
        * config["degree2km"]
        * np.cos(np.deg2rad(config["latitude0"]))
    )
    config["y(km)"] = (
        np.array([config["minlatitude"] - config["latitude0"], config["maxlatitude"] - config["latitude0"]])
        * config["degree2km"]
    )
    if "gamma" not in config:
        config["z(km)"] = (0, 60)
    else:
        config["z(km)"] = [config["gamma"]["zmin_km"], config["gamma"]["zmax_km"]]
    config["bfgs_bounds"] = (
        (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
        (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
        (0, config["z(km)"][1] + 1),  # z
        (None, None),  # t
    )

    # DBSCAN
    # config["dbscan_eps"] = estimate_eps(stations, config["vel"]["p"])  # s
    config["dbscan_eps"] = 10 #s
    config["dbscan_min_samples"] = 3

    ## Eikonal for 1D velocity model
    # zz = [0.0, 5.5, 16.0, 32.0]
    # vp = [5.5, 5.5,  6.7,  7.8]
    # vp_vs_ratio = 1.73
    # vs = [v / vp_vs_ratio for v in vp]
    # h = 0.3
    # # h = 3
    # vel = {"z": zz, "p": vp, "s": vs}
    # config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}


    # set number of cpus
    config["ncpu"] = 32

    # filtering
    config["min_picks_per_eq"] = 5
    config["min_p_picks_per_eq"] = 0
    config["min_s_picks_per_eq"] = 0
    config["max_sigma11"] = 3.0 * 5  # s
    config["max_sigma22"] = 1.0 * 3  # log10(m/s)
    config["max_sigma12"] = 1.0 * 3  # covariance

    ## filter picks without amplitude measurements
    if config["use_amplitude"]:
        picks = picks[picks["amp"] != -1]

    for k, v in config.items():
        print(f"{k}: {v}")

    # %%
    event_idx0 = 0  ## current earthquake index
    assignments = []
    events, assignments = association(picks, stations, config, event_idx0, config["method"])
    event_idx0 += len(events)

    if len(events) > 0:
        ## create catalog
        # events = pd.DataFrame(
        #     events,
        #     columns=["time"]
        #     + config["dims"]
        #     + ["magnitude", "sigma_time", "sigma_amp", "cov_time_amp", "event_index", "gamma_score"],
        # )
        events = pd.DataFrame(events)
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z(km)"]
        events.sort_values("time", inplace=True)
        with open(f"{root_path}/{gamma_events_csv}", "w") as fp:
            events.to_csv(
                fp,
                index=False,
                float_format="%.3f",
                date_format="%Y-%m-%dT%H:%M:%S.%f",
                # columns=[
                #     "time",
                #     "magnitude",
                #     "longitude",
                #     "latitude",
                #     # "depth(m)",
                #     "depth_km",
                #     "sigma_time",
                #     "sigma_amp",
                #     "cov_time_amp",
                #     "event_index",
                #     "gamma_score",
                # ],
            )

        ## add assignment to picks
        assignments = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
        picks = picks.join(assignments.set_index("pick_index")).fillna(-1).astype({"event_index": int})
        picks.rename(columns={"id": "station_id", "timestamp": "phase_time", "type": "phase_type", "prob": "phase_score", "amp": "phase_amplitude"}, inplace=True)
        picks.sort_values(["phase_time"], inplace=True)
        with open(f"{root_path}/{gamma_picks_csv}", "w") as fp:
            picks.to_csv(
                fp,
                index=False,
                date_format="%Y-%m-%dT%H:%M:%S.%f",
                # columns=[
                #     "station_id",
                #     "phase_time",
                #     "phase_type",
                #     "phase_score",
                #     "phase_amplitude",
                #     "event_index",
                #     "gamma_score",
                # ],
            )

        if protocol != "file":
            fs.put(f"{root_path}/{gamma_events_csv}", f"{bucket}/{gamma_events_csv}")
            fs.put(f"{root_path}/{gamma_picks_csv}", f"{bucket}/{gamma_picks_csv}")

    else:
        print(f"No events associated in {picks_csv}")
        with open(f"{root_path}/{gamma_events_csv}", "w") as fp:
            pass
        with open(f"{root_path}/{gamma_picks_csv}", "w") as fp:
            pass

    # %% copy to results/phase_association
    if not os.path.exists(f"{root_path}/{region}/results/phase_association"):
        os.makedirs(f"{root_path}/{region}/results/phase_association")
    os.system(f"cp {root_path}/{gamma_events_csv} {root_path}/{region}/results/phase_association/events_{jday:03d}.csv")
    os.system(f"cp {root_path}/{gamma_picks_csv} {root_path}/{region}/results/phase_association/picks_{jday:03d}.csv")
    if protocol != "file":
        fs.put(
            f"{root_path}/{gamma_events_csv}",
            f"{bucket}/{region}/results/phase_association/events_{jday:03d}.csv",
        )
        print(f"Uploaded {root_path}/{gamma_events_csv} to {bucket}/{region}/results/phase_association/events_{jday:03d}.csv")
        fs.put(
            f"{root_path}/{gamma_picks_csv}",
            f"{bucket}/{region}/results/phase_association/picks_{jday:03d}.csv",
        )
        print(f"Uploaded {root_path}/{gamma_picks_csv} to {bucket}/{region}/results/phase_association/picks_{jday:03d}.csv")

    outputs = NamedTuple("outputs", events=str, picks=str)
    return outputs(events=gamma_events_csv, picks=gamma_picks_csv)


if __name__ == "__main__":
    import json
    import os
    import sys

    import fsspec
    import pandas as pd

    os.environ["OMP_NUM_THREADS"] = "8"

    protocol = "gs"
    token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)

    fs = fsspec.filesystem(protocol, token=token)

    # root_path = "local"
    # region = "ncedc"
    # # if len(sys.argv) > 1:
    # #     root_path = sys.argv[1]
    # #     region = sys.argv[2]
    # with open(f"{root_path}/{region}/config.json", "r") as fp:
    #     config = json.load(fp)

    region = "NC"
    bucket = "quakeflow_catalog"
    root_path = "local"
    with fs.open(f"{bucket}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    year = 2023

    ## Local
    # for jday in range(1, 366)[::-1]:
    # for jday in [200]:
    #     run_gamma.execute(root_path=root_path, region=region, config=config, year=year, jday=jday, protocol=protocol, token=token, bucket="quakeflow_catalog")
    # raise

    ### GCP
    jdays = [i for i in range(1, 366)]
    processed = fs.glob(f"{bucket}/{region}/gamma/{year}/gamma_events_???.csv")
    processed = [int(p.split("_")[-1].split(".")[0]) for p in processed]
    jdays = list(set(jdays) - set(processed))
    print(f"{len(jdays) = }")
    # jdays = [201]

    world_size = min(64, len(jdays))
    config["world_size"] = world_size
    @dsl.pipeline
    def run_pipeline(root_path: str, region: str, config: Dict, bucket:str, protocol:str, token: Dict = None):
        with dsl.ParallelFor(items=jdays, parallelism=world_size) as item:
            gamma_op = run_gamma(
                root_path=root_path,
                region=region,
                config=config,
                year=year,
                jday=item,
                bucket=bucket,
                protocol=protocol,
                token=token,
            )
            gamma_op.set_cpu_request("2100m")
            gamma_op.set_memory_request("12000Mi")

    client = Client("https://4fedc9c19a233c34-dot-us-west1.pipelines.googleusercontent.com")
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
        run_name=f"gamma-{year}",
        enable_caching=False,
    )
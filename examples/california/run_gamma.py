# %%
import argparse
import json
import os
from typing import Dict, NamedTuple

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gamma.utils import association
from pyproj import Proj


def run_gamma(
    root_path: str,
    region: str,
    config: Dict,
    jdays: list,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
):

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    for jday in jdays:
        print(f"Processing {jday}")

        year = int(jday.split(".")[0])
        jday = int(jday.split(".")[1])

        # %%
        result_path = f"{region}/gamma_bo/{year:04d}"
        if not os.path.exists(f"{root_path}/{result_path}"):
            os.makedirs(f"{root_path}/{result_path}")

        # %%
        # station_csv = data_path / "stations.csv"
        # station_json = f"{region}/results/network/stations.json"
        station_json = f"{region}/network/stations.json"
        # if picks_csv is None:
        # picks_csv = f"{region}/results/phase_picking/{year:04d}/phase_picks_{jday:03d}.csv"
        picks_csv = f"{region}/phasenet/{year:04d}/{year:04d}.{jday:03d}.csv"
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

        # ###################
        # # stations = pd.read_json("tests/bo20240616/stations.json", orient="index")
        # # stations["station_id"] = stations.index
        # # gamma_picks = pd.read_csv("tests/bo20240616/gamma_picks_20230101_30min.csv")
        # # phasenet_picks = pd.read_csv("tests/bo20240616/phasenet_picks_20230101_30min.csv")
        # # phasenet_picks["phase_time"] = pd.to_datetime(phasenet_picks["phase_time"])
        # # gamma_picks["phase_time"] = pd.to_datetime(gamma_picks["phase_time"])
        # root_path = "./"
        # gamma_events_csv = f"tests/bo20240616/gamma_events_{jday:03d}.csv"
        # gamma_picks_csv = f"tests/bo20240616/gamma_picks_{jday:03d}.csv"
        # picks = pd.read_csv("tests/bo20240616/phasenet_picks_20230101_30min.csv")
        # ###################

        picks.rename(
            columns={
                "station_id": "id",
                "phase_time": "timestamp",
                "phase_type": "type",
                "phase_score": "prob",
                "phase_amplitude": "amp",
            },
            inplace=True,
        )
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

        # ###################
        # stations = pd.read_json("tests/bo20240616/stations.json", orient="index")
        # stations["id"] = stations.index
        # ###################

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
            config["oversample_factor"] = 2
        if config["method"] == "GMM":  ## GaussianMixture
            config["oversample_factor"] = 1

        ## earthquake location
        config["vel"] = {"p": 6.0, "s": 6.0 / 1.75}
        config["dims"] = ["x(km)", "y(km)", "z(km)"]
        xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
        xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
        config["x(km)"] = (xmin, xmax)
        config["y(km)"] = (ymin, ymax)
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

        ## DBSCAN
        # config["dbscan_eps"] = estimate_eps(stations, config["vel"]["p"])  # s
        eps_year = {
            2023: 10,
            2022: 10.5,
            2021: 10.5,
            2020: 10.5,
            2019: 10.5,
        }
        config["dbscan_eps"] = eps_year[year]
        config["dbscan_min_samples"] = 6

        ## Eikonal for 1D velocity model
        # Southern California
        # zz = [0.0, 5.5, 16.0, 32.0]
        # vp = [5.5, 5.5,  6.7,  7.8]
        # vp_vs_ratio = 1.73
        # vs = [v / vp_vs_ratio for v in vp]
        # Northern California (Gil7)
        zz = [0.0, 1.0, 3.0, 4.0, 5.0, 17.0, 25.0, 62.0]
        vp = [3.2, 3.2, 4.5, 4.8, 5.51, 6.21, 6.89, 7.83]
        vs = [1.5, 1.5, 2.4, 2.78, 3.18, 3.40, 3.98, 4.52]
        h = 0.3
        vel = {"z": zz, "p": vp, "s": vs}
        config["eikonal"] = {
            "vel": vel,
            "h": h,
            "xlim": config["x(km)"],
            "ylim": config["y(km)"],
            "zlim": config["z(km)"],
        }

        ## set number of cpus
        config["ncpu"] = 32

        ## filtering
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
            events = pd.DataFrame(events)
            events[["longitude", "latitude"]] = events.apply(
                lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1
            )
            events["depth_km"] = events["z(km)"]
            events.sort_values("time", inplace=True)
            with open(f"{root_path}/{gamma_events_csv}", "w") as fp:
                events.to_csv(fp, index=False, float_format="%.3f", date_format="%Y-%m-%dT%H:%M:%S.%f")

            # plt.figure()
            # plt.scatter(
            #     events["longitude"],
            #     events["latitude"],
            #     c=events["depth_km"],
            #     s=max(0.1, min(10, 5000 / len(events))),
            #     alpha=0.3,
            #     linewidths=0,
            #     cmap="viridis_r",
            # )
            # plt.colorbar()
            # plt.title(f"Number of events: {len(events)}")
            # plt.savefig(f"{root_path}/{gamma_events_csv.replace('.csv', '')}_scaler60_oversample8.png")

            ## add assignment to picks
            assignments = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
            picks = picks.join(assignments.set_index("pick_index")).fillna(-1).astype({"event_index": int})
            picks.rename(
                columns={
                    "id": "station_id",
                    "timestamp": "phase_time",
                    "type": "phase_type",
                    "prob": "phase_score",
                    "amp": "phase_amplitude",
                },
                inplace=True,
            )
            picks.sort_values(["phase_time"], inplace=True)
            with open(f"{root_path}/{gamma_picks_csv}", "w") as fp:
                picks.to_csv(fp, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")

            if protocol != "file":
                fs.put(f"{root_path}/{gamma_events_csv}", f"{bucket}/{gamma_events_csv}")
                fs.put(f"{root_path}/{gamma_picks_csv}", f"{bucket}/{gamma_picks_csv}")

        else:
            print(f"No events associated in {picks_csv}")
            with open(f"{root_path}/{gamma_events_csv}", "w") as fp:
                pass
            with open(f"{root_path}/{gamma_picks_csv}", "w") as fp:
                pass

        # # %% copy to results/phase_association
        # if not os.path.exists(f"{root_path}/{region}/results/phase_association"):
        #     os.makedirs(f"{root_path}/{region}/results/phase_association")
        # os.system(
        #     f"cp {root_path}/{gamma_events_csv} {root_path}/{region}/results/phase_association/events_{jday:03d}.csv"
        # )
        # os.system(
        #     f"cp {root_path}/{gamma_picks_csv} {root_path}/{region}/results/phase_association/picks_{jday:03d}.csv"
        # )
        # if protocol != "file":
        #     fs.put(
        #         f"{root_path}/{gamma_events_csv}",
        #         f"{bucket}/{region}/results/phase_association/events_{jday:03d}.csv",
        #     )
        #     print(
        #         f"Uploaded {root_path}/{gamma_events_csv} to {bucket}/{region}/results/phase_association/events_{jday:03d}.csv"
        #     )
        #     fs.put(
        #         f"{root_path}/{gamma_picks_csv}",
        #         f"{bucket}/{region}/results/phase_association/picks_{jday:03d}.csv",
        #     )
        #     print(
        #         f"Uploaded {root_path}/{gamma_picks_csv} to {bucket}/{region}/results/phase_association/picks_{jday:03d}.csv"
        #     )

    # outputs = NamedTuple("outputs", events=str, picks=str)
    # return outputs(events=gamma_events_csv, picks=gamma_picks_csv)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gamma on NCEDC/SCEDC data")
    parser.add_argument("--num_nodes", type=int, default=366)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--root_path", type=str, default="local")
    parser.add_argument("--region", type=str, default="Cal")
    parser.add_argument("--bucket", type=str, default="quakeflow_catalog")
    return parser.parse_args()


if __name__ == "__main__":

    os.environ["OMP_NUM_THREADS"] = "8"

    args = parse_args()

    protocol = "gs"
    token_json = f"application_default_credentials.json"
    # token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)

    fs = fsspec.filesystem(protocol, token=token)

    region = args.region
    root_path = args.root_path
    bucket = args.bucket
    num_nodes = args.num_nodes
    node_rank = args.node_rank
    year = args.year

    # %%
    calc_jdays = lambda year: 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
    jdays = [f"{year}.{i:03d}" for i in range(1, calc_jdays(year) + 1)]
    jdays = [jdays[i::num_nodes] for i in range(num_nodes)]

    # %%
    with fs.open(f"{bucket}/{region}/config.json", "r") as fp:
        config = json.load(fp)
    config["world_size"] = num_nodes

    # %%
    print(f"{jdays[node_rank] = }")
    run_gamma(
        root_path=root_path,
        region=region,
        config=config,
        jdays=jdays[node_rank],
        protocol=protocol,
        token=token,
        bucket="quakeflow_catalog",
    )

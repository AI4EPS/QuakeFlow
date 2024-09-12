# %%
import argparse
import json
import multiprocessing as mp
import os
from typing import Dict, List, NamedTuple

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adloc.eikonal2d import init_eikonal2d
from adloc.sacloc2d import ADLoc
from adloc.utils import invert_location, invert_location_iter

# from utils import plotting_ransac
from plotting import plotting, plotting_ransac
from pyproj import Proj


# %%
def run_adloc(
    root_path: str,
    region: str,
    config: Dict,
    jdays: list,
    picks_csv: str = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
):

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    for jday in jdays:
        print(f"Processing jday: {jday}")

        year = int(jday.split(".")[0])
        jday = int(jday.split(".")[1])

        # %%
        data_path = f"{region}/gamma_bo_debug/{year:04d}"
        result_path = f"{region}/adloc/{year:04d}"
        if not os.path.exists(f"{root_path}/{result_path}"):
            os.makedirs(f"{root_path}/{result_path}")
        figure_path = f"{region}/figures/{year:04d}"
        if not os.path.exists(f"{root_path}/{figure_path}"):
            os.makedirs(f"{root_path}/{figure_path}")

        # %%
        station_json = f"{region}/network/stations.json"
        picks_csv = f"{data_path}/gamma_picks_{jday:03d}.csv"
        events_csv = f"{data_path}/gamma_events_{jday:03d}.csv"

        adloc_events_csv = f"{result_path}/adloc_events_{jday:03d}.csv"
        adloc_picks_csv = f"{result_path}/adloc_picks_{jday:03d}.csv"

        try:
            if protocol == "file":
                picks = pd.read_csv(f"{root_path}/{picks_csv}", parse_dates=["phase_time"])
                events = pd.read_csv(f"{root_path}/{events_csv}", parse_dates=["time"])
            else:
                with fs.open(f"{bucket}/{picks_csv}") as fp:
                    picks = pd.read_csv(fp, parse_dates=["phase_time"])
                with fs.open(f"{bucket}/{events_csv}") as fp:
                    events = pd.read_csv(fp, parse_dates=["time"])
        except Exception as e:
            print(f"Error reading {picks_csv}: {e}")
            return

        events = None

        # picks["phase_time"] = pd.to_datetime(picks["phase_time"])
        # events["time"] = pd.to_datetime(events["time"])

        # drop unnecessary columns
        picks.drop(["id", "timestamp", "type", "amp", "prob", "event_idx"], axis=1, inplace=True, errors="ignore")

        # # increase weight for P/S pairs
        # phase_counts = (
        #     picks.groupby(["event_index", "station_id"])["phase_type"].nunique().reset_index(name="phase_count")
        # )
        # merged = picks.merge(phase_counts, on=["event_index", "station_id"])
        # merged.loc[merged["phase_count"] == 2, "phase_score"] *= 1.5
        # picks = merged.drop(columns=["phase_count"])

        # stations = pd.read_csv(stations_file, sep="\t")
        if protocol == "file":
            stations = pd.read_json(station_json, orient="index")
        else:
            with fs.open(f"{bucket}/{station_json}") as fp:
                stations = pd.read_json(fp, orient="index")

        stations["station_id"] = stations.index
        stations.reset_index(drop=True, inplace=True)

        # %%
        config["mindepth"] = 0.0
        config["maxdepth"] = 30.0
        config["use_amplitude"] = True

        # %%
        ## Automatic region; you can also specify a region
        # lon0 = stations["longitude"].median()
        # lat0 = stations["latitude"].median()
        lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
        lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
        proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")

        # %%
        stations["depth_km"] = -stations["elevation_m"] / 1000
        if "station_term_time" not in stations.columns:
            stations["station_term_time"] = 0.0
        if "station_term_amplitude" not in stations.columns:
            stations["station_term_amplitude"] = 0.0
        stations[["x_km", "y_km"]] = stations.apply(
            lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
        )
        stations["z_km"] = stations["elevation_m"].apply(lambda x: -x / 1e3)

        if events is not None:
            events[["x_km", "y_km"]] = events.apply(
                lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
            )
            events["z_km"] = events["depth_km"]

        ## set up the config; you can also specify the region manually
        if ("xlim_km" not in config) or ("ylim_km" not in config) or ("zlim_km" not in config):

            # project minlatitude, maxlatitude, minlongitude, maxlongitude to ymin, ymax, xmin, xmax
            xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
            xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
            zmin, zmax = config["mindepth"], config["maxdepth"]
            config["xlim_km"] = (xmin, xmax)
            config["ylim_km"] = (ymin, ymax)
            config["zlim_km"] = (zmin, zmax)

        config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}

        # %%
        config["eikonal"] = None

        # ## Eikonal for 1D velocity model
        # zz = [0.0, 5.5, 16.0, 32.0]
        # vp = [5.5, 5.5, 6.7, 7.8]
        # vp_vs_ratio = 1.73
        # vs = [v / vp_vs_ratio for v in vp]
        # Northern California (Gil7)
        zz = [0.0, 1.0, 3.0, 4.0, 5.0, 17.0, 25.0, 62.0]
        vp = [3.2, 3.2, 4.5, 4.8, 5.51, 6.21, 6.89, 7.83]
        vs = [1.5, 1.5, 2.4, 2.78, 3.18, 3.40, 3.98, 4.52]
        h = 0.3

        vel = {"Z": zz, "P": vp, "S": vs}
        config["eikonal"] = {
            "vel": vel,
            "h": h,
            "xlim_km": config["xlim_km"],
            "ylim_km": config["ylim_km"],
            "zlim_km": config["zlim_km"],
        }
        config["eikonal"] = init_eikonal2d(config["eikonal"])

        # %% config for location
        config["min_picks"] = 6  # for sampling not for filtering
        config["min_picks_ratio"] = 0.5  # for sampling
        config["max_residual_time"] = 1.0
        config["max_residual_amplitude"] = 1.0
        config["min_score"] = 0.5
        config["min_p_picks"] = 0  # for filtering
        config["min_s_picks"] = 0  # for filtering

        config["bfgs_bounds"] = (
            (config["xlim_km"][0] - 1, config["xlim_km"][1] + 1),  # x
            (config["ylim_km"][0] - 1, config["ylim_km"][1] + 1),  # y
            # (config["zlim_km"][0], config["zlim_km"][1] + 1),  # z
            (0, config["zlim_km"][1] + 1),
            (None, None),  # t
        )

        # %%
        mapping_phase_type_int = {"P": 0, "S": 1}
        config["vel"] = {mapping_phase_type_int[k]: v for k, v in config["vel"].items()}
        picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)
        if "phase_amplitude" in picks.columns:
            picks["phase_amplitude"] = picks["phase_amplitude"].apply(
                lambda x: np.log10(x) + 2.0
            )  # convert to log10(cm/s)

        # %%
        # reindex in case the index does not start from 0 or is not continuous
        stations["idx_sta"] = np.arange(len(stations))
        if events is not None:
            # reindex in case the index does not start from 0 or is not continuous
            events["idx_eve"] = np.arange(len(events))

        else:
            picks = picks.merge(stations[["station_id", "x_km", "y_km", "z_km"]], on="station_id")
            events = picks.groupby("event_index").agg(
                {"x_km": "mean", "y_km": "mean", "z_km": "mean", "phase_time": "min"}
            )
            picks.drop(["x_km", "y_km", "z_km"], axis=1, inplace=True)
            events["z_km"] = 10.0  # km default depth
            events.rename({"phase_time": "time"}, axis=1, inplace=True)
            events["event_index"] = events.index
            events.reset_index(drop=True, inplace=True)
            events["idx_eve"] = np.arange(len(events))

        picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
        picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

        # %%
        estimator = ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, eikonal=config["eikonal"])

        # %%
        NCPU = mp.cpu_count()
        MAX_SST_ITER = 2
        # MIN_SST_S = 0.01
        events_init = events.copy()

        # plotting_ransac(
        #     stations,
        #     f"{root_path}/{figure_path}",
        #     config,
        #     picks,
        #     events_init,
        #     events_init,
        #     suffix=f"_ransac_sst_{iter}",
        # )

        stations["station_term"] = 0.0
        plotting(
            stations,
            f"{root_path}/{figure_path}",
            config,
            picks,
            events_init,
            events_init,
            suffix=f"_ransac_sst_init",
        )

        for iter in range(MAX_SST_ITER):
            if iter == 0:
                config["min_picks"] = 6  # for sampling not for filtering
                config["min_picks_ratio"] = 0.5  # for sampling
                config["max_residual_time"] = 3.0
                config["max_residual_amplitude"] = 3.0
                config["min_score"] = -0.5
                config["min_p_picks"] = 0  # for filtering
                config["min_s_picks"] = 0  # for filtering
            else:
                config["min_picks"] = 6  # for sampling not for filtering
                config["min_picks_ratio"] = 0.5  # for sampling
                config["max_residual_time"] = 1.0
                config["max_residual_amplitude"] = 1.0
                config["min_score"] = 0.5
                config["min_p_picks"] = 0  # for filtering
                config["min_s_picks"] = 0  # for filtering
            stations["station_term_time"] = 0.0  # no station term
            # picks, events = invert_location_iter(picks, stations, config, estimator, events_init=events_init, iter=iter)
            if iter == 0:
                picks, events = invert_location(picks, stations, config, estimator, events_init=events_init, iter=iter)
            else:
                # picks, events = invert_location(picks, stations, config, estimator, events_init=events_init, iter=iter)
                picks, events = invert_location(
                    picks[picks["mask"] == 1], stations, config, estimator, events_init=events_init, iter=iter
                )
            # station_term = picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_time": "mean"}).reset_index()
            station_term_time = (
                picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_time": "mean"}).reset_index()
            )
            station_term_amp = (
                picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_amplitude": "mean"}).reset_index()
            )
            stations["station_term_time"] += (
                stations["idx_sta"].map(station_term_time.set_index("idx_sta")["residual_time"]).fillna(0)
            )
            stations["station_term_amplitude"] += (
                stations["idx_sta"].map(station_term_amp.set_index("idx_sta")["residual_amplitude"]).fillna(0)
            )
            ## Separate P and S station term
            # station_term = (
            #     picks[picks["mask"] == 1.0].groupby(["idx_sta", "phase_type"]).agg({"residual": "mean"}).reset_index()
            # )
            # stations["station_term_p"] = (
            #     stations["idx_sta"]
            #     .map(station_term[station_term["phase_type"] == 0].set_index("idx_sta")["residual"])
            #     .fillna(0)
            # )
            # stations["station_term_s"] = (
            #     stations["idx_sta"]
            #     .map(station_term[station_term["phase_type"] == 1].set_index("idx_sta")["residual"])
            #     .fillna(0)
            # )

            # plotting_ransac(
            #     stations,
            #     f"{root_path}/{figure_path}",
            #     config,
            #     picks,
            #     events_init,
            #     events,
            #     suffix=f"_ransac_sst_{iter}_{config['min_picks']}_s{config['min_score']}_r{config['min_picks_ratio']}_p{config['min_p_picks']}_s{config['min_s_picks']}",
            # )

            if iter == 0:
                MIN_SST_S = (
                    np.mean(np.abs(station_term_time["residual_time"])) / 10.0
                )  # break at 10% of the initial station term
                print(f"MIN_SST (s): {MIN_SST_S}")
            if np.mean(np.abs(station_term_time["residual_time"])) < MIN_SST_S:
                print(f"Mean station term: {np.mean(np.abs(station_term_time['residual_time']))}")
                # break
            iter += 1

        # plotting_ransac(
        #     stations, f"{root_path}/{figure_path}", config, picks, events_init, events, suffix=f"_ransac_sst"
        # )

        # %%
        if "event_index" not in events.columns:
            events["event_index"] = events.merge(picks[["idx_eve", "event_index"]], on="idx_eve")["event_index"]
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z_km"]
        events.drop(["idx_eve", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
        events.sort_values(["time"], inplace=True)

        picks.rename({"mask": "adloc_mask", "residual_time": "adloc_residual_time"}, axis=1, inplace=True)
        if "residual_amplitude" in picks.columns:
            picks.rename({"residual_amplitude": "adloc_residual_amplitude"}, axis=1, inplace=True)
        picks["phase_type"] = picks["phase_type"].map({0: "P", 1: "S"})
        picks.drop(["idx_eve", "idx_sta"], axis=1, inplace=True, errors="ignore")
        picks.sort_values(["phase_time"], inplace=True)

        # stations.drop(["idx_sta", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
        # stations.rename({"station_term": "adloc_station_term_s"}, axis=1, inplace=True)

        # picks.to_csv(os.path.join(result_path, "ransac_picks.csv"), index=False)
        # events.to_csv(os.path.join(result_path, "ransac_events.csv"), index=False)
        # stations.to_csv(os.path.join(result_path, "ransac_stations.csv"), index=False)
        picks.to_csv(f"{root_path}/{adloc_picks_csv}", index=False)
        events.to_csv(f"{root_path}/{adloc_events_csv}", index=False)
        # stations.to_json(f"{root_path}/{result_path}/adloc_stations_{jday:03d}.json", orient="index")

        if protocol != "file":
            fs.put(f"{root_path}/{adloc_picks_csv}", f"{bucket}/{adloc_picks_csv}")
            fs.put(f"{root_path}/{adloc_events_csv}", f"{bucket}/{adloc_events_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ADLoc on NCEDC/SCEDC data")
    parser.add_argument("--num_nodes", type=int, default=366)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--root_path", type=str, default="local")
    parser.add_argument("--region", type=str, default="Cal")
    parser.add_argument("--bucket", type=str, default="quakeflow_catalog")
    return parser.parse_args()


# %%
if __name__ == "__main__":

    args = parse_args()

    protocol = "gs"
    token_json = "application_default_credentials.json"
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
    run_adloc(
        root_path=root_path,
        region=region,
        config=config,
        jdays=jdays[node_rank],
        protocol=protocol,
        token=token,
        bucket="quakeflow_catalog",
    )

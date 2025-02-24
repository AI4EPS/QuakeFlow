# %%
import argparse
import json
import multiprocessing as mp
import os
from glob import glob
from typing import Dict, List, NamedTuple

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adloc.eikonal2d import init_eikonal2d
from adloc.sacloc2d import ADLoc
from adloc.utils import invert_location, invert_location_iter
from args import parse_args
from pyproj import Proj

# from utils import plotting_ransac
from utils.plotting import plotting, plotting_ransac


# %%
def run_adloc(
    root_path: str,
    region: str,
    config: Dict,
    jdays: list,
    iter: int = 0,
    node_rank: int = 0,
    num_nodes: int = 1,
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
        data_path = f"{region}/gamma/{year:04d}"
        result_path = f"{region}/adloc/{year:04d}"
        if not os.path.exists(f"{root_path}/{result_path}"):
            os.makedirs(f"{root_path}/{result_path}")
        figure_path = f"{root_path}/{region}/adloc/figures/{year:04d}"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        # %%
        if iter == 0:
            # station_json = f"{region}/obspy/stations.csv"
            stations_csv = f"{region}/obspy/stations.csv"
            # picks_csv = f"{data_path}/{year:04d}.{jday:03d}.picks.csv"
            # events_csv = f"{data_path}/{year:04d}.{jday:03d}.events.csv"
        else:
            # station_json = f"{result_path}/stations_sst_{iter-1}.csv"
            # stations_csv = f"{result_path}/{year:04d}.{jday:03d}.stations_sst_{iter-1}.csv"
            stations_csv = f"{region}/adloc/adloc_stations_sst_{iter-1}.csv"
            # picks_csv = f"{result_path}/{year:04d}.{jday:03d}.picks_sst_{iter-1}.csv"
            # events_csv = f"{result_path}/{year:04d}.{jday:03d}.events_sst_{iter-1}.csv"

        picks_csv = f"{data_path}/{year:04d}.{jday:03d}.picks.csv"
        events_csv = f"{data_path}/{year:04d}.{jday:03d}.events.csv"

        adloc_events_csv = f"{result_path}/{year:04d}.{jday:03d}.events_sst_{iter}.csv"
        adloc_picks_csv = f"{result_path}/{year:04d}.{jday:03d}.picks_sst_{iter}.csv"
        # stations_csv = f"{result_path}/stations_sst_{iter}.csv"
        station_term_csv = f"{result_path}/{year:04d}.{jday:03d}.stations_sst_{iter}.csv"

        if os.path.exists(f"{root_path}/{adloc_events_csv}") and os.path.exists(f"{root_path}/{adloc_picks_csv}"):
            print(f"Skipping {year}.{jday:03d} because {adloc_events_csv} and {adloc_picks_csv} exist")
            continue

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
            # stations = pd.read_json(f"{root_path}/{station_json}", orient="index")
            stations = pd.read_csv(f"{root_path}/{stations_csv}")
        else:
            with fs.open(f"{bucket}/{station_json}") as fp:
                stations = pd.read_json(fp, orient="index")

        # stations = stations[["station_id", "longitude", "latitude", "elevation_m", "num_picks"]]
        # stations["station_id"] = stations.index
        # stations.reset_index(drop=True, inplace=True)

        # %%
        config["mindepth"] = config["mindepth"] if "mindepth" in config else 0.0
        config["maxdepth"] = config["maxdepth"] if "maxdepth" in config else 60.0
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
        if "station_term_time_p" not in stations.columns:
            stations["station_term_time_p"] = 0.0
        if "station_term_time_s" not in stations.columns:
            stations["station_term_time_s"] = 0.0
        if "station_term_amplitude" not in stations.columns:
            stations["station_term_amplitude"] = 0.0
        stations["num_picks"] = 0  ## used to calcuate average station terms across jdays
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
        zz = [0.0, 5.5, 16.0, 32.0]
        vp = [5.5, 5.5, 6.7, 7.8]
        vp_vs_ratio = 1.73
        vs = [v / vp_vs_ratio for v in vp]
        # Northern California (Gil7)
        # zz = [0.0, 1.0, 3.0, 4.0, 5.0, 17.0, 25.0, 62.0]
        # vp = [3.2, 3.2, 4.5, 4.8, 5.51, 6.21, 6.89, 7.83]
        # vs = [1.5, 1.5, 2.4, 2.78, 3.18, 3.40, 3.98, 4.52]
        h = 0.3

        if os.path.exists(f"{root_path}/{region}/obspy/velocity.csv"):
            velocity = pd.read_csv(f"{root_path}/{region}/obspy/velocity.csv")
            zz = velocity["z_km"].values
            vp = velocity["vp"].values
            vs = velocity["vs"].values
            h = 0.1

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
        config["min_p_picks"] = 1.5  # for filtering
        config["min_s_picks"] = 1.5  # for filtering

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
        # MAX_SST_ITER = 1
        # MIN_SST_S = 0.01
        events_init = events.copy()

        if True:
            # for iter in range(MAX_SST_ITER):
            # picks, events = invert_location_iter(picks, stations, config, estimator, events_init=events_init, iter=iter)
            picks, events = invert_location(
                picks, stations, config, estimator, events_init=events_init, iter=f"{year}.{jday}"
            )

            if picks is None or events is None:
                print(f"No events located for {jday}")
                os.system(f"touch {root_path}/{adloc_events_csv}")
                os.system(f"touch {root_path}/{adloc_picks_csv}")
                continue

            # %%
            station_num_picks = (
                picks[picks["mask"] == 1.0].groupby(["idx_sta", "phase_type"]).size().reset_index(name="num_picks")
            )
            station_num_picks.set_index("idx_sta", inplace=True)
            stations["num_pick_p"] = (
                stations["idx_sta"].map(station_num_picks[station_num_picks["phase_type"] == 0]["num_picks"]).fillna(0)
            )
            stations["num_pick_s"] = (
                stations["idx_sta"].map(station_num_picks[station_num_picks["phase_type"] == 1]["num_picks"]).fillna(0)
            )
            stations["num_pick"] = stations["num_pick_p"] + stations["num_pick_s"]

            # %%
            station_term_amp = (
                picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_amplitude": "median"}).reset_index()
            )
            station_term_amp.set_index("idx_sta", inplace=True)
            stations["station_term_amplitude"] = (
                stations["idx_sta"].map(station_term_amp["residual_amplitude"]).fillna(0)
            )

            ## Same P and S station term
            # station_term_time = picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_time": "mean"}).reset_index()
            # stations["station_term_time_p"] += (
            #     stations["idx_sta"].map(station_term_time.set_index("idx_sta")["residual_time"]).fillna(0)
            # )
            # stations["station_term_time_s"] += (
            #     stations["idx_sta"].map(station_term_time.set_index("idx_sta")["residual_time"]).fillna(0)
            # )

            ## Separate P and S station term
            station_term_time = (
                picks[picks["mask"] == 1.0]
                .groupby(["idx_sta", "phase_type"])
                .agg({"residual_time": "mean"})
                .reset_index()
            )
            station_term_time.set_index("idx_sta", inplace=True)
            stations["station_term_time_p"] = (
                stations["idx_sta"]
                .map(station_term_time[station_term_time["phase_type"] == 0]["residual_time"])
                .fillna(0)
            )
            stations["station_term_time_s"] = (
                stations["idx_sta"]
                .map(station_term_time[station_term_time["phase_type"] == 1]["residual_time"])
                .fillna(0)
            )

        # %%
        plotting_ransac(
            stations, figure_path, config, picks, events_init, events, suffix=f"_{year}.{jday:03d}_sst_{iter}"
        )

        # %%
        if "event_index" not in events.columns:
            events["event_index"] = events.merge(picks[["idx_eve", "event_index"]], on="idx_eve")["event_index"]
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z_km"]
        events.drop(["idx_eve", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
        events.sort_values(["time"], inplace=True)

        picks["phase_amplitude"] = 10 ** (picks["phase_amplitude"] - 2.0)
        picks.rename({"mask": "adloc_mask", "residual_time": "adloc_residual_time"}, axis=1, inplace=True)
        if "residual_amplitude" in picks.columns:
            picks.rename({"residual_amplitude": "adloc_residual_amplitude"}, axis=1, inplace=True)
        picks["phase_type"] = picks["phase_type"].map({0: "P", 1: "S"})
        picks.drop(["idx_eve", "idx_sta"], axis=1, inplace=True, errors="ignore")
        picks.sort_values(["phase_time"], inplace=True)

        stations.drop(["idx_sta", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")

        picks.to_csv(f"{root_path}/{adloc_picks_csv}", index=False)
        events.to_csv(f"{root_path}/{adloc_events_csv}", index=False)
        stations.to_csv(f"{root_path}/{station_term_csv}", index=False)

        if protocol != "file":
            fs.put(f"{root_path}/{adloc_picks_csv}", f"{bucket}/{adloc_picks_csv}")
            fs.put(f"{root_path}/{adloc_events_csv}", f"{bucket}/{adloc_events_csv}")


# %%
if __name__ == "__main__":

    args = parse_args()

    # protocol = "gs"
    # token_json = "application_default_credentials.json"
    # # token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    # with open(token_json, "r") as fp:
    #     token = json.load(fp)

    # fs = fsspec.filesystem(protocol, token=token)

    protocol = "file"
    token = None
    fs = fsspec.filesystem(protocol)

    region = args.region
    root_path = args.root_path
    bucket = args.bucket
    num_nodes = args.num_nodes
    node_rank = args.node_rank
    iter = args.iter

    # %%
    jdays = sorted(glob(f"{root_path}/{region}/gamma/????/????.???.events.csv"))
    jdays = [jday.split("/")[-1].replace(".csv", "") for jday in jdays]
    print(f"Number of pick files: {len(jdays)}")
    jdays = [jdays[i::num_nodes] for i in range(num_nodes)]

    # %%
    if protocol == "file":
        with open(f"{root_path}/{region}/config.json", "r") as fp:
            config = json.load(fp)
    else:
        with fs.open(f"{bucket}/{region}/config.json", "r") as fp:
            config = json.load(fp)

    # %%
    print(f"{jdays[node_rank] = }")

    if iter > 1:
        run_adloc(
            root_path=root_path,
            region=region,
            config=config,
            jdays=jdays[node_rank],
            iter=iter,
        )
        os.system(
            f"python merge_adloc_picks.py --region {region} --root_path {root_path} --bucket {bucket} --iter {iter}"
        )

    else:
        if num_nodes == 1:
            max_iter = 10
        else:
            max_iter = 1
        for i in range(max_iter):
            run_adloc(
                root_path=root_path,
                region=region,
                config=config,
                jdays=jdays[node_rank],
                iter=i,
                protocol=protocol,
                token=token,
                bucket=bucket,
            )
            os.system(
                f"python merge_adloc_picks.py --region {region} --root_path {root_path} --bucket {bucket} --iter {i}"
            )

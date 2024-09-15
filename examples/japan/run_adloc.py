# %%
import json
import multiprocessing as mp
import os
import sys
from typing import Dict, List, NamedTuple

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adloc.eikonal2d import init_eikonal2d
from adloc.sacloc2d import ADLoc
from adloc.utils import invert_location, invert_location_iter

# from utils import plotting_ransac
from plotting import plotting_ransac
from pyproj import Proj


# %%
def run_adloc(
    root_path: str,
    region: str,
    config: Dict,
    node_rank: int = 0,
    num_nodes: int = 1,
    picks_csv: str = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
):

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    result_path = f"{root_path}/{region}/adloc"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    figure_path = f"{root_path}/{region}/adloc/figures"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # %%
    data_path = f"{root_path}/{region}/gamma"
    picks_file = os.path.join(data_path, f"gamma_picks.csv")
    events_file = os.path.join(data_path, f"gamma_events.csv")
    # stations_file = os.path.join(data_path, "stations.csv")
    # stations_file = f"{root_path}/{region}/obspy/stations.json"
    stations_file = f"{root_path}/{region}/results/data/stations.json"

    picks = pd.read_csv(picks_file)
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    events = pd.read_csv(events_file)
    events["time"] = pd.to_datetime(events["time"])

    # drop unnecessary columns
    picks.drop(["id", "timestamp", "type", "amp", "prob", "event_idx"], axis=1, inplace=True, errors="ignore")
    # picks = picks[picks["phase_time"] < pd.to_datetime("2019-07-05 00:00:00")]
    # events = events[events["time"] < pd.to_datetime("2019-07-05 00:00:00")]

    # stations = pd.read_csv(stations_file, sep="\t")
    stations = pd.read_json(stations_file, orient="index")
    stations["station_id"] = stations.index
    stations.reset_index(drop=True, inplace=True)

    config["mindepth"] = config["mindepth"] if "mindepth" in config else 0.0
    config["maxdepth"] = config["maxdepth"] if "maxdepth" in config else 30.0
    config["use_amplitude"] = True

    # ## Eikonal for 1D velocity model
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 5.5, 6.7, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.3

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
    config["min_picks"] = 6
    config["min_picks_ratio"] = 0.5
    config["max_residual_time"] = 1.0
    config["max_residual_amplitude"] = 1.0
    config["min_score"] = 0.5
    config["min_s_picks"] = 0.5
    config["min_p_picks"] = 1.5

    config["bfgs_bounds"] = (
        (config["xlim_km"][0] - 1, config["xlim_km"][1] + 1),  # x
        (config["ylim_km"][0] - 1, config["ylim_km"][1] + 1),  # y
        # (config["zlim_km"][0], config["zlim_km"][1] + 1),  # z
        (0, config["zlim_km"][1] + 1),
        (None, None),  # t
    )

    # %%
    plt.figure()
    plt.scatter(stations["x_km"], stations["y_km"], c=stations["depth_km"], cmap="viridis_r", s=100, marker="^")
    plt.colorbar(label="Depth (km)")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.xlim(config["xlim_km"])
    plt.ylim(config["ylim_km"])
    plt.title("Stations")
    plt.savefig(os.path.join(figure_path, "stations.png"), bbox_inches="tight", dpi=300)

    # %%
    mapping_phase_type_int = {"P": 0, "S": 1}
    config["vel"] = {mapping_phase_type_int[k]: v for k, v in config["vel"].items()}
    picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)
    if "phase_amplitude" in picks.columns:
        picks["phase_amplitude"] = picks["phase_amplitude"].apply(lambda x: np.log10(x) + 2.0)  # convert to log10(cm/s)

    # %%
    # reindex in case the index does not start from 0 or is not continuous
    stations["idx_sta"] = np.arange(len(stations))
    if events is not None:
        # reindex in case the index does not start from 0 or is not continuous
        events["idx_eve"] = np.arange(len(events))

    else:
        picks = picks.merge(stations[["station_id", "x_km", "y_km", "z_km"]], on="station_id")
        events = picks.groupby("event_index").agg({"x_km": "mean", "y_km": "mean", "z_km": "mean", "phase_time": "min"})
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
    MAX_SST_ITER = 10
    # MIN_SST_S = 0.01
    events_init = events.copy()

    for iter in range(MAX_SST_ITER):
        # picks, events = invert_location_iter(picks, stations, config, estimator, events_init=events_init, iter=iter)
        picks, events = invert_location(picks, stations, config, estimator, events_init=events_init, iter=iter)
        # station_term = picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_time": "mean"}).reset_index()
        station_term_time = picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_time": "mean"}).reset_index()
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

        plotting_ransac(stations, figure_path, config, picks, events_init, events, suffix=f"_ransac_sst_{iter}")

        if "event_index" not in events.columns:
            events["event_index"] = events.merge(picks[["idx_eve", "event_index"]], on="idx_eve")["event_index"]
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z_km"]

        picks["adloc_mask"] = picks["mask"]
        picks["adloc_residual_time"] = picks["residual_time"]
        picks["adloc_residual_amplitude"] = picks["residual_amplitude"]
        picks.to_csv(os.path.join(result_path, f"ransac_picks_sst_{iter}.csv"), index=False)
        events.to_csv(os.path.join(result_path, f"ransac_events_sst_{iter}.csv"), index=False)
        stations.to_csv(os.path.join(result_path, f"ransac_stations_sst_{iter}.csv"), index=False)

        if iter == 0:
            MIN_SST_S = (
                np.mean(np.abs(station_term_time["residual_time"])) / 10.0
            )  # break at 10% of the initial station term
            print(f"MIN_SST (s): {MIN_SST_S}")
        if np.mean(np.abs(station_term_time["residual_time"])) < MIN_SST_S:
            print(f"Mean station term: {np.mean(np.abs(station_term_time['residual_time']))}")
            # break
        iter += 1

    # %%
    plotting_ransac(stations, figure_path, config, picks, events_init, events, suffix=f"_ransac")

    if "event_index" not in events.columns:
        events["event_index"] = events.merge(picks[["idx_eve", "event_index"]], on="idx_eve")["event_index"]
    events[["longitude", "latitude"]] = events.apply(
        lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
    )
    events["depth_km"] = events["z_km"]
    events.drop(["idx_eve", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
    events.sort_values(["time"], inplace=True)

    # picks.rename({"mask": "adloc_mask", "residual": "adloc_residual"}, axis=1, inplace=True)
    picks["phase_type"] = picks["phase_type"].map({0: "P", 1: "S"})
    picks.drop(
        ["idx_eve", "idx_sta", "mask", "residual_time", "residual_amplitude"], axis=1, inplace=True, errors="ignore"
    )
    picks.sort_values(["phase_time"], inplace=True)

    stations.drop(["idx_sta", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
    # stations.rename({"station_term": "adloc_station_term_s"}, axis=1, inplace=True)

    picks.to_csv(os.path.join(result_path, "ransac_picks.csv"), index=False)
    events.to_csv(os.path.join(result_path, "ransac_events.csv"), index=False)
    stations.to_csv(os.path.join(result_path, "ransac_stations.csv"), index=False)


# %%
if __name__ == "__main__":

    root_path = "local"
    region = "hinet"

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    run_adloc(root_path=root_path, region=region, config=config)

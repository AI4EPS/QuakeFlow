# %%
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from glob import glob
from threading import Lock, Thread

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from args import parse_args
from obspy import read_inventory
from obspy.clients.fdsn import Client
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from utils.plotting import plotting_ransac

# %%
if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region
    iter = args.iter
    print(f"Merge adloc picks iter={iter}")

    data_path = f"{region}/adloc"
    result_path = f"{region}/adloc"
    figure_path = f"{region}/adloc/figures"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # %%
    # protocol = "gs"
    # token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    # with open(token_json, "r") as fp:
    #     token = json.load(fp)
    # fs = fsspec.filesystem(protocol, token=token)

    # %%
    event_csvs = sorted(glob(f"{root_path}/{data_path}/????/????.???.events_sst_{iter}.csv"))

    # %%
    events = []
    picks = []
    stations = []
    for event_csv in tqdm(event_csvs, desc="Load event csvs"):
        pick_csv = event_csv.replace(f"events_sst_{iter}.csv", f"picks_sst_{iter}.csv")
        station_csv = event_csv.replace(f"events_sst_{iter}.csv", f"stations_sst_{iter}.csv")

        year, jday = event_csv.split("/")[-1].split(".")[:2]
        events_ = pd.read_csv(event_csv, dtype=str)
        picks_ = pd.read_csv(pick_csv, dtype=str)
        stations_ = pd.read_csv(station_csv)
        events_["year"] = year
        events_["jday"] = jday
        picks_["year"] = year
        picks_["jday"] = jday
        stations_["year"] = year
        stations_["jday"] = jday
        events.append(events_)
        picks.append(picks_)
        stations.append(stations_)

    events = pd.concat(events, ignore_index=True)
    picks = pd.concat(picks, ignore_index=True)
    stations = pd.concat(stations, ignore_index=True)

    station_terms = (
        stations.groupby(["station_id"])
        .apply(
            lambda x: pd.Series(
                {
                    "station_term_time_p": (
                        (x.station_term_time_p * x.num_pick_p).sum() / x.num_pick_p.sum()
                        if x.num_pick_p.sum() > 0
                        else 0
                    ),
                    "station_term_time_s": (
                        (x.station_term_time_s * x.num_pick_s).sum() / x.num_pick_s.sum()
                        if x.num_pick_s.sum() > 0
                        else 0
                    ),
                    "station_term_amplitude": (
                        (x.station_term_amplitude * x.num_pick).sum() / x.num_pick.sum() if x.num_pick.sum() > 0 else 0
                    ),
                }
            )
        )
        .reset_index()
    )
    if iter > 0:
        stations_prev = pd.read_csv(f"{root_path}/{result_path}/adloc_stations_sst_{iter-1}.csv")
        stations_prev.set_index("station_id", inplace=True)

        station_terms["station_term_time_p"] += (
            station_terms["station_id"].map(stations_prev["station_term_time_p"]).fillna(0)
        )
        station_terms["station_term_time_s"] += (
            station_terms["station_id"].map(stations_prev["station_term_time_s"]).fillna(0)
        )
        station_terms["station_term_amplitude"] += (
            station_terms["station_id"].map(stations_prev["station_term_amplitude"]).fillna(0)
        )

    stations = stations.groupby(["station_id"]).first().reset_index()
    stations.drop(["station_term_time_p", "station_term_time_s", "station_term_amplitude"], axis=1, inplace=True)
    stations = stations.merge(station_terms, on="station_id")

    events["dummy_id"] = events["year"] + "." + events["jday"] + "." + events["event_index"]
    picks["dummy_id"] = picks["year"] + "." + picks["jday"] + "." + picks["event_index"]

    events["event_index"] = np.arange(len(events))
    picks = picks.drop("event_index", axis=1)
    picks = picks.merge(events[["dummy_id", "event_index"]], on="dummy_id")

    events.drop(["year", "jday", "dummy_id"], axis=1, inplace=True)
    picks.drop(["year", "jday", "dummy_id"], axis=1, inplace=True)
    stations.drop(["year", "jday"], axis=1, inplace=True)

    events.to_csv(f"{root_path}/{result_path}/adloc_events_sst_{iter}.csv", index=False)
    picks.to_csv(f"{root_path}/{result_path}/adloc_picks_sst_{iter}.csv", index=False)
    stations.to_csv(f"{root_path}/{result_path}/adloc_stations_sst_{iter}.csv", index=False)

    ## save current iteration as the latest
    events.to_csv(f"{root_path}/{result_path}/adloc_events.csv", index=False)
    picks.to_csv(f"{root_path}/{result_path}/adloc_picks.csv", index=False)
    stations.to_csv(f"{root_path}/{result_path}/adloc_stations.csv", index=False)

    # %%
    events = pd.read_csv(f"{root_path}/{result_path}/adloc_events_sst_{iter}.csv")
    picks = pd.read_csv(f"{root_path}/{result_path}/adloc_picks_sst_{iter}.csv")
    stations = pd.read_csv(f"{root_path}/{result_path}/adloc_stations_sst_{iter}.csv")

    fig, ax = plt.subplots(3, 3, figsize=(12, 10))
    ax[0, 0].scatter(events["longitude"], events["latitude"], c=events["depth_km"], s=1, cmap="viridis_r")
    ax[0, 0].set_title(f"Events {len(events)}")
    ax[0, 1].scatter(events["longitude"], events["depth_km"], c=events["depth_km"], s=1, cmap="viridis_r")
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_title(f"Events depth")
    ax[0, 2].scatter(events["latitude"], events["depth_km"], c=events["depth_km"], s=1, cmap="viridis_r")
    ax[0, 2].invert_yaxis()
    ax[0, 2].set_title(f"Events latitude")
    ax[1, 0].scatter(
        stations["longitude"], stations["latitude"], c=stations["station_term_time_p"], marker="^", cmap="viridis_r"
    )
    ax[1, 0].set_title(f"Station term time P {stations['station_term_time_p'].mean():.2f} s")
    ax[1, 1].scatter(
        stations["longitude"], stations["latitude"], c=stations["station_term_time_s"], marker="^", cmap="viridis_r"
    )
    ax[1, 1].set_title(f"Station term time S {stations['station_term_time_s'].mean():.2f} s")
    ax[1, 2].scatter(
        stations["longitude"], stations["latitude"], c=stations["station_term_amplitude"], marker="^", cmap="viridis_r"
    )
    ax[1, 2].set_title(f"Station term amplitude {stations['station_term_amplitude'].mean():.2f} m")
    ax[2, 0].hist(events["adloc_residual_time"], bins=30, edgecolor="white")
    ax[2, 0].set_title(f"Event residual time")
    ax[2, 1].hist(events["adloc_residual_amplitude"], bins=30, edgecolor="white")
    ax[2, 1].set_title(f"Event residual amplitude")
    idx = picks["adloc_mask"] == 1
    ax[2, 2].hist(picks.loc[idx, "adloc_residual_time"], bins=30, edgecolor="white")
    ax[2, 2].set_title(f"Pick residual time")
    # ax[2, 2].hist(picks["adloc_residual_amplitude"], bins=30, edgecolor="white")
    # ax[2, 2].set_title(f"Pick residual amplitude")
    plt.tight_layout()
    plt.savefig(f"{root_path}/{figure_path}/adloc_summary_{iter}.png")
    plt.close()

# %%
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread

import fsspec
import numpy as np
import pandas as pd
import pyproj
from obspy import read_inventory
from obspy.clients.fdsn import Client
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from args import parse_args
from glob import glob


def load_data(year, jday, data_path, root_path, bucket, protocol, token):

    fs = fsspec.filesystem(protocol, token=token)
    adloc_events_csv = f"{data_path}/{year:04d}/adloc_events_{jday:03d}.csv"
    adloc_picks_csv = f"{data_path}/{year:04d}/adloc_picks_{jday:03d}.csv"
    if protocol == "file":
        events = pd.read_csv(f"{root_path}/{adloc_events_csv}", parse_dates=["time"])
        picks = pd.read_csv(f"{root_path}/{adloc_picks_csv}", parse_dates=["phase_time"])
    else:
        with fs.open(f"{bucket}/{adloc_events_csv}", "r") as fp:
            events = pd.read_csv(fp, parse_dates=["time"])
        with fs.open(f"{bucket}/{adloc_picks_csv}", "r") as fp:
            picks = pd.read_csv(fp, parse_dates=["phase_time"])

    events["year"] = year
    events["jday"] = jday
    picks["year"] = year
    picks["jday"] = jday

    return events, picks


# %%
if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region

    data_path = f"{region}/gamma"
    result_path = f"{region}/gamma"

    # %%
    # protocol = "gs"
    # token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    # with open(token_json, "r") as fp:
    #     token = json.load(fp)
    # fs = fsspec.filesystem(protocol, token=token)

    # %%
    event_csvs = sorted(glob(f"{root_path}/{data_path}/????/????.???.events.csv"))

    # %%
    events = []
    picks = []
    for event_csv in tqdm(event_csvs, desc="Load event csvs"):
        pick_csv = event_csv.replace("events.csv", "picks.csv")
        year, jday = event_csv.split("/")[-1].split(".")[:2]
        events_ = pd.read_csv(event_csv, dtype=str)
        picks_ = pd.read_csv(pick_csv, dtype=str)
        events_["year"] = year
        events_["jday"] = jday
        picks_["year"] = year
        picks_["jday"] = jday
        events.append(events_)
        picks.append(picks_)

    events = pd.concat(events, ignore_index=True)
    picks = pd.concat(picks, ignore_index=True)

    events["dummy_id"] = events["year"] + "." + events["jday"] + "." + events["event_index"]
    picks["dummy_id"] = picks["year"] + "." + picks["jday"] + "." + picks["event_index"]

    events["event_index"] = np.arange(len(events))
    picks = picks.drop("event_index", axis=1)
    picks = picks.merge(events[["dummy_id", "event_index"]], on="dummy_id")

    events.drop(["year", "jday", "dummy_id"], axis=1, inplace=True)
    picks.drop(["year", "jday", "dummy_id"], axis=1, inplace=True)

    events.to_csv(f"{root_path}/{result_path}/gamma_events.csv", index=False)
    picks.to_csv(f"{root_path}/{result_path}/gamma_picks.csv", index=False)

# %%

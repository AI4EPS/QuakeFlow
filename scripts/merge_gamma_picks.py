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
    picks = picks.merge(events[["dummy_id", "event_index"]], on="dummy_id", how="left")

    events.drop(["year", "jday", "dummy_id"], axis=1, inplace=True)
    picks.drop(["year", "jday", "dummy_id"], axis=1, inplace=True)

    events.to_csv(f"{root_path}/{result_path}/gamma_events.csv", index=False)
    picks.to_csv(f"{root_path}/{result_path}/gamma_picks.csv", index=False)

# %%

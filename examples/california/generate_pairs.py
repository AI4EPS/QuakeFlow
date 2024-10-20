# %%
import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from contextlib import nullcontext

import fsspec
import numpy as np
import pandas as pd
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# %%
def pairing_picks(event_pairs, picks, config):

    picks = picks[["idx_eve", "idx_sta", "phase_type", "phase_score", "phase_time"]].copy()
    merged = pd.merge(
        event_pairs,
        picks,
        left_on="idx_eve1",
        right_on="idx_eve",
    )
    merged = pd.merge(
        merged,
        picks,
        left_on=["idx_eve2", "idx_sta", "phase_type"],
        right_on=["idx_eve", "idx_sta", "phase_type"],
        suffixes=("_1", "_2"),
    )
    merged = merged.rename(columns={"phase_time_1": "phase_time1", "phase_time_2": "phase_time2"})
    merged["phase_score"] = (merged["phase_score_1"] + merged["phase_score_2"]) / 2.0

    merged["travel_time1"] = (merged["phase_time1"] - merged["event_time1"]).dt.total_seconds()
    merged["travel_time2"] = (merged["phase_time2"] - merged["event_time2"]).dt.total_seconds()
    merged["phase_dtime"] = merged["travel_time1"] - merged["travel_time2"]

    # filtering
    # merged = merged.sort_values("phase_score", ascending=False)
    merged = (
        merged.groupby(["idx_eve1", "idx_eve2"], group_keys=False)
        .apply(lambda x: (x.nlargest(config["MAX_OBS"], "phase_score") if len(x) > config["MIN_OBS"] else None))
        .reset_index(drop=True)
    )

    return merged[["idx_eve1", "idx_eve2", "idx_sta", "phase_type", "phase_score", "phase_dtime"]]


# %%
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gamma on NCEDC/SCEDC data")
    parser.add_argument("--num_nodes", type=int, default=366)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--root_path", type=str, default="local")
    parser.add_argument("--region", type=str, default="Cal")
    parser.add_argument("--bucket", type=str, default="quakeflow_catalog")
    return parser.parse_args()


# %%
if __name__ == "__main__":

    # %%
    protocol = "gs"
    token_json = f"application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)

    fs = fsspec.filesystem(protocol, token=token)

    # %%
    args = parse_args()
    region = args.region
    root_path = args.root_path
    bucket = args.bucket
    num_nodes = args.num_nodes
    node_rank = args.node_rank
    year = args.year

    data_path = f"{region}/adloc2"
    result_path = f"{region}/adloc_dd_2022"

    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    station_json = f"{region}/network/stations.json"
    if protocol == "file":
        stations = pd.read_json(f"{root_path}/{station_json}", orient="index")
    else:
        with fs.open(f"{bucket}/{station_json}", "r") as fp:
            stations = pd.read_json(fp, orient="index")
    stations["station_id"] = stations.index

    # %%
    events = []
    picks = []
    jobs = []
    ctx = mp.get_context("spawn")
    ncpu = min(32, mp.cpu_count())
    # years = range(2015, 2024)
    years = [2022]
    # num_days = sum([366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365 for year in years])
    num_days = 365 * len(years)
    pbar = tqdm(total=num_days, desc="Loading data")
    with ctx.Pool(processes=ncpu) as pool:
        for year in years:
            # num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
            num_jday = 365
            for jday in range(1, num_jday + 1):
                job = pool.apply_async(
                    load_data,
                    args=(year, jday, data_path, root_path, bucket, protocol, token),
                    callback=lambda x: pbar.update(),
                )
                jobs.append(job)

        pool.close()
        pool.join()
        for job in jobs:
            events_, picks_ = job.get()
            events.append(events_)
            picks.append(picks_)

    pbar.close()
    events = pd.concat(events, ignore_index=True)
    picks = pd.concat(picks, ignore_index=True)

    events = events.sort_values("time")
    events["dummy_id"] = (
        events["year"].astype(str)
        + "."
        + events["jday"].astype(str).str.zfill(3)
        + "."
        + events["event_index"].astype(str).str.zfill(4)
    )
    picks["dummy_id"] = (
        picks["year"].astype(str)
        + "."
        + picks["jday"].astype(str).str.zfill(3)
        + "."
        + picks["event_index"].astype(str).str.zfill(4)
    )
    events["event_index"] = np.arange(len(events))
    picks = picks.drop("event_index", axis=1)
    picks = picks.merge(events[["dummy_id", "event_index"]], on="dummy_id")

    print(f"Processing {len(events)} events, {len(picks)} picks")

    events.to_csv(f"{root_path}/{result_path}/events.csv", index=False)
    picks.to_csv(f"{root_path}/{result_path}/picks.csv", index=False)

    # # %%
    # events = pd.read_csv(f"{root_path}/{result_path}/events.csv", parse_dates=["time"])
    # picks = pd.read_csv(f"{root_path}/{result_path}/picks.csv", parse_dates=["phase_time"])

    # %%
    MAX_PAIR_DIST = 10  # km
    MAX_NEIGHBORS = 50
    MIN_NEIGHBORS = 8
    MIN_OBS = 8
    MAX_OBS = 100
    config = {}
    config["MAX_PAIR_DIST"] = MAX_PAIR_DIST
    config["MAX_NEIGHBORS"] = MAX_NEIGHBORS
    config["MIN_NEIGHBORS"] = MIN_NEIGHBORS
    config["MIN_OBS"] = MIN_OBS
    config["MAX_OBS"] = MAX_OBS
    mapping_phase_type_int = {"P": 0, "S": 1}

    picks = picks[picks["event_index"] != -1]
    # check phase_type is P/S or 0/1
    # if set(picks["phase_type"].unique()).issubset(set(mapping_phase_type_int.keys())):  # P/S
    picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)

    # %%
    if "idx_eve" in events.columns:
        events = events.drop("idx_eve", axis=1)
    if "idx_sta" in stations.columns:
        stations = stations.drop("idx_sta", axis=1)
    if "idx_eve" in picks.columns:
        picks = picks.drop("idx_eve", axis=1)
    if "idx_sta" in picks.columns:
        picks = picks.drop("idx_sta", axis=1)

    # %%
    # reindex in case the index does not start from 0 or is not continuous
    stations = stations[stations["station_id"].isin(picks["station_id"].unique())]
    events = events[events["event_index"].isin(picks["event_index"].unique())]
    stations["idx_sta"] = np.arange(len(stations))
    events["idx_eve"] = np.arange(len(events))

    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    # %%
    lon0 = stations["longitude"].median()
    lat0 = stations["latitude"].median()
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")

    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["depth_km"] = -stations["elevation_m"] / 1000
    stations["z_km"] = stations["depth_km"]

    events[["x_km", "y_km"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events["z_km"] = events["depth_km"]

    picks = picks.merge(events[["idx_eve", "time"]], on="idx_eve")
    picks["travel_time"] = (picks["phase_time"] - picks["time"]).dt.total_seconds()
    picks.drop("time", axis=1, inplace=True)

    # %%
    # # Option 1: Radius neighbors
    # neigh = NearestNeighbors(radius=MAX_PAIR_DIST, n_jobs=-1)
    # print("Fitting NearestNeighbors")
    # neigh.fit(events[["x_km", "y_km", "z_km"]].values)
    # pairs = set()
    # print("Get neighbors")
    # neigh_ind = neigh.radius_neighbors(sort_results=True)[1]
    # print("Generating pairs")
    # for i, neighs in enumerate(tqdm(neigh_ind, desc="Generating pairs")):
    #     if len(neighs) < MIN_NEIGHBORS:
    #         continue
    #     for j in neighs[:MAX_NEIGHBORS]:
    #         if i < j:
    #             pairs.add((i, j))
    #         else:
    #             pairs.add((j, i))

    # Option 2: K-nearest neighbors
    neigh = NearestNeighbors(n_neighbors=MAX_NEIGHBORS, n_jobs=-1)
    print("Fitting NearestNeighbors...")
    neigh.fit(events[["x_km", "y_km", "z_km"]].values)
    pairs = set()
    print("Get neighbors...")
    neigh_dist, neigh_ind = neigh.kneighbors()
    print("Generating pairs...")
    for i, (dists, inds) in enumerate(tqdm(zip(neigh_dist, neigh_ind), desc="Generating pairs", total=len(neigh_ind))):
        inds = inds[dists <= MAX_PAIR_DIST]
        if len(inds) < MIN_NEIGHBORS:
            continue
        for j in inds:
            if i < j:
                pairs.add((i, j))
            else:
                pairs.add((j, i))

    pairs = list(pairs)
    event_pairs = pd.DataFrame(list(pairs), columns=["idx_eve1", "idx_eve2"])
    print(f"Number of events: {len(events)}")
    print(f"Number of event pairs: {len(event_pairs)}")
    event_pairs["event_time1"] = events["time"].iloc[event_pairs["idx_eve1"]].values
    event_pairs["event_time2"] = events["time"].iloc[event_pairs["idx_eve2"]].values

    # %%
    chunk_size = 100_000
    num_chunk = len(event_pairs) // chunk_size
    pbar = tqdm(total=num_chunk, desc="Pairing picks")

    results = []
    jobs = []
    ctx = mp.get_context("spawn")
    ncpu = min(num_chunk, min(32, mp.cpu_count()))
    picks["idx_eve"] = picks["idx_eve"].astype("category")
    with ctx.Pool(processes=ncpu) as pool:
        for i in np.array_split(np.arange(len(event_pairs)), num_chunk):
            event_pairs_ = event_pairs.iloc[i]
            idx = np.unique(event_pairs_[["idx_eve1", "idx_eve2"]].values.flatten())
            picks_ = picks[picks["idx_eve"].isin(idx)]
            job = pool.apply_async(pairing_picks, args=(event_pairs_, picks_, config), callback=lambda x: pbar.update())
            jobs.append(job)
        pool.close()
        pool.join()
        for job in jobs:
            results.append(job.get())

    event_pairs = pd.concat(results, ignore_index=True)
    event_pairs = event_pairs.drop_duplicates()

    print(f"Number of pick pairs: {len(event_pairs)}")

    dtypes = np.dtype(
        [
            ("idx_eve1", np.int32),
            ("idx_eve2", np.int32),
            ("idx_sta", np.int32),
            ("phase_type", np.int32),
            ("phase_score", np.float32),
            ("phase_dtime", np.float32),
        ]
    )
    pairs_array = np.memmap(
        f"{root_path}/{result_path}/pair_dt.dat",
        mode="w+",
        shape=(len(event_pairs),),
        dtype=dtypes,
    )
    pairs_array["idx_eve1"] = event_pairs["idx_eve1"].values
    pairs_array["idx_eve2"] = event_pairs["idx_eve2"].values
    pairs_array["idx_sta"] = event_pairs["idx_sta"].values
    pairs_array["phase_type"] = event_pairs["phase_type"].values
    pairs_array["phase_score"] = event_pairs["phase_score"].values
    pairs_array["phase_dtime"] = event_pairs["phase_dtime"].values
    with open(f"{root_path}/{result_path}/pair_dtypes.pkl", "wb") as f:
        pickle.dump(dtypes, f)

    events.to_csv(f"{root_path}/{result_path}/pair_events.csv", index=False)
    stations.to_csv(f"{root_path}/{result_path}/pair_stations.csv", index=False)
    picks.to_csv(f"{root_path}/{result_path}/pair_picks.csv", index=False)

# %%

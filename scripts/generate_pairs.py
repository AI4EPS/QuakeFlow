# %%
import argparse
import json
import multiprocessing as mp
import os
import pickle
from contextlib import nullcontext

import numpy as np
import pandas as pd
from args import parse_args
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
if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region

    data_path = f"{root_path}/{region}/adloc"
    result_path = f"{root_path}/{region}/adloc_dd"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # %%
    pick_file = os.path.join(data_path, "ransac_picks.csv")
    event_file = os.path.join(data_path, "ransac_events.csv")
    station_file = os.path.join(data_path, "ransac_stations.csv")

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

    # %%
    stations = pd.read_csv(station_file)
    picks = pd.read_csv(pick_file, parse_dates=["phase_time"])
    events = pd.read_csv(event_file, parse_dates=["time"])

    picks = picks[picks["event_index"] != -1]
    # check phase_type is P/S or 0/1
    if set(picks["phase_type"].unique()).issubset(set(mapping_phase_type_int.keys())):  # P/S
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
    picks_by_event = picks.groupby("idx_eve")

    # Option 1:
    neigh = NearestNeighbors(radius=MAX_PAIR_DIST, n_jobs=-1)
    neigh.fit(events[["x_km", "y_km", "z_km"]].values)
    pairs = set()
    neigh_ind = neigh.radius_neighbors(sort_results=True)[1]
    for i, neighs in enumerate(tqdm(neigh_ind, desc="Generating pairs")):
        if len(neighs) < MIN_NEIGHBORS:
            continue
        for j in neighs[:MAX_NEIGHBORS]:
            if i < j:
                pairs.add((i, j))
    pairs = list(pairs)
    event_pairs = pd.DataFrame(list(pairs), columns=["idx_eve1", "idx_eve2"])
    print(f"Number of events: {len(events)}")
    print(f"Number of event pairs: {len(event_pairs)}")
    event_pairs["event_time1"] = events["time"].iloc[event_pairs["idx_eve1"]].values
    event_pairs["event_time2"] = events["time"].iloc[event_pairs["idx_eve2"]].values

    # Option 2:
    # neigh = NearestNeighbors(radius=MAX_PAIR_DIST, n_jobs=-1)
    # neigh.fit(events[["x_km", "y_km", "z_km"]].values)
    # pairs = set()
    # neigh_ind = neigh.radius_neighbors()[1]
    # for i, neighs in enumerate(tqdm(neigh_ind, desc="Generating pairs")):
    #     if len(neighs) < MIN_NEIGHBORS:
    #         continue
    #     neighs = neighs[np.argsort(events.loc[neighs, "num_picks"])]  ## TODO: check if useful
    #     for j in neighs[:MAX_NEIGHBORS]:
    #         if i > j:
    #             pairs.add((j, i))
    #         else:
    #             pairs.add((i, j))
    # pairs = list(pairs)

    # %%
    chunk_size = 10_000
    num_chunk = len(event_pairs) // chunk_size
    pbar = tqdm(total=num_chunk, desc="Pairing picks")
    results = []
    jobs = []
    ctx = mp.get_context("spawn")
    ncpu = min(num_chunk, min(32, mp.cpu_count()))
    picks = picks.set_index("idx_eve")
    with ctx.Pool(processes=ncpu) as pool:
        for i in np.array_split(np.arange(len(event_pairs)), num_chunk):
            event_pairs_ = event_pairs.iloc[i]
            idx = np.unique(event_pairs_[["idx_eve1", "idx_eve2"]].values.flatten())
            picks_ = picks.loc[idx].reset_index()
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
            ("event_index1", np.int32),
            ("event_index2", np.int32),
            ("station_index", np.int32),
            ("phase_type", np.int32),
            ("phase_score", np.float32),
            ("phase_dtime", np.float32),
        ]
    )
    pairs_array = np.memmap(
        os.path.join(result_path, "pair_dt.dat"),
        mode="w+",
        shape=(len(event_pairs),),
        dtype=dtypes,
    )
    pairs_array["event_index1"] = event_pairs["idx_eve1"].values
    pairs_array["event_index2"] = event_pairs["idx_eve2"].values
    pairs_array["station_index"] = event_pairs["idx_sta"].values
    pairs_array["phase_type"] = event_pairs["phase_type"].values
    pairs_array["phase_score"] = event_pairs["phase_score"].values
    pairs_array["phase_dtime"] = event_pairs["phase_dtime"].values
    with open(os.path.join(result_path, "pair_dtypes.pkl"), "wb") as f:
        pickle.dump(dtypes, f)

    events.to_csv(os.path.join(result_path, "pair_events.csv"), index=False)
    stations.to_csv(os.path.join(result_path, "pair_stations.csv"), index=False)
    picks.to_csv(os.path.join(result_path, "pair_picks.csv"), index=False)

# %%

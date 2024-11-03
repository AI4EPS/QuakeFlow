# %%
import json
import os
from glob import glob
from typing import Dict

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from args import parse_args
from pyproj import Proj
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from tqdm import tqdm


def associate(
    picks: pd.DataFrame,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    config: Dict,
):

    VPVS_RATIO = config["VPVS_RATIO"]
    VP = config["VP"]

    proj = Proj(proj="merc", datum="WGS84", units="km")
    stations[["x_km", "y_km"]] = stations.apply(lambda x: pd.Series(proj(x.longitude, x.latitude)), axis=1)

    # dist_matrix = squareform(pdist(stations[["x_km", "y_km"]].values))
    # mst = minimum_spanning_tree(dist_matrix)
    # dx = np.median(mst.data[mst.data > 0])
    # print(f"dx: {dx:.3f}")
    # eps_t = dx / VP * 2.0
    # eps_t = 6.0
    # eps_xy = eps_t * VP * 2 / (1.0 + VPVS_RATIO)
    # print(f"eps_t: {eps_t:.3f}, eps_xy: {eps_xy:.3f}")
    eps_xy = 30.0
    print(f"eps_xy: {eps_xy:.3f}")

    # %%
    t0 = min(events["event_time"].min(), picks["phase_time"].min())
    events["timestamp"] = events["event_time"].apply(lambda x: (x - t0).total_seconds())
    events["timestamp_center"] = events["center_time"].apply(lambda x: (x - t0).total_seconds())
    picks["timestamp"] = picks["phase_time"].apply(lambda x: (x - t0).total_seconds())

    # %%
    events = events.merge(stations[["station_id", "x_km", "y_km"]], on="station_id", how="left")

    scaling = np.array([1.0, 1.0 / eps_xy, 1.0 / eps_xy])
    clustering = DBSCAN(eps=2.0, min_samples=4).fit(events[["timestamp", "x_km", "y_km"]] * scaling)
    # clustering = DBSCAN(eps=2.0, min_samples=4).fit(events[["timestamp"]])
    # clustering = DBSCAN(eps=3.0, min_samples=3).fit(events[["timestamp"]])
    # clustering = DBSCAN(eps=1.0, min_samples=3).fit(events[["timestamp"]])
    events["event_index"] = clustering.labels_
    print(f"Number of associated events: {len(events['event_index'].unique())}")

    # %% link picks to events
    picks["event_index"] = -1
    picks.set_index("station_id", inplace=True)

    for group_id, event in tqdm(events.groupby("station_id"), desc="Linking picks to events"):
        # travel time tt = (tp + ts) / 2 = (ps_ratio + 1)/2 * tp,
        # (ts - tp) = (ps_ratio - 1) tp = tt * (ps_ratio + 1) * 2 * (ps_ratio - 1)
        ps_delta = event["travel_time_s"] / (VPVS_RATIO + 1) * 2 * (VPVS_RATIO - 1)
        t1 = event["timestamp_center"] - ps_delta * 1.2
        t2 = event["timestamp_center"] + ps_delta * 1.2
        index = event["event_index"]

        mask = (picks.loc[group_id, "timestamp"].values[None, :] >= t1.values[:, None]) & (
            picks.loc[group_id, "timestamp"].values[None, :] <= t2.values[:, None]
        )
        picks.loc[group_id, "event_index"] = np.where(
            mask.any(axis=0), index.values[mask.argmax(axis=0)], picks.loc[group_id, "event_index"]
        )

    picks.reset_index(inplace=True)

    picks.drop(columns=["timestamp"], inplace=True)
    events.drop(columns=["timestamp", "timestamp_center"], inplace=True)

    events = events.merge(stations[["station_id", "latitude", "longitude"]], on="station_id", how="left")
    events = (
        events.groupby("event_index")
        .agg(
            {
                "event_time": "median",
                "event_score": "sum",
                "latitude": "median",
                "longitude": "median",
            }
        )
        .reset_index()
    )
    events.rename(columns={"event_time": "time"}, inplace=True)
    # drop event index -1
    events = events[events["event_index"] != -1]

    return events, picks


def run_association(
    root_path: str,
    region: str,
    jdays: list,
    config: Dict,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    for jday in jdays:

        # %%
        print(f"Processing {jday}")

        year = int(jday.split(".")[0])
        jday = int(jday.split(".")[1])

        # %%
        data_path = f"{region}/phasenet_plus/{year:04d}"
        result_path = f"{region}/phasenet_plus/{year:04d}"
        if not os.path.exists(f"{root_path}/{result_path}"):
            os.makedirs(f"{root_path}/{result_path}")

        # %%
        stations = pd.read_json(f"{root_path}/{region}/obspy/stations.json", orient="index")
        stations["station_id"] = stations.index
        events = pd.read_csv(
            f"{root_path}/{data_path}/{year:04d}.{jday:03d}.events.csv", parse_dates=["center_time", "event_time"]
        )
        picks = pd.read_csv(f"{root_path}/{data_path}/{year:04d}.{jday:03d}.picks.csv", parse_dates=["phase_time"])

        events, picks = associate(picks, events, stations, config)

        # %%
        # plt.figure(figsize=(10, 5))
        # plt.scatter(
        #     picks["phase_time"],
        #     picks["station_id"].map(mapping),
        #     # c=[f"C{x}" if x != -1 else "k" for x in picks["phase_type"]],
        #     c=["b" if x == "P" else "r" if x == "S" else "k" for x in picks["phase_type"]],
        #     marker=".",
        #     s=3,
        # )
        # plt.scatter([], [], c="b", label="P")
        # plt.scatter([], [], c="r", label="S")
        # plt.legend(loc="upper right")
        # plt.ylabel("Station #")
        # plt.xlim(pd.Timestamp("2019-07-04T17:40:00"), pd.Timestamp("2019-07-04T17:45:00"))
        # # plt.xlim(pd.Timestamp("2019-07-04T18:01:50"), pd.Timestamp("2019-07-04T18:05:00"))
        # plt.savefig("demo_phasenet_plus_picks.png")

        # plt.figure(figsize=(10, 5))
        # plt.scatter(
        #     events["event_time"],
        #     events["station_id"].map(mapping),
        #     # c=[f"C{x}" if x != -1 else "k" for x in events["event_index"]],
        #     c=["g" for x in events["event_index"]],
        #     marker="x",
        #     s=10,
        # )
        # plt.scatter(
        #     picks["phase_time"],
        #     picks["station_id"].map(mapping),
        #     # c=[f"C{x}" if x != -1 else "k" for x in picks["event_index"]],
        #     c=["b" if x == "P" else "r" if x == "S" else "k" for x in picks["phase_type"]],
        #     marker=".",
        #     s=3,
        #     alpha=0.2,
        # )
        # plt.scatter([], [], c="b", label="P")
        # plt.scatter([], [], c="r", label="S")
        # plt.scatter([], [], c="g", marker="x", label="Event OT")
        # plt.legend(loc="upper right")
        # plt.ylabel("Station #")
        # plt.xlim(pd.Timestamp("2019-07-04T17:40:00"), pd.Timestamp("2019-07-04T17:45:00"))
        # plt.savefig("demo_phasenet_plus_events.png")

        # plt.figure(figsize=(10, 5))
        # plt.scatter(
        #     events["event_time"],
        #     events["station_id"].map(mapping),
        #     c=[f"C{x}" if x != -1 else "k" for x in events["event_index"]],
        #     marker="x",
        #     s=10,
        # )
        # plt.scatter(
        #     picks["phase_time"],
        #     picks["station_id"].map(mapping),
        #     # c=[f"C{x}" if x != -1 else "k" for x in picks["event_index"]],
        #     c=["b" if x == "P" else "r" if x == "S" else "k" for x in picks["phase_type"]],
        #     marker=".",
        #     s=3,
        #     alpha=0.2,
        # )
        # plt.scatter([], [], c="b", label="P")
        # plt.scatter([], [], c="r", label="S")
        # plt.scatter([], [], c="k", marker="x", label="Event OT")
        # plt.legend(loc="upper right")
        # plt.ylabel("Station #")
        # plt.xlim(pd.Timestamp("2019-07-04T17:40:00"), pd.Timestamp("2019-07-04T17:45:00"))
        # # plt.xlim(pd.Timestamp("2019-07-04T18:01:50"), pd.Timestamp("2019-07-04T18:05:00"))
        # plt.savefig("demo_phasenet_plus.png")

        # %%
        # events.drop(columns=["timestamp", "timestamp_center"], inplace=True, errors="ignore")
        # picks.drop(columns=["timestamp"], inplace=True, errors="ignore")
        events.to_csv(f"{root_path}/{result_path}/{year:04d}.{jday:03d}.events_associated.csv", index=False)
        picks.to_csv(f"{root_path}/{result_path}/{year:04d}.{jday:03d}.picks_associated.csv", index=False)

        # if protocol != "file":
        #     fs.put(
        #         f"{root_path}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv",
        #         f"{bucket}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv",
        #     )
        #     fs.put(
        #         f"{root_path}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv",
        #         f"{bucket}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv",
        #     )

        # # copy to results/phase_association
        # if not os.path.exists(f"{root_path}/{region}/results/phase_association"):
        #     os.makedirs(f"{root_path}/{region}/results/phase_association")
        # os.system(
        #     f"cp {root_path}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv {root_path}/{region}/results/phase_association/events_{rank:03d}.csv"
        # )
        # os.system(
        #     f"cp {root_path}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv {root_path}/{region}/results/phase_association/picks_{rank:03d}.csv"
        # )
        # if protocol != "file":
        #     fs.put(
        #         f"{root_path}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv",
        #         f"{bucket}/{region}/results/phase_association/events_{rank:03d}.csv",
        #     )
        #     fs.put(
        #         f"{root_path}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv",
        #         f"{bucket}/{region}/results/phase_association/picks_{rank:03d}.csv",
        #     )


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region
    num_nodes = args.num_nodes
    node_rank = args.node_rank

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    config.update({"VPVS_RATIO": 1.73, "VP": 6.0})

    jdays = sorted(glob(f"{root_path}/{region}/phasenet_plus/????/????.???.events.csv"))
    jdays = [jday.split("/")[-1].replace(".csv", "") for jday in jdays]
    print(f"Number of event files: {len(jdays)}")

    jdays = [jdays[i::num_nodes] for i in range(num_nodes)][node_rank]

    run_association(root_path=root_path, region=region, jdays=jdays, config=config)

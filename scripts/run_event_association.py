# %%
import json
import os
from glob import glob
from typing import Dict

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from args import parse_args
from pyproj import Proj
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from tqdm import tqdm


def plotting_debug(xt, hist, topk_index, topk_score, picks, events, stations, config):

    # timestamp0 = config["timestamp0"]
    # events_compare = pd.read_csv("local/Ridgecrest_debug5/adloc_gamma/ransac_events.csv")
    # picks_compare = pd.read_csv("local/Ridgecrest_debug5/adloc_gamma/ransac_picks.csv")
    # # events_compare = pd.read_csv("local/Ridgecrest_debug5/adloc_plus2/ransac_events_sst_0.csv")
    # # picks_compare = pd.read_csv("local/Ridgecrest_debug5/adloc_plus2/ransac_picks_sst_0.csv")
    # events_compare["time"] = pd.to_datetime(events_compare["time"])
    # events_compare["timestamp"] = events_compare["time"].apply(lambda x: (x - timestamp0).total_seconds())
    # picks_compare["phase_time"] = pd.to_datetime(picks_compare["phase_time"])
    # picks_compare["timestamp"] = picks_compare["phase_time"].apply(lambda x: (x - timestamp0).total_seconds())

    DT = config["DT"]
    MIN_STATION = config["MIN_STATION"]

    # map station_id to int
    stations["xy"] = stations["longitude"] - stations["latitude"]
    stations.sort_values(by="xy", inplace=True)
    mapping_id = {v: i for i, v in enumerate(stations["station_id"])}
    mapping_color = {v: f"C{i}" if v != -1 else "k" for i, v in enumerate(events["event_index"].unique())}

    NX = 100
    for i in tqdm(range(0, len(hist), NX)):
        bins = np.arange(i, i + NX, DT)

        fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # plot hist
        idx = (xt > i) & (xt < i + NX)
        ax[0].bar(xt[idx], hist[idx], width=DT)

        ylim = ax[0].get_ylim()
        idx = (xt[topk_index] > i) & (xt[topk_index] < i + NX)
        ax[0].vlines(xt[topk_index][idx], ylim[0], ylim[1], color="k", linewidth=1)

        # idx = (events_compare["timestamp"] > i) & (events_compare["timestamp"] < i + NX)
        # ax[0].vlines(events_compare["timestamp"][idx], ylim[0], ylim[1], color="r", linewidth=1, linestyle="--")

        # plot picks-events match
        idx = (events["timestamp"] > i) & (events["timestamp"] < i + NX)
        ax[1].scatter(
            events["timestamp"][idx],
            events["station_id"][idx].map(mapping_id),
            c=events["event_index"][idx].map(mapping_color),
            marker=".",
            s=30,
        )

        idx = (picks["timestamp"] > i) & (picks["timestamp"] < i + NX)
        ax[1].scatter(
            picks["timestamp"][idx],
            picks["station_id"][idx].map(mapping_id),
            c=picks["event_index"][idx].map(mapping_color),
            marker="x",
            linewidth=0.5,
            s=10,
        )

        # idx = (picks_compare["timestamp"] > i) & (picks_compare["timestamp"] < i + NX)
        # ax[1].scatter(
        #     picks_compare["timestamp"][idx],
        #     picks_compare["station_id"][idx].map(mapping_id),
        #     facecolors="none",
        #     edgecolors="r",
        #     linewidths=0.1,
        #     s=30,
        # )

        if not os.path.exists(f"figures"):
            os.makedirs(f"figures")
        plt.savefig(f"figures/debug_{i:04d}.png", dpi=300, bbox_inches="tight")


def associate(
    picks: pd.DataFrame,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    config: Dict,
):

    VPVS_RATIO = config["VPVS_RATIO"]
    VP = config["VP"]
    DT = 2.0  # seconds
    MIN_STATION = 3

    # %%
    timestamp0 = min(events["event_time"].min(), picks["phase_time"].min())

    events["timestamp"] = events["event_time"].apply(lambda x: (x - timestamp0).total_seconds())
    events["timestamp_center"] = events["center_time"].apply(lambda x: (x - timestamp0).total_seconds())
    picks["timestamp"] = picks["phase_time"].apply(lambda x: (x - timestamp0).total_seconds())

    t0 = min(events["timestamp"].min(), picks["timestamp"].min())
    t1 = max(events["timestamp"].max(), picks["timestamp"].max())

    # %% Using DBSCAN to cluster events
    # proj = Proj(proj="merc", datum="WGS84", units="km")
    # stations[["x_km", "y_km"]] = stations.apply(lambda x: pd.Series(proj(x.longitude, x.latitude)), axis=1)
    # events = events.merge(stations[["station_id", "x_km", "y_km"]], on="station_id", how="left")
    # scaling = np.array([1.0, 1.0 / eps_xy, 1.0 / eps_xy])
    # clustering = DBSCAN(eps=2.0, min_samples=4).fit(events[["timestamp", "x_km", "y_km"]] * scaling)
    # # clustering = DBSCAN(eps=2.0, min_samples=4).fit(events[["timestamp"]])
    # events["event_index"] = clustering.labels_

    ## Using histogram to cluster events
    events["event_index"] = -1
    t = np.arange(t0, t1, DT)
    hist, edge = np.histogram(events["timestamp"], bins=t, weights=events["event_score"])
    xt = (edge[:-1] + edge[1:]) / 2  # center of the bin
    # hist_numpy = hist.copy()

    hist = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0)
    hist_pool = F.max_pool1d(hist, kernel_size=3, padding=1, stride=1)
    mask = hist_pool == hist
    hist = hist * mask
    hist = hist.squeeze(0).squeeze(0)
    K = int((t[-1] - t[0]) / 5)  # assume max 1 event per 10 seconds on average
    topk_score, topk_index = torch.topk(hist, k=K)
    topk_index = topk_index[topk_score >= MIN_STATION]  # min 3 stations
    topk_score = topk_score[topk_score >= MIN_STATION]
    topk_index = topk_index.numpy()
    topk_score = topk_score.numpy()
    num_events = len(topk_index)
    t00 = xt[topk_index - 1]
    t11 = xt[topk_index + 1]
    timestamp = events["timestamp"].values
    for i in tqdm(range(num_events), desc="Assigning event index"):
        mask = (timestamp >= t00[i]) & (timestamp <= t11[i])
        events.loc[mask, "event_index"] = i
    events["num_picks"] = events.groupby("event_index").size()
    ## logPGV = -4.75 + 1.68 * logR + 0.93M => M = (logPGV - 4.175 - 1.68 * logR) / 0.93
    events["magnitude"] = (
        np.log10(events["event_amplitude"])
        + 4.175
        + 1.68 * np.log10(events["travel_time"] * VP * (VPVS_RATIO + 1.0) / 2.0)
    ) / 0.93

    # # refine event index using DBSCAN
    # events["group_index"] = -1
    # for group_id, event in tqdm(events.groupby("event_index"), desc="DBSCAN clustering"):
    #     if len(event) < MIN_STATION:
    #         events.loc[event.index, "event_index"] = -1
    #     clustering = DBSCAN(eps=20, min_samples=MIN_STATION).fit(event[["x_km", "y_km"]])
    #     events.loc[event.index, "group_index"] = clustering.labels_
    # events["dummy_index"] = events["event_index"].astype(str) + "." + events["group_index"].astype(str)
    # mapping = {v: i for i, v in enumerate(events["dummy_index"].unique())}
    # events["dummy_index"] = events["dummy_index"].map(mapping)
    # events.loc[(events["event_index"] == -1) | (events["group_index"] == -1), "dummy_index"] = -1
    # events["event_index"] = events["dummy_index"]
    # events.drop(columns=["dummy_index"], inplace=True)

    # %% link picks to events
    picks["event_index"] = -1
    picks.set_index("station_id", inplace=True)

    for group_id, event in tqdm(events.groupby("station_id"), desc="Linking picks to events"):
        # travel time tt = (tp + ts) / 2 = (1 + ps_ratio)/2 * tp => tp = tt * 2 / (1 + ps_ratio)
        # (ts - tp) = (ps_ratio - 1) tp = tt * 2 * (ps_ratio - 1) / (ps_ratio + 1)

        event = event.sort_values(by="num_picks", ascending=True)
        ps_delta = event["travel_time"].values * 2 * (VPVS_RATIO - 1) / (VPVS_RATIO + 1)
        # t1 = event["timestamp_center"].values - ps_delta * 1.1 - 1.0
        # t2 = event["timestamp_center"].values + ps_delta * 1.1 + 1.0
        t1 = event["timestamp_center"].values - (ps_delta * 0.6 + 1.0)
        t2 = event["timestamp_center"].values + (ps_delta * 0.6 + 1.0)

        picks_ = picks.loc[group_id, "timestamp"].values  # (Npk, )
        mask = (picks_[None, :] >= t1[:, None]) & (picks_[None, :] <= t2[:, None])  # (Nev, Npk)
        # picks.loc[group_id, "event_index"] = np.where(
        #     mask.any(axis=0), index.values[mask.argmax(axis=0)], picks.loc[group_id, "event_index"]
        # )
        mask_true = mask.any(axis=0)
        mask_idx = mask.argmax(axis=0)
        picks.loc[group_id, "event_index"] = np.where(mask_true, event["event_index"].values[mask_idx], -1)
        picks.loc[group_id, "sp_ratio"] = np.where(mask_true, event["sp_ratio"].values[mask_idx], np.nan)
        picks.loc[group_id, "event_amplitude"] = np.where(mask_true, event["event_amplitude"].values[mask_idx], np.nan)

    picks.reset_index(inplace=True)

    # plotting_debug(
    #     xt,
    #     hist_numpy,
    #     topk_index,
    #     topk_score,
    #     picks,
    #     events,
    #     stations,
    #     {"DT": DT, "MIN_STATION": MIN_STATION, "timestamp0": timestamp0},
    # )

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
                "magnitude": "median",
            }
        )
        .reset_index()
    )
    events.rename(columns={"event_time": "time"}, inplace=True)
    # drop event index -1
    events = events[events["event_index"] != -1]

    print(f"Number of associated events: {len(events['event_index'].unique()):,}")
    print(f"Number of associated picks: {len(picks[picks['event_index'] != -1]):,} / {len(picks):,}")

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

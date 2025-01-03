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


def associate(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    template_events: pd.DataFrame,
    config: Dict,
):

    DT = 2.0  # seconds
    MIN_STATION = 3

    # %%
    timestamp0 = picks["origin_time"].min()

    picks["timestamp"] = picks["origin_time"].apply(lambda x: (x - timestamp0).total_seconds())
    picks["phase_score"] = picks["cc"]

    t0 = picks["timestamp"].min()
    t1 = picks["timestamp"].max()

    # %% Using DBSCAN to cluster events
    # proj = Proj(proj="merc", datum="WGS84", units="km")
    # stations[["x_km", "y_km"]] = stations.apply(lambda x: pd.Series(proj(x.longitude, x.latitude)), axis=1)
    # events = events.merge(stations[["station_id", "x_km", "y_km"]], on="station_id", how="left")
    # scaling = np.array([1.0, 1.0 / eps_xy, 1.0 / eps_xy])
    # clustering = DBSCAN(eps=2.0, min_samples=4).fit(events[["timestamp", "x_km", "y_km"]] * scaling)
    # # clustering = DBSCAN(eps=2.0, min_samples=4).fit(events[["timestamp"]])
    # events["event_index"] = clustering.labels_

    ## Using histogram to cluster events
    t = np.arange(t0, t1, DT)
    # hist, edge = np.histogram(picks["timestamp"], bins=t, weights=picks["phase_score"])
    hist, edge = np.histogram(picks["timestamp"], bins=t)
    xt = (edge[:-1] + edge[1:]) / 2  # center of the bin

    # hist_back = hist.copy()  # for debug
    # xt_back = xt.copy()  # for debug

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
    t00 = xt[np.maximum(topk_index - 1, 0)]
    t11 = xt[np.minimum(topk_index + 1, len(xt) - 1)]
    timestamp = picks["timestamp"].values
    picks["event_index"] = -1
    for i in tqdm(range(num_events), desc="Assigning event index"):
        mask = (timestamp >= t00[i]) & (timestamp <= t11[i])
        picks.loc[mask, "event_index"] = i

    picks.drop(columns=["timestamp"], inplace=True)

    events = picks.merge(template_events, on="idx_eve")
    events = (
        events.groupby("event_index")
        .agg(
            {
                "origin_time": "median",
                "phase_score": "sum",
                "latitude": "median",
                "longitude": "median",
                "depth_km": "median",
                # "magnitude": "median",
            }
        )
        .reset_index()
    )
    events.rename(columns={"origin_time": "time", "phase_score": "qtm_score"}, inplace=True)
    events = events[events["event_index"] != -1]  # drop event index -1

    print(f"Number of associated events: {len(events['event_index'].unique()):,}")
    print(f"Number of associated picks: {len(picks[picks['event_index'] != -1]):,} / {len(picks):,}")

    # # %% Debug plotting
    # plt.figure(figsize=(10, 10))

    # ts = pd.Timestamp("2019-07-04 17:33:49")
    # te = ts + pd.Timedelta(minutes=8)

    # plt.subplot(2, 1, 1)

    # picks = picks[(picks["phase_time"] >= ts) & (picks["phase_time"] <= te)]
    # events = events[(events["origin_time"] >= ts) & (events["origin_time"] <= te)]

    # picks = picks.merge(stations[["idx_sta", "latitude", "longitude"]], on="idx_sta")
    # color = lambda x: f"C{int(x)}" if x != -1 else "k"
    # plt.scatter(picks["phase_time"], picks["latitude"], c=picks["event_index"].map(color), s=0.5)
    # plt.scatter(picks["origin_time"], picks["latitude"], c=picks["event_index"].map(color), s=0.5, marker="x")
    # for i, pick in picks.iterrows():
    #     plt.plot(
    #         [pick["origin_time"], pick["phase_time"]],
    #         [pick["latitude"], pick["latitude"]],
    #         color=color(pick["event_index"]),
    #         linewidth=0.1,
    #     )
    # ylim = plt.ylim()
    # for i, event in events.iterrows():
    #     plt.vlines(event["origin_time"], ylim[0], ylim[1], color=color(event["event_index"]), linewidth=0.5)

    # plt.subplot(2, 1, 2)
    # ts_sec = (ts - timestamp0).total_seconds()
    # te_sec = (te - timestamp0).total_seconds()
    # plt.bar(
    #     xt_back[(xt_back >= ts_sec) & (xt_back <= te_sec)],
    #     hist_back[(xt_back >= ts_sec) & (xt_back <= te_sec)],
    #     width=DT,
    # )
    # plt.grid(which="both")
    # plt.savefig("debug_picks.png")

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
        data_path = f"{region}/qtm/ccpairs"
        result_path = f"{region}/qtm/"
        if not os.path.exists(f"{root_path}/{result_path}"):
            os.makedirs(f"{root_path}/{result_path}")

        # %%
        stations = pd.read_csv(f"{root_path}/{region}/cctorch/cctorch_stations.csv")
        template_events = pd.read_csv(f"{root_path}/{region}/cctorch/cctorch_events.csv")
        template_picks = pd.read_csv(f"{root_path}/{region}/cctorch/cctorch_picks.csv")
        template_events = template_events[["idx_eve", "latitude", "longitude", "depth_km", "magnitude"]]
        print(f"Number of template events: {len(template_events)}")
        print(f"Number of template picks: {len(template_picks)}")

        # %%
        picks = glob(f"{root_path}/{data_path}/TM_???_???.csv")
        picks = [pd.read_csv(pick, parse_dates=["phase_time", "origin_time"]) for pick in picks]
        picks = pd.concat(picks)

        events, picks = associate(picks, stations, template_events, config)

        # %%
        # events.rename(columns={"origin_time": "time"}, inplace=True)
        # events.drop(columns=["timestamp", "timestamp_center"], inplace=True, errors="ignore")
        picks.drop(columns=["cc"], inplace=True, errors="ignore")
        events = events.sort_values(by="time")
        picks = picks.sort_values(by="phase_time")
        events.to_csv(f"{root_path}/{result_path}/qtm_events.csv", index=False)
        picks.to_csv(f"{root_path}/{result_path}/qtm_picks.csv", index=False)


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region
    num_nodes = args.num_nodes
    node_rank = args.node_rank

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    jdays = ["2019.001"]
    print(f"Number of event files: {len(jdays)}")

    jdays = [jdays[i::num_nodes] for i in range(num_nodes)][node_rank]

    run_association(root_path=root_path, region=region, jdays=jdays, config=config)

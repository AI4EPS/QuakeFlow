# %%
import json
import multiprocessing as mp
import os
import sys
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "8"


# %%
def extract_picks(pair, detects, config, mseeds, picks):
    h5, id1 = pair

    time_before = config["time_before"]
    min_cc_score = config["min_cc_score"]
    dt = 1.0 / config["sampling_rate"]

    with h5py.File(h5, "r") as fp:
        gp = fp[id1]
        id1 = int(id1)

        for id2 in gp:
            ds = gp[id2]
            id2 = int(id2)

            cc_score = ds["cc_score"][:]  # [nch, nsta, nev]
            cc_index = ds["cc_index"][:]  # [nch, nsta, nev]

            select_idx = np.where(cc_score >= min_cc_score)
            cc_score = cc_score[select_idx]
            cc_index = cc_index[select_idx]

            phase_time = pd.to_datetime(mseeds.iloc[id1]["begin_time"]) + pd.to_timedelta(
                cc_index * dt + time_before, unit="s"
            )
            origin_time = phase_time - pd.to_timedelta(picks.iloc[id2]["travel_time"], unit="s")

            station_id = picks.iloc[id2]["station_id"]
            phase_type = picks.iloc[id2]["phase_type"]
            template_index = picks.iloc[id2]["event_index"]
            for o, p, s in zip(origin_time, phase_time, cc_score):
                detects.append([o, p, s, station_id, phase_type, template_index])

    return 0


if __name__ == "__main__":
    # %%
    root_path = "local"
    region = "demo"

    # %%
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    # %%
    result_path = f"{region}/qtm"

    # %%
    with open(f"{root_path}/{result_path}/config.json", "r") as fp:
        config = json.load(fp)
    config["min_cc_score"] = 0.6
    config["min_cc_diff"] = 0.0

    h5_list = sorted(list(glob(f"{root_path}/{result_path}/ccpairs/*.h5")))

    # %%
    picks = pd.read_csv(f"{root_path}/{result_path}/picks.csv", parse_dates=["phase_time"])
    picks["travel_time"] = picks["phase_timestamp"] - picks["event_timestamp"]

    # %%
    mseeds = pd.read_csv(f"{root_path}/{result_path}/mseed_list.csv", parse_dates=["begin_time"])

    # %%
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        data = manager.list()
        pair_list = []
        num_pair = 0
        for h5 in h5_list:
            with h5py.File(h5, "r") as fp:
                for id1 in tqdm(fp, desc=f"Loading {h5.split('/')[-1]}", leave=True):
                    gp1 = fp[id1]
                    pair_list.append([h5, id1])
                    num_pair += len(gp1.keys())

        ncpu = max(1, min(16, mp.cpu_count() - 1))
        print(f"Total pairs: {num_pair}. Using {ncpu} cores.")
        pbar = tqdm(total=len(pair_list), desc="Extracting pairs")

        with ctx.Pool(processes=ncpu) as pool:
            # with mp.Pool(processes=ncpu) as pool:
            for pair in pair_list:
                pool.apply_async(
                    extract_picks, args=(pair, data, config, mseeds, picks), callback=lambda x: pbar.update()
                )
            pool.close()
            pool.join()
        pbar.close()

        data = list(data)

    # %%
    detects = pd.DataFrame(
        data, columns=["event_time", "phase_time", "cc_score", "station_id", "phase_type", "template_index"]
    )
    detects["cc_score"] = detects["cc_score"].round(3)
    detects.sort_values(by="event_time", inplace=True)
    detects.to_csv(f"{root_path}/{result_path}/qtm_detects_raw.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    print(f"Number of detected phases: {len(detects)}")

    # %%
    ###################### Association based only on event time ######################
    # t0 = detects["event_time"].min()
    # detects["timestamp"] = detects["event_time"].apply(lambda x: (x - t0).total_seconds())
    # clustering = DBSCAN(eps=3, min_samples=3).fit(detects[["timestamp"]])
    # detects["event_index"] = clustering.labels_

    # events = (
    #     detects.groupby("event_index")
    #     .agg({"timestamp": "mean", "cc_score": "sum", "station_id": "count"})
    #     .reset_index()
    # )
    # events.rename(columns={"station_id": "num_station"}, inplace=True)
    # events["event_time"] = events["timestamp"].apply(lambda x: t0 + pd.to_timedelta(x, unit="s"))
    # events.sort_values(by="event_time", inplace=True)
    # events = events[events["event_index"] != -1]
    # if "timestamp" in events.columns:
    #     events.drop(columns=["timestamp"], inplace=True)
    # events["cc_score"] = events["cc_score"].round(3)
    # events.to_csv(f"{root_path}/{result_path}/qtm_events.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    # print(f"Number of associated events: {len(events)}")

    # detects.sort_values(by="event_time", inplace=True)
    # if "timestamp" in detects.columns:
    #     detects.drop(columns=["timestamp"], inplace=True)
    # detects.to_csv(f"{root_path}/{result_path}/qtm_detects.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    # print(f"Number of detected phases: {len(detects)}")
    # print(f"Number of associated phases: {len(detects[detects['event_index'] != -1])}")

    # %%
    ###################### Association based on phase time then event time ######################
    # detects = pd.read_csv(f"{root_path}/{result_path}/qtm_detects_raw.csv", parse_dates=["event_time", "phase_time"])
    t0 = detects["event_time"].min()
    detects["timestamp"] = detects["phase_time"].apply(lambda x: (x - t0).total_seconds())

    # %%
    detects_filt = []
    for station in tqdm(detects["station_id"].unique(), desc="Filtering"):
        detects_ = detects[detects["station_id"] == station].copy()

        clustering = DBSCAN(eps=2, min_samples=1).fit(detects_[["timestamp"]])
        detects_["event_index"] = clustering.labels_
        grouped = detects_.groupby("event_index")["cc_score"].idxmax()
        detects_ = detects_.loc[grouped]
        detects_filt.append(detects_)

    detects = pd.concat(detects_filt)
    print(f"Number of filtered phases: {len(detects)}")

    # %%
    detects["timestamp"] = detects["event_time"].apply(lambda x: (x - t0).total_seconds())
    clustering = DBSCAN(eps=2, min_samples=3).fit(detects[["timestamp"]])
    detects["event_index"] = clustering.labels_
    detects.sort_values(by="event_time", inplace=True)
    detects.to_csv(f"{root_path}/{result_path}/qtm_detects.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")

    # %%
    events = (
        detects.groupby("event_index")
        .agg({"timestamp": "mean", "cc_score": "sum", "station_id": "count"})
        .reset_index()
    )
    events.rename(columns={"station_id": "num_station"}, inplace=True)
    events["event_time"] = events["timestamp"].apply(lambda x: t0 + pd.to_timedelta(x, unit="s"))
    events.sort_values(by="event_time", inplace=True)
    events = events[events["event_index"] != -1]
    if "timestamp" in events.columns:
        events.drop(columns=["timestamp"], inplace=True)
    events["cc_score"] = events["cc_score"].round(3)
    events.to_csv(f"{root_path}/{result_path}/qtm_events.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    print(f"Number of associated events: {len(events)}")

    # # %%
    # ###### Debug ######
    # root_path = "local"
    # region = "demo"
    # result_path = f"{region}/qtm"
    # events = pd.read_csv(f"{root_path}/{result_path}/qtm_events.csv", parse_dates=["event_time"])
    # detects = pd.read_csv(f"{root_path}/{result_path}/qtm_detects.csv", parse_dates=["event_time", "phase_time"])
    # catalog = pd.read_csv(f"{root_path}/{region}/results/data/catalog.csv", parse_dates=["time"])
    # mapping = {k: v for v, k in enumerate(detects["station_id"].unique())}

    # print(f"Number of associated events: {len(events)}")
    # print(f"Number of detected phases: {len(detects)}")

    # # %%
    # plt.figure(figsize=(10, 5))
    # bins = pd.date_range(start=detects["event_time"].min(), end=detects["event_time"].max(), periods=30)
    # plt.hist(events["event_time"], bins=bins, edgecolor="k", linewidth=0.5, label=f"QTM ({len(events)})")
    # plt.hist(catalog["time"], bins=bins, edgecolor="k", linewidth=0.5, alpha=0.5, label=f"Catalog ({len(catalog)})")
    # plt.savefig("debug_qtm_events_hist.png", dpi=300)

    # # %%
    # events = events[
    #     (events["event_time"] >= pd.to_datetime("2019-07-04T23:00:00"))
    #     & (events["event_time"] <= pd.to_datetime("2019-07-05T00:00:00"))
    # ]
    # detects = detects[
    #     (detects["event_time"] >= pd.to_datetime("2019-07-04T23:00:00"))
    #     & (detects["event_time"] <= pd.to_datetime("2019-07-05T00:00:00"))
    # ]
    # print(f"Selected associated events: {len(events)}")
    # print(f"Selected detected phases: {len(detects)}")
    # # %%
    # plt.figure(figsize=(20, 5))
    # plt.scatter(
    #     detects["event_time"],
    #     detects["station_id"].map(mapping),
    #     c=[f"C{x}" if x != -1 else "k" for x in detects["event_index"]],
    #     # s=10 ** detects["cc_score"],
    #     s=1,
    # )
    # ylim = plt.ylim()
    # xlim = plt.xlim()
    # for i, row in catalog.iterrows():
    #     plt.plot([row["time"], row["time"]], ylim, "-k", alpha=0.3, linewidth=0.5)
    # plt.ylim(ylim)
    # plt.xlim(xlim)
    # plt.savefig("debug_qtm_events.png", dpi=300)

    # # %%
    # plt.figure(figsize=(20, 5))
    # plt.scatter(
    #     detects["phase_time"],
    #     detects["station_id"].map(mapping),
    #     c=[f"C{x}" if x != -1 else "k" for x in detects["event_index"]],
    #     # s=2 ** detects["cc_score"],
    #     s=1,
    # )
    # ylim = plt.ylim()
    # xlim = plt.xlim()
    # for i, row in catalog.iterrows():
    #     plt.plot([row["time"], row["time"]], ylim, "-k", alpha=0.3, linewidth=0.5)
    # plt.ylim(ylim)
    # plt.xlim(xlim)
    # plt.savefig("debug_qtm_phases.png", dpi=300)

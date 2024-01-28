# %%
from typing import Dict, List

from kfp import dsl


@dsl.component()
def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    rank: int = 0,
    data_type: str = "continuous",
    overwrite: bool = False,
    mseed_list: List = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:
    # %%
    import os
    from glob import glob

    import fsspec
    import torch

    # %%
    if data_type == "continuous":
        folder_depth = 3
    elif data_type == "event":
        folder_depth = 1

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    result_path = f"{region}/phasenet_plus"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}", exist_ok=True)

    # %%
    waveform_dir = f"{region}/waveforms"
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        if protocol != "file":
            fs.get(f"{bucket}/{waveform_dir}/", f"{root_path}/{waveform_dir}/", recursive=True)

    if mseed_list is None:
        if data_type == "continuous":
            mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????-???/??/*.mseed"))
        elif data_type == "event":
            mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/*.mseed"))
        else:
            raise ValueError("data_type must be either continuous or event")

    # %% group channels into stations
    if data_type == "continuous":
        mseed_list = list(set([x.split(".mseed")[0][:-1] + "*.mseed" for x in mseed_list]))
    mseed_list = sorted(mseed_list)

    # %% skip processed files
    if not overwrite:
        picks_list = sorted(glob(f"{root_path}/{result_path}/picks_phasenet_plus/????-???/??/*.csv"))
        processed_list = set(["/".join(x.replace(".csv", "*.mseed").split("/")[-folder_depth:]) for x in picks_list])
        mseed_list = [x for x in mseed_list if "/".join(x.split("/")[-folder_depth:]) not in processed_list]

    # %%
    if protocol != "file":
        fs.get(f"{bucket}/{region}/obspy/inventory.xml", f"{root_path}/{region}/obspy/inventory.xml")

    # %%
    with open(f"{root_path}/{result_path}/mseed_list_{rank:03d}.csv", "w") as fp:
        fp.write("\n".join(mseed_list))

    num_gpu = torch.cuda.device_count()
    print(f"num_gpu = {num_gpu}")
    # base_cmd = f"../EQNet/predict.py --model phasenet_plus --add_polarity --add_event --format mseed --data_list={root_path}/{result_path}/mseed_list_{rank:03d}.csv --response_xml={root_path}/{region}/obspy/inventory.xml --result_path={root_path}/{result_path} --batch_size 1 --workers 1 --folder_depth {folder_depth}"
    base_cmd = f"../EQNet/predict.py --model phasenet_plus --add_polarity --add_event --format mseed --data_list={root_path}/{result_path}/mseed_list_{rank:03d}.csv --response_path={root_path}/{region}/results/data/inventory --result_path={root_path}/{result_path} --batch_size 1 --workers 1 --folder_depth {folder_depth}"
    if num_gpu == 0:
        cmd = f"python {base_cmd} --device=cpu"
    elif num_gpu == 1:
        cmd = f"python {base_cmd}"
    else:
        cmd = f"torchrun --standalone --nproc_per_node {num_gpu} {base_cmd}"
    print(cmd)
    os.system(cmd)

    os.system(
        f"cp {root_path}/{result_path}/picks_phasenet_plus.csv {root_path}/{result_path}/phasenet_plus_picks_{rank:03d}.csv"
    )
    os.system(
        f"cp {root_path}/{result_path}/events_phasenet_plus.csv {root_path}/{result_path}/phasenet_plus_events_{rank:03d}.csv",
    )

    if protocol != "file":
        fs.put(
            f"{root_path}/{result_path}/phasenet_plus_picks_{rank:03d}.csv",
            f"{bucket}/{result_path}/phasenet_plus_picks_{rank:03d}.csv",
        )
        fs.put(
            f"{root_path}/{result_path}/phasenet_plus_events_{rank:03d}.csv",
            f"{bucket}/{result_path}/phasenet_plus_events_{rank:03d}.csv",
        )

    # copy to results/phase_picking
    if not os.path.exists(f"{root_path}/{region}/results/phase_picking"):
        os.makedirs(f"{root_path}/{region}/results/phase_picking")
    os.system(
        f"cp {root_path}/{result_path}/phasenet_plus_picks_{rank:03d}.csv {root_path}/{region}/results/phase_picking/phase_picks_{rank:03d}.csv"
    )
    os.system(
        f"cp {root_path}/{result_path}/phasenet_plus_events_{rank:03d}.csv {root_path}/{region}/results/phase_picking/phase_events_{rank:03d}.csv"
    )
    if protocol != "file":
        fs.put(
            f"{root_path}/{result_path}/phasenet_plus_picks_{rank:03d}.csv",
            f"{bucket}/{region}/results/phase_picking/phase_picks_{rank:03d}.csv",
        )
        fs.put(
            f"{root_path}/{result_path}/phasenet_plus_events_{rank:03d}.csv",
            f"{bucket}/{region}/results/phase_picking/phase_events_{rank:03d}.csv",
        )

    return f"{result_path}/phasenet_picks_{rank:03d}.csv"


@dsl.component()
def run_association(
    root_path: str,
    region: str,
    config: Dict,
    rank: int = 0,
    pick_csv: str = "phasenet_plus_picks_000.csv",
    event_csv: str = "phasenet_plus_events_000.csv",
    station_json: str = "stations.json",
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:
    # %%
    import json
    import os

    import fsspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN
    from tqdm import tqdm

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    PS_RATIO = 1.73
    VP = 6.0

    # %%
    data_path = f"{region}/phasenet_plus"
    result_path = f"{region}/phasenet_plus"

    # %%
    # stations = pd.read_json(f"{root_path}/{region}/results/data/{station_json}", orient="index")
    # stations["station_id"] = stations.index
    events = pd.read_csv(f"{root_path}/{data_path}/{event_csv}", parse_dates=["center_time", "event_time"])
    picks = pd.read_csv(f"{root_path}/{data_path}/{pick_csv}", parse_dates=["phase_time"])

    # %%
    # events = events.merge(stations, on="station_id", how="left")
    # events["x_s"] = events["x_km"] / VP
    # events["y_s"] = events["y_km"] / VP
    # %%
    t0 = min(events["event_time"].min(), picks["phase_time"].min())
    events["timestamp"] = events["event_time"].apply(lambda x: (x - t0).total_seconds())
    events["timestamp_center"] = events["center_time"].apply(lambda x: (x - t0).total_seconds())
    picks["timestamp"] = picks["phase_time"].apply(lambda x: (x - t0).total_seconds())

    # %%
    # station_ids = events.sort_values(["x_km", "y_km"])["station_id"].unique()
    station_ids = events["station_id"].unique()
    mapping = {station_id: i for i, station_id in enumerate(station_ids)}

    # %%
    # clustering = DBSCAN(eps=3, min_samples=3).fit(events[["timestamp", "x_s", "y_s"]])
    clustering = DBSCAN(eps=3, min_samples=3).fit(events[["timestamp"]])
    events["event_index"] = clustering.labels_
    print(f"Number of associated events: {len(events['event_index'].unique())}")

    # plt.figure(figsize=(10, 5))
    # plt.scatter(
    #     events["event_time"],
    #     events["station_id"].map(mapping),
    #     c=[f"C{x}" if x != -1 else "k" for x in events["event_index"]],
    #     s=1,
    # )
    # plt.xlim(pd.Timestamp("2019-07-04T18:00:00"), pd.Timestamp("2019-07-04T18:10:00"))

    # %% link picks to events
    picks["event_index"] = -1
    picks.set_index("station_id", inplace=True)

    for group_id, event in tqdm(events.groupby("station_id"), desc="Linking picks to events"):
        # travel time tt = (tp + ts) / 2 = (ps_ratio + 1)/2 * tp,
        # (ts - tp) = (ps_ratio - 1) tp = tt * (ps_ratio + 1) * 2 * (ps_ratio - 1)
        ps_delta = event["travel_time_s"] / (PS_RATIO + 1) * 2 * (PS_RATIO - 1)
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
    events.to_csv(f"{root_path}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv", index=False)
    picks.to_csv(f"{root_path}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv", index=False)

    if protocol != "file":
        fs.put(
            f"{root_path}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv",
            f"{bucket}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv",
        )
        fs.put(
            f"{root_path}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv",
            f"{bucket}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv",
        )

    # copy to results/phase_association
    if not os.path.exists(f"{root_path}/{region}/results/phase_association"):
        os.makedirs(f"{root_path}/{region}/results/phase_association")
    os.system(
        f"cp {root_path}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv {root_path}/{region}/results/phase_association/events_{rank:03d}.csv"
    )
    os.system(
        f"cp {root_path}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv {root_path}/{region}/results/phase_association/picks_{rank:03d}.csv"
    )
    if protocol != "file":
        fs.put(
            f"{root_path}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv",
            f"{bucket}/{region}/results/phase_association/events_{rank:03d}.csv",
        )
        fs.put(
            f"{root_path}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv",
            f"{bucket}/{region}/results/phase_association/picks_{rank:03d}.csv",
        )


if __name__ == "__main__":
    import json
    import os
    import sys

    root_path = "local"
    region = "demo"
    data_type = "continuous"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]
    if len(sys.argv) > 3:
        data_type = sys.argv[3]

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    run_phasenet.execute(root_path=root_path, region=region, config=config, data_type=data_type)

    run_association.execute(root_path=root_path, region=region, config=config)

# %%

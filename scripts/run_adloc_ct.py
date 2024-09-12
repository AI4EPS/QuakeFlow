# %%
import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from adloc.adloc import TravelTimeDD
from adloc.data import PhaseDatasetDT
from adloc.eikonal2d import init_eikonal2d
from adloc.inversion import optimize_dd
from args import parse_args
from matplotlib import pyplot as plt
from pyproj import Proj
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.plotting import plotting_dd

torch.manual_seed(0)
np.random.seed(0)


# %%
if __name__ == "__main__":

    # %%
    args = parse_args()
    root_path = args.root_path
    region = args.region
    data_path = f"{root_path}/{region}/adloc_dd"
    result_path = f"{root_path}/{region}/adloc_dd"
    figure_path = f"{root_path}/{region}/adloc_dd/figures"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # %%
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend="gloo")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        print(f"DDP rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size}")
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        print("Non-DDP run")

    # %% reading from the generated files
    events = pd.read_csv(os.path.join(result_path, "pair_events.csv"), parse_dates=["time"])
    stations = pd.read_csv(os.path.join(result_path, "pair_stations.csv"))
    picks = pd.read_csv(os.path.join(result_path, "pair_picks.csv"), parse_dates=["phase_time"])
    dtypes = pickle.load(open(os.path.join(result_path, "pair_dtypes.pkl"), "rb"))
    pairs = np.memmap(os.path.join(result_path, "pair_dt.dat"), mode="r", dtype=dtypes)

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)
    print(json.dumps(config, indent=4))
    config["use_amplitude"] = True

    # ## Eikonal for 1D velocity model
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 5.5, 6.7, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.3

    # %%
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # %%
    ## Automatic region; you can also specify a region
    # lon0 = stations["longitude"].median()
    # lat0 = stations["latitude"].median()
    lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
    lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0} +units=km")

    stations["x_km"], stations["y_km"] = proj(stations["longitude"], stations["latitude"])
    stations["z_km"] = stations["depth_km"]
    events["time"] = pd.to_datetime(events["time"])
    events["x_km"], events["y_km"] = proj(events["longitude"], events["latitude"])
    events["z_km"] = events["depth_km"]

    events_init = events.copy()

    ## set up the config; you can also specify the region manually
    if ("xlim_km" not in config) or ("ylim_km" not in config) or ("zlim_km" not in config):
        xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
        xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
        zmin, zmax = config["mindepth"], config["maxdepth"]
        config["xlim_km"] = (xmin, xmax)
        config["ylim_km"] = (ymin, ymax)
        config["zlim_km"] = (zmin, zmax)

    mapping_phase_type_int = {"P": 0, "S": 1}
    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}
    config["vel"] = {mapping_phase_type_int[k]: v for k, v in config["vel"].items()}

    # %%
    config["eikonal"] = None
    ## Eikonal for 1D velocity model
    # zz = [0.0, 5.5, 16.0, 32.0]
    # vp = [5.5, 5.5, 6.7, 7.8]
    # vp_vs_ratio = 1.73
    # vs = [v / vp_vs_ratio for v in vp]
    # zz = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0]
    # vp = [4.746, 4.793, 4.799, 5.045, 5.721, 5.879, 6.504, 6.708, 6.725, 7.800]
    # vs = [2.469, 2.470, 2.929, 2.930, 3.402, 3.403, 3.848, 3.907, 3.963, 4.500]
    # h = 0.3
    vel = {"Z": zz, "P": vp, "S": vs}
    config["eikonal"] = {
        "vel": vel,
        "h": h,
        "xlim_km": config["xlim_km"],
        "ylim_km": config["ylim_km"],
        "zlim_km": config["zlim_km"],
    }
    config["eikonal"] = init_eikonal2d(config["eikonal"])

    # %%
    config["bfgs_bounds"] = (
        (config["xlim_km"][0] - 1, config["xlim_km"][1] + 1),  # x
        (config["ylim_km"][0] - 1, config["ylim_km"][1] + 1),  # y
        (0, config["zlim_km"][1] + 1),
        (None, None),  # t
    )

    # %%
    phase_dataset = PhaseDatasetDT(pairs, picks, events, stations, rank=ddp_local_rank, world_size=ddp_world_size)
    data_loader = DataLoader(phase_dataset, batch_size=None, shuffle=False, num_workers=0, drop_last=False)

    # %%
    num_event = len(events)
    num_station = len(stations)
    station_loc = stations[["x_km", "y_km", "z_km"]].values
    event_loc = events[["x_km", "y_km", "z_km"]].values  # + np.random.randn(num_event, 3) * 2.0
    travel_time = TravelTimeDD(
        num_event,
        num_station,
        station_loc=station_loc,
        event_loc=event_loc,
        # event_time=event_time,
        eikonal=config["eikonal"],
    )
    if ddp:
        travel_time = DDP(travel_time)
    raw_travel_time = travel_time.module if ddp else travel_time

    if ddp_local_rank == 0:
        print(f"Dataset: {len(events)} events, {len(stations)} stations, {len(data_loader)} batches")

    ## invert loss
    ######################################################################################################
    optimizer = optim.Adam(params=travel_time.parameters(), lr=0.1)
    valid_index = np.ones(len(pairs), dtype=bool)
    EPOCHS = 100
    for epoch in range(EPOCHS):
        loss = 0
        optimizer.zero_grad()
        # for meta in tqdm(phase_dataset, desc=f"Epoch {i}"):
        for meta in data_loader:
            out = travel_time(
                meta["idx_sta"],
                meta["idx_eve"],
                meta["phase_type"],
                meta["phase_time"],
                meta["phase_weight"],
            )
            pred_, loss_ = out["phase_time"], out["loss"]

            loss_.backward()

            if ddp:
                dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
            # loss_ /= ddp_world_size

            loss += loss_

        # torch.nn.utils.clip_grad_norm_(travel_time.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            raw_travel_time.event_loc.weight.data[:, 2].clamp_(
                min=config["zlim_km"][0] + 0.1, max=config["zlim_km"][1] - 0.1
            )
            raw_travel_time.event_loc.weight.data[torch.isnan(raw_travel_time.event_loc.weight)] = 0.0
        if ddp_local_rank == 0:
            print(f"Epoch {epoch}: loss {loss:.6e} of {np.sum(valid_index)} picks, {loss / np.sum(valid_index):.6e}")

        ### filtering
        pred_time = []
        phase_dataset.valid_index = np.ones(len(pairs), dtype=bool)
        for meta in phase_dataset:
            meta = travel_time(
                meta["idx_sta"],
                meta["idx_eve"],
                meta["phase_type"],
                meta["phase_time"],
                meta["phase_weight"],
            )
            pred_time.append(meta["phase_time"].detach().numpy())

        pred_time = np.concatenate(pred_time)
        valid_index = (
            np.abs(pred_time - pairs["phase_dtime"]) < np.std((pred_time - pairs["phase_dtime"])[valid_index]) * 3.0
        )  # * (np.cos(epoch * np.pi / EPOCHS) + 2.0) # 3std -> 1std

        pairs_df = pd.DataFrame(
            {
                "event_index1": pairs["event_index1"],
                "event_index2": pairs["event_index2"],
                "station_index": pairs["station_index"],
            }
        )
        pairs_df = pairs_df[valid_index]
        config["MIN_OBS"] = 8
        pairs_df = pairs_df.groupby(["event_index1", "event_index2"], as_index=False, group_keys=False).filter(
            lambda x: len(x) >= config["MIN_OBS"]
        )
        valid_index = np.zeros(len(pairs), dtype=bool)
        valid_index[pairs_df.index] = True

        phase_dataset.valid_index = valid_index

        invert_event_loc = raw_travel_time.event_loc.weight.clone().detach().numpy()
        invert_event_time = raw_travel_time.event_time.weight.clone().detach().numpy()
        valid_event_index = np.unique(pairs["event_index1"][valid_index])
        valid_event_index = np.concatenate(
            [np.unique(pairs["event_index1"][valid_index]), np.unique(pairs["event_index2"][valid_index])]
        )
        valid_event_index = np.sort(np.unique(valid_event_index))

        if ddp_local_rank == 0 and (epoch % 10 == 0):
            events = events_init.copy()
            events["time"] = events["time"] + pd.to_timedelta(np.squeeze(invert_event_time), unit="s")
            events["x_km"] = invert_event_loc[:, 0]
            events["y_km"] = invert_event_loc[:, 1]
            events["z_km"] = invert_event_loc[:, 2]
            events[["longitude", "latitude"]] = events.apply(
                lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
            )
            events["depth_km"] = events["z_km"]
            events = events.iloc[valid_event_index]
            events.to_csv(
                f"{result_path}/adloc_dt_events_{epoch//10}.csv",
                index=False,
                float_format="%.5f",
                date_format="%Y-%m-%dT%H:%M:%S.%f",
            )
            plotting_dd(events, stations, config, figure_path, events_init, suffix=f"_dt_{epoch//10}")

    # ######################################################################################################
    # optimizer = optim.LBFGS(params=raw_travel_time.parameters(), max_iter=10, line_search_fn="strong_wolfe")

    # def closure():
    #     optimizer.zero_grad()
    #     loss = 0
    #     # for meta in tqdm(phase_dataset, desc=f"BFGS"):
    #     if ddp_local_rank == 0:
    #         print(f"BFGS: ", end="")
    #     for meta in phase_dataset:
    #         if ddp_local_rank == 0:
    #             print(".", end="")

    #         loss_ = travel_time(
    #              meta["idx_sta"],
    #             meta["idx_eve"],
    #             meta["phase_type"],
    #             meta["phase_time"],
    #             meta["phase_weight"],
    #         )["loss"]
    #         loss_.backward()

    #         if ddp:
    #             dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
    #             # loss_ /= ddp_world_size

    #         loss += loss_

    #     if ddp_local_rank == 0:
    #         print(f"Loss: {loss}")
    #     raw_travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])
    #     return loss

    # optimizer.step(closure)
    # ######################################################################################################

    # %%
    if ddp_local_rank == 0:

        plotting_dd(events, stations, config, figure_path, events_init, suffix="_dt")

        invert_event_loc = raw_travel_time.event_loc.weight.clone().detach().numpy()
        invert_event_time = raw_travel_time.event_time.weight.clone().detach().numpy()

        events = events_init.copy()
        events["time"] = events["time"] + pd.to_timedelta(np.squeeze(invert_event_time), unit="s")
        events["x_km"] = invert_event_loc[:, 0]
        events["y_km"] = invert_event_loc[:, 1]
        events["z_km"] = invert_event_loc[:, 2]
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z_km"]
        events = events.iloc[valid_event_index]
        events.to_csv(
            f"{result_path}/adloc_dt_events.csv", index=False, float_format="%.5f", date_format="%Y-%m-%dT%H:%M:%S.%f"
        )

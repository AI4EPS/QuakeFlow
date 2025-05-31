# %%
import json
import os
from concurrent.futures import ThreadPoolExecutor

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from cut_templates_cc import generate_pairs
from pyproj import Proj

# %%
if __name__ == "__main__":

    # %%
    region = "Cal"
    result_path = f"{region}/cctorch"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # %%
    protocol = "gs"
    bucket = "quakeflow_catalog"
    folder = result_path
    token_json = f"application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)

    fs = fsspec.filesystem(protocol, token=token)

    # %%
    def plot_templates(templates, events, picks):
        templates = templates - np.nanmean(templates, axis=(-1), keepdims=True)
        std = np.std(templates, axis=(-1), keepdims=True)
        std[std == 0] = 1.0
        templates = templates / std

        plt.figure(figsize=(10, 10))
        plt.imshow(templates[:, -1, 0, :], origin="lower", aspect="auto", vmin=-0.3, vmax=0.3, cmap="RdBu_r")
        plt.colorbar()
        plt.show()

    # %%

    region = "Cal"
    root_path = "local"
    protocol = "gs"
    bucket = "quakeflow_catalog"    
    station_json = f"{region}/network/stations.json"
    if protocol == "file":
        stations = pd.read_json(f"{root_path}/{station_json}", orient="index")
    else:
        with fs.open(f"{bucket}/{station_json}", "r") as fp:
            stations = pd.read_json(fp, orient="index")
    stations["station_id"] = stations.index
    stations.sort_values(by=["latitude", "longitude"], inplace=True)
    print(f"stations: {len(stations):,} ")
    print(stations.iloc[:5])

    # %%
    years = [2019]
    # jdays = [185, 186, 187]
    jdays = [185, 186]
    # jdays = [185]

    picks = []
    events = []
    templates = []
    traveltimes = []
    traveltime_indices = []
    traveltime_masks = []

    for year in years:
        # num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
        # jdays = [185, 186, 187]
       
        for jday in tqdm(jdays, desc=f"Loading {year}"):

            if not fs.exists(f"{bucket}/{folder}/{year}/{jday:03d}/template.dat"):
                continue

            with fs.open(f"{bucket}/{folder}/{year}/{jday:03d}/cctorch_picks.csv", "r") as fp:
                picks_ = pd.read_csv(fp, dtype=str)
            with fs.open(f"{bucket}/{folder}/{year}/{jday:03d}/cctorch_events.csv", "r") as fp:
                events_ = pd.read_csv(fp, dtype=str)
            with fs.open(f"{bucket}/{folder}/{year}/{jday:03d}/config.json", "r") as fp:
                config = json.load(fp)
            template_file_ = fs.open(f"{bucket}/{folder}/{year}/{jday:03d}/template.dat", "rb")
            templates_ = np.frombuffer(template_file_.read(), dtype=np.float32).reshape(tuple(config["template_shape"]))
            template_file_.close()
            traveltime_file_ = fs.open(f"{bucket}/{folder}/{year}/{jday:03d}/traveltime.dat", "rb")
            traveltimes_ = np.frombuffer(traveltime_file_.read(), dtype=np.float32).reshape(tuple(config["traveltime_shape"]))
            traveltime_file_.close()
            traveltime_index_file_ = fs.open(f"{bucket}/{folder}/{year}/{jday:03d}/traveltime_index.dat", "rb")
            traveltime_indices_ = np.frombuffer(traveltime_index_file_.read(), dtype=np.int32).reshape(tuple(config["traveltime_shape"]))
            traveltime_index_file_.close()
            traveltime_mask_file_ = fs.open(f"{bucket}/{folder}/{year}/{jday:03d}/traveltime_mask.dat", "rb")
            traveltime_masks_ = np.frombuffer(traveltime_mask_file_.read(), dtype=bool).reshape(tuple(config["traveltime_shape"]))
            traveltime_mask_file_.close()

            #### DEBUG ####
            minlat, maxlat = 35.205, 36.205
            minlon, maxlon = -118.004, -117.004
            events_["latitude"] = events_["latitude"].astype(float)
            events_["longitude"] = events_["longitude"].astype(float)
            events_ = events_[(events_["latitude"] > minlat) & (events_["latitude"] < maxlat) & (events_["longitude"] > minlon) & (events_["longitude"] < maxlon)]
            ##############


            picks_["year"] = year
            picks_["jday"] = jday
            events_["year"] = year
            events_["jday"] = jday

            # idx = picks_["idx_pick"].values.astype(int)
            idx = picks_["idx_pick"].values[picks_["idx_eve"].isin(events_["idx_eve"].values)]
            idx = idx.astype(int)
            picks_ = picks_.iloc[idx]
            templates_ = templates_[idx]
            traveltimes_ = traveltimes_[idx]
            traveltime_indices_ = traveltime_indices_[idx]
            traveltime_masks_ = traveltime_masks_[idx]

            picks.append(picks_)
            events.append(events_)
            templates.append(templates_)
            traveltimes.append(traveltimes_)
            traveltime_indices.append(traveltime_indices_)
            traveltime_masks.append(traveltime_masks_)

    picks = pd.concat(picks, ignore_index=True)
    templates = np.concatenate(templates)
    traveltimes = np.concatenate(traveltimes)
    traveltime_indices = np.concatenate(traveltime_indices)
    traveltime_masks = np.concatenate(traveltime_masks)

    picks["idx_pick"] = np.arange(len(picks))
    assert len(picks) == len(templates)
    assert len(picks) == len(traveltimes)
    assert len(picks) == len(traveltime_indices)
    assert len(picks) == len(traveltime_masks)

    stations = stations[stations["station_id"].isin(picks["station_id"])]
    stations["idx_sta"] = np.arange(len(stations))
    print(f"stations: {len(stations):,} ")
    print(stations.iloc[:5])

    events = pd.concat(events, ignore_index=True)
    events = events.sort_values("event_time")
    events["dummy_id"] = (
        events["year"].astype(str)
        + "."
        + events["jday"].astype(str).str.zfill(3)
        + "."
        + events["idx_eve"].astype(str).str.zfill(6)
    )
    picks["dummy_id"] = (
        picks["year"].astype(str)
        + "."
        + picks["jday"].astype(str).str.zfill(3)
        + "."
        + picks["idx_eve"].astype(str).str.zfill(6)
    )
    events["idx_eve"] = np.arange(len(events))
    events["event_index"] = np.arange(len(events))
    print(f"events: {len(events):,} ")
    print(events.iloc[:5])


    # %%
    lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
    lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
    proj = Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0}  +units=km")
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z_km"] = stations["elevation_m"].apply(lambda x: -x / 1e3)
    if "station_term_time_p" not in stations:
        stations["station_term_time_p"] = 0.0
    if "station_term_time_s" not in stations:
        stations["station_term_time_s"] = 0.0

    events[["x_km", "y_km"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events["z_km"] = events["depth_km"]

    # %%
    picks = picks.drop(["idx_eve", "idx_sta"], axis=1)
    picks = picks.merge(events[["dummy_id", "idx_eve"]], on="dummy_id")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")
    print(f"picks: {len(picks):,} ")
    print(picks.iloc[:5])

    # %%
    stations.to_csv(f"{root_path}/{result_path}/cctorch_stations.csv", index=False)
    events.to_csv(f"{root_path}/{result_path}/cctorch_events.csv", index=False)

    events = events[["idx_eve", "x_km", "y_km", "z_km", "event_index", "event_time", "event_timestamp"]]
    stations = stations[["idx_sta", "x_km", "y_km", "z_km", "station_id", "component", "network", "station"]]
    picks = picks[
        [
            "idx_eve",
            "idx_sta",
            "phase_type",
            "phase_score",
            "phase_time",
            "phase_timestamp",
            "phase_source",
            "station_id",
            "idx_pick",
        ]
    ]
    events.set_index("idx_eve", inplace=True)
    stations.set_index("idx_sta", inplace=True)

    # %%
    template_fname = f"{root_path}/{result_path}/template.dat"
    traveltime_fname = f"{root_path}/{result_path}/traveltime.dat"
    traveltime_index_fname = f"{root_path}/{result_path}/traveltime_index.dat"
    traveltime_mask_fname = f"{root_path}/{result_path}/traveltime_mask.dat"
    pair_fname = f"{root_path}/{result_path}/pairs.txt"

    picks.to_csv(f"{root_path}/{result_path}/cctorch_picks.csv", index=False)
    template_array = np.memmap(template_fname, dtype=np.float32, mode="w+", shape=templates.shape)
    template_array[:] = templates
    template_array.flush()
    traveltime_array = np.memmap(traveltime_fname, dtype=np.float32, mode="w+", shape=traveltimes.shape)
    traveltime_array[:] = traveltimes
    traveltime_array.flush()
    traveltime_index_array = np.memmap(traveltime_index_fname, dtype=np.int32, mode="w+", shape=traveltime_indices.shape)
    traveltime_index_array[:] = traveltime_indices
    traveltime_index_array.flush()
    traveltime_mask_array = np.memmap(traveltime_mask_fname, dtype=bool, mode="w+", shape=traveltime_masks.shape)
    traveltime_mask_array[:] = traveltime_masks
    traveltime_mask_array.flush()
    config["template_shape"] = templates.shape
    config["traveltime_shape"] = traveltimes.shape
    config["template_file"] = template_fname
    config["traveltime_file"] = traveltime_fname
    config["traveltime_index_file"] = traveltime_index_fname
    config["traveltime_mask_file"] = traveltime_mask_fname
    config["pair_file"] = pair_fname
    with open(f"{root_path}/{result_path}/config.json", "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    # %%
    pairs = generate_pairs(
        picks,
        events,
        stations,
        max_pair_dist=config["max_pair_dist_km"],
        fname=config["pair_file"],
    )

    
    # raise

    # picks.to_csv(f"{result_path}/picks_{year:04d}_{jday:03d}.csv", index=False)
    # events.to_csv(f"{result_path}/events_{year:04d}_{jday:03d}.csv", index=False)
    # np.save(f"{result_path}/templates_{year:04d}_{jday:03d}.npy", templates)

    # plot_templates(templates, events, picks)

    # break

# %%

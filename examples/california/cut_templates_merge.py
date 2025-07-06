# %%
import json
import os
from concurrent.futures import ThreadPoolExecutor
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# from cut_templates_cc import generate_pairs
from pyproj import Proj
from args import parse_args
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime


# %%
def generate_pairs(picks, events, stations, max_pair_dist=10, max_neighbors=50, fname="pairs.txt"):
    ncpu = min(32, mp.cpu_count())
    neigh = NearestNeighbors(radius=max_pair_dist, n_neighbors=max_neighbors, n_jobs=ncpu)
    neigh.fit(events[["x_km", "y_km", "z_km"]].values)

    print(f"Generating pairs with max_pair_dist={max_pair_dist} km, max_neighbors={max_neighbors}")
    # event_pairs = []
    # for i, event in tqdm(events.iterrows(), total=len(events), desc="Generating pairs"):
    #     neigh_dist, neigh_ind = neigh.radius_neighbors([event[["x_km", "y_km", "z_km"]].values], sort_results=True)
    #     event_pairs.extend([[i, j] for j in neigh_ind[0][1:] if i < j])

    event_pairs = set()
    neigh_ind = neigh.radius_neighbors(sort_results=True)[1]
    for i, neighs in enumerate(tqdm(neigh_ind, desc="Generating event pairs")):
        for j in neighs[:max_neighbors]:
            if i < j:
                event_pairs.add((i, j))
            else:
                event_pairs.add((j, i))

    event_pairs = list(event_pairs)

    picks = picks.set_index(["idx_eve", "idx_sta", "phase_type"])
    idx_pick_dict = picks["idx_pick"].to_dict()  ## much faster than using .loc

    pairs = []
    for i, j in tqdm(event_pairs, desc="Generating pick pairs"):
        for idx_sta in stations.index:
            for phase_type in ["P", "S"]:

                if (i, idx_sta, phase_type) not in idx_pick_dict or (j, idx_sta, phase_type) not in idx_pick_dict:
                    continue

                idx_pick = idx_pick_dict[(i, idx_sta, phase_type)]
                jdx_pick = idx_pick_dict[(j, idx_sta, phase_type)]
                pairs.append([idx_pick, jdx_pick])

    np.savetxt(fname, pairs, fmt="%d,%d")

    return pairs


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

    # %%
    result_path = f"{region}/cctorch"
    folder = result_path
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)

    # %%
    with fs.open(f"{bucket}/{region}/config.json", "r") as fp:
        config = json.load(fp)
    # with open(args.config, "r") as fp:
    #     config = json.load(fp)
    config.update(vars(args))
    print(json.dumps(config, indent=4, sort_keys=True))

    # %% CCTorch
    # config["max_pair_dist_km"] = 10
    # config["max_neighbors"] = 50
    # config["max_obs"] = 100
    # config["min_obs"] = 8
    # config["nt"] = 400
    # config["maxdepth"] = 60
    # config["mindepth"] = 0
    # config["max_epicenter_dist_km"] = 200.0

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
    # station_json = f"{region}/network/stations.json"
    # if protocol == "file":
    #     stations = pd.read_json(f"{root_path}/{station_json}", orient="index")
    # else:
    #     with fs.open(f"{bucket}/{station_json}", "r") as fp:
    #         stations = pd.read_json(fp, orient="index")
    # stations["station_id"] = stations.index

    station_csv = f"{region}/adloc/ransac_stations.csv"
    # station_csv = f"{region}/adloc/adloc_stations.csv"
    if protocol == "file":
        stations = pd.read_csv(f"{root_path}/{station_csv}")
    else:
        with fs.open(f"{bucket}/{station_csv}", "r") as fp:
            stations = pd.read_csv(fp)
    stations.sort_values(by=["latitude", "longitude"], inplace=True)
    print(f"stations: {len(stations):,} ")
    print(stations.iloc[:5])

    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    # years = [2019]
    # jdays = [185, 186, 187]
    # jdays = [185, 186]
    # jdays = [185]
    # years = [2023]
    # years = [2019]
    # years = range(2000, datetime.datetime.now().year + 1)
    # jdays = []
    # for year in tqdm(years, desc="Checking existing days"):
    #     num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
    #     exist_jdays = fs.glob(f"{bucket}/{folder}/{year:04d}/???/template.dat")
    #     exist_jdays = [x.split("/")[-2] for x in exist_jdays]
    #     jdays.extend([(year, jday) for jday in range(1, num_jday + 1) if f"{jday:03d}" not in exist_jdays])

    exit_jdays = fs.glob(f"{bucket}/{folder}/????/???/template.dat")
    # exit_jdays = [(year, jday) for year_jday in exit_jdays for year, jday in year_jday.split("/")[-3:-1]]
    # for exit_jday in exit_jdays:
    #     year, jday = exit_jday.split("/")[-3:-1]
    # exit_jdays.append((year, jday))
    # selected_years = [2015]
    # selected_jdays = [185, 186, 187]
    # selected_years = [2020, 2021, 2022, 2023]
    year_jday = [
        (year, jday)
        for tmp in exit_jdays
        for (year, jday) in [tmp.split("/")[-3:-1]]
        # if (int(year) in selected_years)
        # if (int(year) in selected_years and int(jday) in selected_jdays)
    ]
    print(f"Selected jdays: {len(year_jday)}")

    # raise
    # year_jday = [(year, jday) for year in years for jday in jdays]

    #     year_jday.extend([(year, jday) for jday in range(1, num_jday + 1)])
    # # year_jday = [(year, jday) for year in years for jday in jdays]

    # exist = fs.glob(f"{bucket}/{folder}/????/???/template.dat")
    # print(f"Exist: {len(exist)}")
    # raise

    picks = []
    events = []
    templates = []
    traveltimes = []
    traveltime_indices = []
    traveltime_masks = []

    # for year in years:
    #     # num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
    #     # jdays = [185, 186, 187]

    #     for jday in tqdm(jdays, desc=f"Loading {year}"):
    # for year, jday in tqdm(year_jday, desc="Loading"):
    def process_day(year_jday):
        year, jday = year_jday

        if not fs.exists(f"{bucket}/{folder}/{year}/{jday}/template.dat"):
            print(f"Not found {bucket}/{folder}/{year}/{jday}/template.dat")
            # continue
            return ([], [], [], [], [], [])

        with fs.open(f"{bucket}/{folder}/{year}/{jday}/cctorch_picks.csv", "r") as fp:
            picks_ = pd.read_csv(fp, dtype=str)
        with fs.open(f"{bucket}/{folder}/{year}/{jday}/cctorch_events.csv", "r") as fp:
            events_ = pd.read_csv(fp, dtype=str)
        with fs.open(f"{bucket}/{folder}/{year}/{jday}/config.json", "r") as fp:
            config_ = json.load(fp)
        template_file_ = fs.open(f"{bucket}/{folder}/{year}/{jday}/template.dat", "rb")
        templates_ = np.frombuffer(template_file_.read(), dtype=np.float32).reshape(tuple(config_["template_shape"]))
        template_file_.close()
        traveltime_file_ = fs.open(f"{bucket}/{folder}/{year}/{jday}/traveltime.dat", "rb")
        traveltimes_ = np.frombuffer(traveltime_file_.read(), dtype=np.float32).reshape(
            tuple(config_["traveltime_shape"])
        )
        traveltime_file_.close()
        traveltime_index_file_ = fs.open(f"{bucket}/{folder}/{year}/{jday}/traveltime_index.dat", "rb")
        traveltime_indices_ = np.frombuffer(traveltime_index_file_.read(), dtype=np.int32).reshape(
            tuple(config_["traveltime_shape"])
        )
        traveltime_index_file_.close()
        traveltime_mask_file_ = fs.open(f"{bucket}/{folder}/{year}/{jday}/traveltime_mask.dat", "rb")
        traveltime_masks_ = np.frombuffer(traveltime_mask_file_.read(), dtype=bool).reshape(
            tuple(config_["traveltime_shape"])
        )
        traveltime_mask_file_.close()

        # #### DEBUG ####
        # minlat, maxlat = 35.205, 36.205
        # minlon, maxlon = -118.004, -117.004
        # events_["latitude"] = events_["latitude"].astype(float)
        # events_["longitude"] = events_["longitude"].astype(float)
        # events_ = events_[
        #     (events_["latitude"] > minlat)
        #     & (events_["latitude"] < maxlat)
        #     & (events_["longitude"] > minlon)
        #     & (events_["longitude"] < maxlon)
        # ]
        # ##############

        for k, v in config_.items():
            if k not in config:
                config[k] = v

        events_["latitude"] = events_["latitude"].astype(float)
        events_["longitude"] = events_["longitude"].astype(float)
        events_ = events_[
            (events_["latitude"] > config["minlatitude"])
            & (events_["latitude"] < config["maxlatitude"])
            & (events_["longitude"] > config["minlongitude"])
            & (events_["longitude"] < config["maxlongitude"])
        ]

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

        # picks.append(picks_)
        # events.append(events_)
        # templates.append(templates_)
        # traveltimes.append(traveltimes_)
        # traveltime_indices.append(traveltime_indices_)
        # traveltime_masks.append(traveltime_masks_)
        return (picks_, events_, templates_, traveltimes_, traveltime_indices_, traveltime_masks_)

    ncpu = min(32, mp.cpu_count())
    results = []
    with ThreadPoolExecutor(max_workers=ncpu) as executor:
        futures = [executor.submit(process_day, yj) for yj in year_jday]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
            results.append(future.result())
    picks, events, templates, traveltimes, traveltime_indices, traveltime_masks = zip(*results)

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
    traveltime_index_array = np.memmap(
        traveltime_index_fname, dtype=np.int32, mode="w+", shape=traveltime_indices.shape
    )
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

    # # %%
    # cmd = "python run_cctorch.py"

    # cmd = "python run_hypodd_cc.py"

    # cmd = "python run_growclust_cc.py"

    # cmd = "python plot_catalog.py"

    # picks.to_csv(f"{result_path}/picks_{year:04d}_{jday:03d}.csv", index=False)
    # events.to_csv(f"{result_path}/events_{year:04d}_{jday:03d}.csv", index=False)
    # np.save(f"{result_path}/templates_{year:04d}_{jday:03d}.npy", templates)

    # plot_templates(templates, events, picks)

    # break

    if protocol == "gs":
        print(f"{root_path}/{result_path}/cctorch_picks.csv -> {bucket}/{folder}/cctorch_picks.csv")
        fs.put(f"{root_path}/{result_path}/cctorch_picks.csv", f"{bucket}/{folder}/cctorch_picks.csv")
        print(f"{root_path}/{result_path}/cctorch_events.csv -> {bucket}/{folder}/cctorch_events.csv")
        fs.put(f"{root_path}/{result_path}/cctorch_events.csv", f"{bucket}/{folder}/cctorch_events.csv")
        print(f"{root_path}/{result_path}/cctorch_stations.csv -> {bucket}/{folder}/cctorch_stations.csv")
        fs.put(f"{root_path}/{result_path}/cctorch_stations.csv", f"{bucket}/{folder}/cctorch_stations.csv")
        print(f"{root_path}/{result_path}/config.json -> {bucket}/{folder}/config.json")
        fs.put(f"{root_path}/{result_path}/config.json", f"{bucket}/{folder}/config.json")
        print(f"{root_path}/{result_path}/pairs.txt -> {bucket}/{folder}/pairs.txt")
        fs.put(f"{root_path}/{result_path}/pairs.txt", f"{bucket}/{folder}/pairs.txt")
        # print(f"{root_path}/{result_path}/template.dat -> {bucket}/{folder}/template.dat")
        # fs.put(f"{root_path}/{result_path}/template.dat", f"{bucket}/{folder}/template.dat")
        # print(f"{root_path}/{result_path}/traveltime.dat -> {bucket}/{folder}/traveltime.dat")
        # fs.put(f"{root_path}/{result_path}/traveltime.dat", f"{bucket}/{folder}/traveltime.dat")
        # print(f"{root_path}/{result_path}/traveltime_index.dat -> {bucket}/{folder}/traveltime_index.dat")
        # fs.put(f"{root_path}/{result_path}/traveltime_index.dat", f"{bucket}/{folder}/traveltime_index.dat")
        # print(f"{root_path}/{result_path}/traveltime_mask.dat -> {bucket}/{folder}/traveltime_mask.dat")
        # fs.put(f"{root_path}/{result_path}/traveltime_mask.dat", f"{bucket}/{folder}/traveltime_mask.dat")
        

# %%

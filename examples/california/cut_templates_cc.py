# %%
import argparse
import json
import multiprocessing as mp
import os
import sys
from glob import glob

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from adloc.eikonal2d import calc_traveltime, init_eikonal2d
from pyproj import Proj
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

np.random.seed(42)


# %%
def fillin_missing_picks(picks, events, stations, config):

    # reference_t0 = config["reference_t0"]
    reference_t0 = pd.Timestamp(config["reference_t0"])
    vp_vs_ratio = config["vp_vs_ratio"]
    min_phase_score = config["min_phase_score"]

    pivoted = picks.pivot(index=["event_index", "station_id"], columns="phase_type", values="traveltime")
    pivoted.columns.name = None
    pivoted = pivoted.reset_index()

    pivoted["P_pred"] = np.where(pivoted["P"].isna(), pivoted["S"] / vp_vs_ratio, pivoted["P"])
    pivoted["S_pred"] = np.where(pivoted["S"].isna(), pivoted["P"] * vp_vs_ratio, pivoted["S"])

    pivoted["P_source"] = np.where(pivoted["P"].isna(), "pred", "ml")
    pivoted["S_source"] = np.where(pivoted["S"].isna(), "pred", "ml")

    picks_ps = pd.melt(
        pivoted,
        id_vars=["event_index", "station_id", "P_source", "S_source"],
        value_vars=["P_pred", "S_pred"],
        var_name="phase_type",
        value_name="traveltime",
    )

    picks_ps["phase_type"] = picks_ps["phase_type"].str[0]
    picks_ps["phase_source"] = picks_ps.apply(lambda row: row[f"{row['phase_type']}_source"], axis=1)
    picks_ps.drop(columns=["P_source", "S_source"], inplace=True)

    picks_ps = picks_ps.merge(events[["event_index", "event_timestamp"]], on="event_index")
    # picks_ps = picks_ps.merge(stations[["station_id", "station_term_time"]], on="station_id")
    picks_ps = picks_ps.merge(stations[["station_id", "station_term_time_p", "station_term_time_s"]], on="station_id")
    picks_ps["station_term_time"] = picks_ps.apply(
        lambda row: row[f"station_term_time_{row['phase_type'].lower()}"], axis=1
    )  ## Separate P and S station term
    picks_ps.drop(columns=["station_term_time_p", "station_term_time_s"], inplace=True)
    picks_ps["phase_timestamp"] = picks_ps["event_timestamp"] + picks_ps["traveltime"] + picks_ps["station_term_time"]
    picks_ps["phase_time"] = reference_t0 + pd.to_timedelta(picks_ps["phase_timestamp"], unit="s")
    picks_ps["phase_score"] = min_phase_score

    picks.set_index(["event_index", "station_id", "phase_type"], inplace=True)
    picks_ps.set_index(["event_index", "station_id", "phase_type"], inplace=True)
    picks_ps.update(picks)
    picks_ps.reset_index(inplace=True)

    print(f"Original picks: {len(picks)}, Filled picks: {len(picks_ps)}")
    print(picks_ps.iloc[:10])
    picks = picks_ps

    return picks


# %%
def predict_full_picks(picks, events, stations, config):

    vp_vs_ratio = config["vp_vs_ratio"]
    # reference_t0 = config["reference_t0"]
    reference_t0 = pd.Timestamp(config["reference_t0"])
    eikonal = config["eikonal"]
    min_phase_score = config["min_phase_score"]

    if picks is None:
        event_index = events["event_index"].values
        station_id = stations["station_id"].values
    else:
        event_index = picks["event_index"].unique()
        station_id = picks["station_id"].unique()
    event_index, station_id = np.meshgrid(event_index, station_id, indexing="ij")
    picks_full = pd.DataFrame({"event_index": event_index.flatten(), "station_id": station_id.flatten()})
    p_picks = picks_full.assign(phase_type="P")
    s_picks = picks_full.assign(phase_type="S")
    picks_full = pd.concat([p_picks, s_picks])
    phase_type = picks_full["phase_type"].values
    # picks_full = picks_full.merge(
    #     stations[["station_id", "station_term_time", "x_km", "y_km", "z_km"]], on="station_id"
    # )
    picks_full = picks_full.merge(
        stations[["station_id", "station_term_time_p", "station_term_time_s", "x_km", "y_km", "z_km"]], on="station_id"
    )
    picks_full["station_term_time"] = picks_full.apply(
        lambda row: row[f"station_term_time_{row['phase_type'].lower()}"], axis=1
    )
    picks_full.drop(columns=["station_term_time_p", "station_term_time_s"], inplace=True)
    station_dt = picks_full["station_term_time"].values
    station_locs = picks_full[["x_km", "y_km", "z_km"]].values
    picks_full.rename(columns={"x_km": "station_x_km", "y_km": "station_y_km", "z_km": "station_z_km"}, inplace=True)
    picks_full = picks_full.merge(events[["event_index", "event_timestamp", "x_km", "y_km", "z_km"]], on="event_index")
    event_ot = picks_full["event_timestamp"].values
    event_locs = picks_full[["x_km", "y_km", "z_km"]].values
    picks_full.rename(columns={"x_km": "event_x_km", "y_km": "event_y_km", "z_km": "event_z_km"}, inplace=True)

    traveltime = calc_traveltime(
        event_locs=event_locs,
        station_locs=station_locs,
        phase_type=phase_type,
        eikonal=eikonal,
    )

    picks_full["travel_time"] = traveltime
    picks_full["phase_timestamp"] = event_ot + traveltime + station_dt
    picks_full["phase_time"] = reference_t0 + pd.to_timedelta(picks_full["phase_timestamp"], unit="s")
    picks_full["phase_score"] = min_phase_score
    picks_full["phase_source"] = "pred"
    picks_full["dist_km"] = np.linalg.norm(event_locs - station_locs, axis=-1)

    picks_full.set_index(["event_index", "station_id", "phase_type"], inplace=True)
    if picks is not None:
        picks.set_index(["event_index", "station_id", "phase_type"], inplace=True)
        picks_full.update(picks)
    picks_full.reset_index(inplace=True)

    print(f"Full picks: {len(picks_full)}")
    print(picks_full.iloc[:10])

    picks = picks_full
    return picks


# %%
def extract_template_numpy(
    template_fname,
    traveltime_fname,
    traveltime_index_fname,
    traveltime_mask_fname,
    picks_group,
    events,
    config,
    lock,
):

    # reference_t0 = config["reference_t0"]
    reference_t0 = pd.Timestamp(config["reference_t0"])

    template_array = np.memmap(template_fname, dtype=np.float32, mode="r+", shape=tuple(config["template_shape"]))
    traveltime_array = np.memmap(traveltime_fname, dtype=np.float32, mode="r+", shape=tuple(config["traveltime_shape"]))
    traveltime_index_array = np.memmap(
        traveltime_index_fname, dtype=np.float32, mode="r+", shape=tuple(config["traveltime_shape"])
    )
    traveltime_mask = np.memmap(traveltime_mask_fname, dtype=bool, mode="r+", shape=tuple(config["traveltime_shape"]))

    # template_array = np.zeros(tuple(config["template_shape"]), dtype=np.float32)
    # traveltime_array = np.zeros(tuple(config["traveltime_shape"]), dtype=np.float32)
    # traveltime_index_array = np.zeros(tuple(config["traveltime_shape"]), dtype=np.float32)
    # traveltime_mask = np.zeros(tuple(config["traveltime_shape"]), dtype=bool)

    for picks in picks_group:

        waveforms_dict = {}
        picks = picks.set_index(["idx_eve", "idx_sta", "phase_type"])
        picks_index = list(picks.index.unique())

        ## Cut templates
        for (idx_eve, idx_sta, phase_type), pick in picks.iterrows():

            idx_pick = pick["idx_pick"]
            phase_timestamp = pick["phase_timestamp"]

            event = events.loc[idx_eve]
            ENZ = pick["ENZ"].split(",")

            for c in ENZ:
                if c not in waveforms_dict:
                    try:
                        with fsspec.open(c, "rb", anon=True) as f:
                            stream = obspy.read(f)
                            stream.merge(fill_value="latest")
                            if len(stream) > 1:
                                print(f"More than one trace: {stream}")
                            trace = stream[0]
                            if trace.stats.sampling_rate != config["sampling_rate"]:
                                if trace.stats.sampling_rate % config["sampling_rate"] == 0:
                                    trace.decimate(int(trace.stats.sampling_rate / config["sampling_rate"]))
                                else:
                                    trace.resample(config["sampling_rate"])
                            # trace.detrend("linear")
                            # trace.taper(max_percentage=0.05, type="cosine")
                            trace.filter("bandpass", freqmin=2.0, freqmax=12.0, corners=4, zerophase=True)
                            waveforms_dict[c] = trace
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
                else:
                    trace = waveforms_dict[c]

                ic = config["component_mapping"][trace.stats.channel[-1]]

                trace_starttime = (
                    pd.to_datetime(trace.stats.starttime.datetime, utc=True) - reference_t0
                ).total_seconds()

                begin_time = phase_timestamp - trace_starttime - config[f"time_before_{phase_type.lower()}"]
                end_time = phase_timestamp - trace_starttime + config[f"time_after_{phase_type.lower()}"]

                if phase_type == "P" and ((idx_eve, idx_sta, "S") in picks_index):

                    s_begin_time = (
                        picks.loc[idx_eve, idx_sta, "S"]["phase_timestamp"] - trace_starttime - config[f"time_before_s"]
                    )
                    if config["no_overlapping"]:
                        end_time = min(end_time, s_begin_time)

                begin_time_index = max(0, int(round(begin_time * config["sampling_rate"])))
                end_time_index = max(0, int(round(end_time * config["sampling_rate"])))

                ## define traveltime at the exact data point of event origin time
                traveltime_array[idx_pick, ic, 0] = begin_time_index / config["sampling_rate"] - (
                    event["event_timestamp"] - trace_starttime - config[f"time_before_{phase_type.lower()}"]
                )
                traveltime_index_array[idx_pick, ic, 0] = begin_time_index - int(
                    (event["event_timestamp"] - trace_starttime - config[f"time_before_{phase_type.lower()}"])
                    * config["sampling_rate"]
                )
                traveltime_mask[idx_pick, ic, 0] = True

                trace_data = trace.data[begin_time_index:end_time_index].astype(np.float32)
                template_array[idx_pick, ic, 0, : len(trace_data)] = trace_data  # - np.mean(trace_data)

        if lock is not None:
            with lock:
                template_array.flush()
                traveltime_array.flush()
                traveltime_index_array.flush()
                traveltime_mask.flush()

    return


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
def cut_templates(jdays, root_path, region, config, bucket, protocol, token):

    # %%
    fs = fsspec.filesystem(protocol, token=token)

    # %%
    # data_path = f"{region}/adloc"
    # result_path = f"{region}/cctorch"
    data_path = f"{region}/adloc2"
    result_path = f"{region}/cctorch"

    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    ## TODO: move to config
    sampling_rate = 100.0

    # CC parameters
    time_before_p = 0.3
    time_after_p = 2.5 - time_before_p
    time_before_s = 0.3
    time_after_s = 4.0 - time_before_s
    no_overlapping = True
    # TM parameters
    # time_before_p = 0.3
    # time_after_p = 4.0 - time_before_p
    # time_before_s = 0.3
    # time_after_s = 4.0 - time_before_s
    # no_overlapping = False
    # DEBUG
    # time_before_p = 1.0
    # time_after_p = 5.0 - time_before_p
    # time_before_s = 1.0
    # time_after_s = 5.0 - time_before_s
    # no_overlapping = True

    time_window = max((time_before_p + time_after_p), (time_before_s + time_after_s))
    nt = int(round(time_window * sampling_rate))
    max_epicenter_dist = 200.0
    max_pair_dist = 10
    max_neighbors = 50
    min_cc_score = 0.5
    min_obs = 8
    max_obs = 100
    # component_mapping = {
    #     "123": {"3": 0, "2": 1, "1": 2},
    #     "12Z": {"1": 0, "2": 1, "Z": 2},
    #     "ENU": {"E": 0, "N": 1, "Z": 2},
    #     "ENZ": {"E": 0, "N": 1, "Z": 2},
    # } ## TODO: improve component_mapping
    component_mapping = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}

    config.update(
        {
            "time_before_p": time_before_p,
            "time_after_p": time_after_p,
            "time_before_s": time_before_s,
            "time_after_s": time_after_s,
            "time_window": time_window,
            "no_overlapping": no_overlapping,
            "nt": nt,
            "max_epicenter_dist_km": max_epicenter_dist,
            "max_pair_dist_km": max_pair_dist,
            "max_neighbors": max_neighbors,
            "min_cc_score": min_cc_score,
            "min_obs": min_obs,
            "max_obs": max_obs,
            "sampling_rate": sampling_rate,
            "component_mapping": component_mapping,
        }
    )

    # %%
    lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
    lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")

    xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
    xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
    # zmin, zmax = config["mindepth"], config["maxdepth"]
    zmin = config["mindepth"] if "mindepth" in config else 0.0
    zmax = config["maxdepth"] if "maxdepth" in config else 60.0
    config["xlim_km"] = (xmin, xmax)
    config["ylim_km"] = (ymin, ymax)
    config["zlim_km"] = (zmin, zmax)

    # %% Eikonal for 1D velocity model
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 5.5, 6.7, 7.8]
    # zz = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30]
    # vp = [4.74, 5.01, 5.35, 5.71, 6.07, 6.17, 6.27, 6.34, 6.39, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.3

    if os.path.exists(f"{root_path}/{region}/obspy/velocity.csv"):
        velocity = pd.read_csv(f"{root_path}/{region}/obspy/velocity.csv")
        zz = velocity["z_km"].values
        vp = velocity["vp"].values
        vs = velocity["vs"].values
        h = 0.1

    vel = {"Z": zz, "P": vp, "S": vs}
    eikonal = {
        "vel": vel,
        "h": h,
        "xlim_km": config["xlim_km"],
        "ylim_km": config["ylim_km"],
        "zlim_km": config["zlim_km"],
    }
    eikonal = init_eikonal2d(eikonal)

    for jday in jdays:
        year, jday = jday.split(".")
        year, jday = int(year), int(jday)

        if not os.path.exists(f"{root_path}/{result_path}/{year:04d}"):
            os.makedirs(f"{root_path}/{result_path}/{year:04d}")

        # %%
        # stations = pd.read_csv(f"{root_path}/{data_path}/ransac_stations.csv")
        # stations = pd.read_csv("adloc_stations.csv")
        station_json = f"{region}/network/stations.json"
        if protocol == "file":
            stations = pd.read_json(f"{root_path}/{station_json}", orient="index")
        else:
            with fs.open(f"{bucket}/{station_json}", "r") as fp:
                stations = pd.read_json(fp, orient="index")
        stations["station_id"] = stations.index
        stations.sort_values(by=["latitude", "longitude"], inplace=True)

        # ############### DEBUG ###############
        # "minlatitude": 35.205,
        # "maxlatitude": 36.205,
        # "minlongitude": -118.004,
        # "maxlongitude": -117.004,
        minlat, maxlat = 35.205, 36.205
        minlon, maxlon = -118.004, -117.004
        # ############### DEBUG ###############

        # %%
        # events = pd.read_csv(f"{root_path}/{data_path}/ransac_events.csv", parse_dates=["time"])
        # events = pd.read_csv("adloc_events.csv", parse_dates=["time"])
        if protocol == "file":
            events = pd.read_csv(
                f"{root_path}/{data_path}/{year:04d}/adloc_events_{jday:03d}.csv", parse_dates=["time"]
            )
        else:
            with fs.open(f"{bucket}/{data_path}/{year:04d}/adloc_events_{jday:03d}.csv", "r") as fp:
                events = pd.read_csv(fp, parse_dates=["time"])

        # ############### DEBUG ###############
        # events = events[
        #     (events["latitude"] >= minlat)
        #     & (events["latitude"] <= maxlat)
        #     & (events["longitude"] >= minlon)
        #     & (events["longitude"] <= maxlon)
        # ]
        # plt.figure(figsize=(10, 10))
        # plt.scatter(events["longitude"], events["latitude"], s=1)
        # plt.savefig(f"events_{year:04d}_{jday:03d}.png")
        # ############### DEBUG ###############

        events.rename(columns={"time": "event_time"}, inplace=True)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        reference_t0 = events["event_time"].min()
        events["event_timestamp"] = events["event_time"].apply(lambda x: (x - reference_t0).total_seconds())
        print(f"{len(events) = }")
        print(events.iloc[:5])

        # %%
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
        # picks = pd.read_csv(f"{root_path}/{region}/adloc/ransac_picks.csv")
        # picks = pd.read_csv(f"{root_path}/{data_path}/ransac_picks.csv")
        # picks = pd.read_csv("adloc_picks.csv")
        if protocol == "file":
            picks = pd.read_csv(f"{root_path}/{data_path}/{year:04d}/adloc_picks_{jday:03d}.csv")
        else:
            with fs.open(f"{bucket}/{data_path}/{year:04d}/adloc_picks_{jday:03d}.csv", "r") as fp:
                picks = pd.read_csv(fp)

        # ############### DEBUG ###############
        # picks = picks[(picks["event_index"].isin(events["event_index"]))]
        # ############### DEBUG ###############

        picks = picks[picks["adloc_mask"] == 1]
        picks["phase_time"] = pd.to_datetime(picks["phase_time"], utc=True)
        min_phase_score = picks["phase_score"].min()

        picks = picks.merge(events[["event_index", "event_timestamp"]], on="event_index")
        # picks = picks.merge(stations[["station_id", "station_term_time"]], on="station_id")
        picks = picks.merge(stations[["station_id", "station_term_time_p", "station_term_time_s"]], on="station_id")
        picks["station_term_time"] = picks.apply(
            lambda x: x[f"station_term_time_{x['phase_type'].lower()}"], axis=1
        )  ## Separate P and S station term
        picks.drop(columns=["station_term_time_p", "station_term_time_s"], inplace=True)
        picks["phase_timestamp"] = picks["phase_time"].apply(lambda x: (x - reference_t0).total_seconds())
        picks["traveltime"] = (
            picks["phase_timestamp"] - picks["event_timestamp"] - picks["station_term_time"]
        )  ## Separate P and S station term

        picks = fillin_missing_picks(
            picks,
            events,
            stations,
            config={"reference_t0": reference_t0, "vp_vs_ratio": vp_vs_ratio, "min_phase_score": min_phase_score},
        )
        if "dist_km" not in picks:
            picks = picks.merge(stations[["station_id", "x_km", "y_km", "z_km"]], on="station_id")
            picks.rename(columns={"x_km": "station_x_km", "y_km": "station_y_km", "z_km": "station_z_km"}, inplace=True)
            picks = picks.merge(events[["event_index", "x_km", "y_km", "z_km"]], on="event_index")
            picks.rename(columns={"x_km": "event_x_km", "y_km": "event_y_km", "z_km": "event_z_km"}, inplace=True)
            picks["dist_km"] = np.linalg.norm(
                picks[["event_x_km", "event_y_km", "event_z_km"]].values
                - picks[["station_x_km", "station_y_km", "station_z_km"]].values,
                axis=-1,
            )
            picks.drop(
                columns=["station_x_km", "station_y_km", "station_z_km", "event_x_km", "event_y_km", "event_z_km"],
                inplace=True,
            )

        print(f"{len(picks) = }")
        picks = picks[picks["dist_km"] < config["max_epicenter_dist_km"]]
        print(f"{len(picks) = } with dist_km < {config['max_epicenter_dist_km']} km")
        print(picks.iloc[:5])

        ## filter stations
        stations = stations[stations["station_id"].isin(picks["station_id"])]
        print(f"{len(stations) = }")
        print(stations.iloc[:5])

        # %% Reindex event and station to make it continuous
        stations["idx_sta"] = np.arange(len(stations))
        events["idx_eve"] = np.arange(len(events))
        picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
        picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

        # %%
        # stations.to_csv(f"{root_path}/{result_path}/cctorch_stations.csv", index=False)
        # events.to_csv(f"{root_path}/{result_path}/cctorch_events.csv", index=False)
        # picks.to_csv(f"{root_path}/{result_path}/cctorch_picks.csv", index=False)
        # if not os.path.exists(f"{root_path}/{result_path}/{year:04d}"):
        #     os.makedirs(f"{root_path}/{result_path}/{year:04d}")
        stations.to_csv(f"{root_path}/{result_path}/{year:04d}/cctorch_stations_{jday:03d}.csv", index=False)
        events.to_csv(f"{root_path}/{result_path}/{year:04d}/cctorch_events_{jday:03d}.csv", index=False)
        # picks.to_csv(f"{root_path}/{result_path}/{year:04d}/cctorch_picks_{jday:03d}.csv", index=False)

        # %%
        # pair_fname = f"{root_path}/{result_path}/pairs.txt"
        pair_fname = f"{root_path}/{result_path}/{year:04d}/pairs_{jday:03d}.txt"
        config["pair_file"] = pair_fname

        # %%
        nch = 3  ## [E,N,Z] components
        nt = config["nt"]
        npk = len(picks)
        nev = len(events)
        nst = len(stations)
        print(f"npk: {npk}, nev: {nev}, nch: {nch}, nst: {nst}, nt: {nt}")
        template_shape = (npk, 3, 1, nt)
        traveltime_shape = (npk, 3, 1)
        config["template_shape"] = template_shape
        config["traveltime_shape"] = traveltime_shape

        # template_fname = f"{root_path}/{result_path}/template.dat"
        # traveltime_fname = f"{root_path}/{result_path}/traveltime.dat"
        # traveltime_index_fname = f"{root_path}/{result_path}/traveltime_index.dat"
        # traveltime_mask_fname = f"{root_path}/{result_path}/traveltime_mask.dat"
        template_fname = f"{root_path}/{result_path}/{year:04d}/template_{jday:03d}.dat"
        traveltime_fname = f"{root_path}/{result_path}/{year:04d}/traveltime_{jday:03d}.dat"
        traveltime_index_fname = f"{root_path}/{result_path}/{year:04d}/traveltime_index_{jday:03d}.dat"
        traveltime_mask_fname = f"{root_path}/{result_path}/{year:04d}/traveltime_mask_{jday:03d}.dat"

        config["template_file"] = template_fname
        config["traveltime_file"] = traveltime_fname
        config["traveltime_index_file"] = traveltime_index_fname
        config["traveltime_mask_file"] = traveltime_mask_fname

        template_array = np.memmap(template_fname, dtype=np.float32, mode="w+", shape=template_shape)
        traveltime_array = np.memmap(traveltime_fname, dtype=np.float32, mode="w+", shape=traveltime_shape)
        traveltime_index_array = np.memmap(traveltime_index_fname, dtype=np.float32, mode="w+", shape=traveltime_shape)
        traveltime_mask = np.memmap(traveltime_mask_fname, dtype=bool, mode="w+", shape=traveltime_shape)

        # with open(f"{root_path}/{result_path}/config.json", "w") as f:
        with open(f"{root_path}/{result_path}/{year:04d}/config_{jday:03d}.json", "w") as f:
            json.dump(config, f, indent=4, sort_keys=True)

        # %%
        config["reference_t0"] = reference_t0.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
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
            ]
        ]
        events.set_index("idx_eve", inplace=True)
        stations.set_index("idx_sta", inplace=True)
        picks.sort_values(by=["idx_eve", "idx_sta", "phase_type"], inplace=True)
        picks["idx_pick"] = np.arange(len(picks))

        # picks.to_csv(f"{root_path}/{result_path}/cctorch_picks.csv", index=False)
        picks.to_csv(f"{root_path}/{result_path}/{year:04d}/cctorch_picks_{jday:03d}.csv", index=False)

        ############################# CLOUD #########################################
        # dirs = sorted(glob(f"{root_path}/{region}/waveforms/????/???/??"), reverse=True)
        protocol = "gs"
        bucket = "quakeflow_catalog"
        folder = "SC"
        token_json = "application_default_credentials.json"
        with open(token_json, "r") as fp:
            token = json.load(fp)
        fs = fsspec.filesystem(protocol=protocol, token=token)
        # year = 2019
        mseeds_df = []
        for folder in ["SC", "NC"]:
            with fs.open(f"{bucket}/{folder}/mseed_list/{year}_3c.txt", "r") as f:
                mseeds = f.readlines()
            mseeds = [x.strip("\n") for x in mseeds]
            mseeds = pd.DataFrame(mseeds, columns=["ENZ"])
            if folder == "SC":
                mseeds["fname"] = mseeds["ENZ"].apply(lambda x: x.split("/")[-1])
                mseeds["network"] = mseeds["fname"].apply(lambda x: x[:2])
                mseeds["station"] = mseeds["fname"].apply(lambda x: x[2:7].strip("_"))
                mseeds["instrument"] = mseeds["fname"].apply(lambda x: x[7:9])
                mseeds["location"] = mseeds["fname"].apply(lambda x: x[10:12].strip("_"))
                mseeds["year"] = mseeds["fname"].apply(lambda x: x[13:17])
                mseeds["jday"] = mseeds["fname"].apply(lambda x: x[17:20])
            if folder == "NC":
                mseeds["fname"] = mseeds["ENZ"].apply(lambda x: x.split("/")[-1])
                mseeds["network"] = mseeds["fname"].apply(lambda x: x.split(".")[1])
                mseeds["station"] = mseeds["fname"].apply(lambda x: x.split(".")[0])
                mseeds["instrument"] = mseeds["fname"].apply(lambda x: x.split(".")[2][:-1])
                mseeds["location"] = mseeds["fname"].apply(lambda x: x.split(".")[3])
                mseeds["year"] = mseeds["fname"].apply(lambda x: x.split(".")[5])
                mseeds["jday"] = mseeds["fname"].apply(lambda x: x.split(".")[6])
            mseeds_df.append(mseeds)
        mseeds_df = pd.concat(mseeds_df)
        print(mseeds_df.head())
        print(mseeds_df.tail())

        picks["network"] = picks["station_id"].apply(lambda x: x.split(".")[0])
        picks["station"] = picks["station_id"].apply(lambda x: x.split(".")[1])
        picks["location"] = picks["station_id"].apply(lambda x: x.split(".")[2])
        picks["instrument"] = picks["station_id"].apply(lambda x: x.split(".")[3])
        picks["year"] = picks["phase_time"].dt.strftime("%Y")
        picks["jday"] = picks["phase_time"].dt.strftime("%j")

        mseeds_df = mseeds_df[(mseeds_df["year"].astype(int) == year) & (mseeds_df["jday"].astype(int) == jday)]
        picks = picks[(picks["year"].astype(int) == year) & (picks["jday"].astype(int) == jday)]

        picks = picks.merge(mseeds_df, on=["network", "station", "location", "instrument", "year", "jday"])
        picks.drop(columns=["fname", "station_id", "network", "location", "instrument", "year", "jday"], inplace=True)

        if len(picks) == 0:
            print(f"No picks found for {year:04d}/{jday:03d}")
            continue

        # ####
        # out = picks.drop(columns=["ENZ"])
        # out.to_csv(f"{root_path}/{result_path}/{year:04d}/cctorch_picks_{jday:03d}.csv", index=False)
        # events.to_csv(f"{root_path}/{result_path}/{year:04d}/cctorch_events_{jday:03d}.csv", index=False)
        # stations.to_csv(f"{root_path}/{result_path}/{year:04d}/cctorch_stations_{jday:03d}.csv", index=False)

        ###########################################################################

        picks_group = picks.copy()
        picks_group = picks_group.groupby("ENZ")

        ############################################################

        # ncpu = min(32, mp.cpu_count())
        ncpu = 32
        # nsplit = min(ncpu * 8, len(picks_group))
        nsplit = len(picks_group)
        print(f"Using {ncpu} cores")

        pbar = tqdm(total=nsplit, desc="Cutting templates")
        ctx = mp.get_context("spawn")

        with ctx.Manager() as manager:
            lock = manager.Lock()
            with ctx.Pool(ncpu) as pool:
                jobs = []

                group_chunk = np.array_split(list(picks_group.groups.keys()), nsplit)
                picks_group_chunk = [[picks_group.get_group(g) for g in group] for group in group_chunk]

                for picks_group in picks_group_chunk:

                    job = pool.apply_async(
                        extract_template_numpy,
                        (
                            template_fname,
                            traveltime_fname,
                            traveltime_index_fname,
                            traveltime_mask_fname,
                            picks_group,
                            events,
                            config,
                            lock,
                        ),
                        callback=lambda x: pbar.update(),
                    )
                    jobs.append(job)

                pool.close()
                pool.join()
                for job in jobs:
                    out = job.get()
                    if out is not None:
                        print(out)

        pbar.close()

        # pairs = generate_pairs(
        #     picks,
        #     events,
        #     stations,
        #     max_pair_dist=config["max_pair_dist_km"],
        #     fname=config["pair_file"],
        # )

        if protocol == "gs":
            fs.put(
                f"{root_path}/{result_path}/{year:04d}/cctorch_picks_{jday:03d}.csv",
                f"{bucket}/{result_path}/{year:04d}/cctorch_picks_{jday:03d}.csv",
            )
            fs.put(
                f"{root_path}/{result_path}/{year:04d}/cctorch_events_{jday:03d}.csv",
                f"{bucket}/{result_path}/{year:04d}/cctorch_events_{jday:03d}.csv",
            )
            # fs.put(
            #     f"{root_path}/{result_path}/{year:04d}/cctorch_stations_{jday:03d}.csv",
            #     f"{bucket}/{result_path}/{year:04d}/cctorch_stations_{jday:03d}.csv",
            # )
            fs.put(
                f"{root_path}/{result_path}/{year:04d}/config_{jday:03d}.json",
                f"{bucket}/{result_path}/{year:04d}/config_{jday:03d}.json",
            )
            fs.put(
                f"{root_path}/{result_path}/{year:04d}/template_{jday:03d}.dat",
                f"{bucket}/{result_path}/{year:04d}/template_{jday:03d}.dat",
            )
            print(
                f"{root_path}/{result_path}/{year:04d}/template_{jday:03d}.npy -> {bucket}/{result_path}/{year:04d}/template_{jday:03d}.npy"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gamma on NCEDC/SCEDC data")
    parser.add_argument("--num_nodes", type=int, default=1)
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

    # # %%
    # args = parse_args()
    # root_path = args.root_path
    # region = args.region

    # with open(f"{root_path}/{region}/config.json", "r") as fp:
    #     config = json.load(fp)

    # config.update(config["cctorch"])

    # %%
    args = parse_args()
    region = args.region
    root_path = args.root_path
    bucket = args.bucket
    num_nodes = args.num_nodes
    node_rank = args.node_rank
    year = args.year

    # %%
    # with fs.open(f"{bucket}/{region}/config.json", "r") as fp:
    #     config = json.load(fp)
    with open("config.json", "r") as fp:
        config = json.load(fp)
    config["world_size"] = num_nodes
    config.update(vars(args))

    # %%
    print(json.dumps(config, indent=4, sort_keys=True))

    # cut_templates(root_path, region, config)

    # %%
    # events = []
    # picks = []
    # jobs = []
    # ctx = mp.get_context("spawn")
    # ncpu = min(32, mp.cpu_count())
    # years = [2023]
    # num_days = sum([366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365 for year in years])
    # pbar = tqdm(total=num_days, desc="Loading data")
    # with ctx.Pool(processes=ncpu) as pool:
    #     for year in years:
    #         num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
    #         for jday in range(1, num_jday + 1):
    #             cut_templates(year, jday, root_path, region, config, bucket, protocol, token)
    #             # job = pool.apply_async(
    #             #     cut_templates,
    #             #     args=(year, jday, root_path, region, config, bucket, protocol, token),
    #             #     callback=lambda x: pbar.update(),
    #             # )
    #             # jobs.append(job)

    #     pool.close()
    #     pool.join()

    # pbar.close()

    # %%
    # years = [2023]
    # years = [2019]
    # jdays = []
    # for year in years:
    #     num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
    #     jdays.extend([f"{year}.{i:03d}" for i in range(1, num_jday + 1)])

    num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
    jdays = [f"{year}.{i:03d}" for i in range(1, num_jday + 1)]

    jdays = np.array_split(jdays, num_nodes)[node_rank]
    # jdays = ["2019.185"]
    print(f"{node_rank}/{num_nodes}: {jdays[0] = }, {jdays[-1] = }")

    cut_templates(jdays, root_path, region, config, bucket, protocol, token)

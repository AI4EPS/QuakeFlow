# %%
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime, timedelta, timezone
from glob import glob

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

    reference_t0 = config["reference_t0"]
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
    picks_ps = picks_ps.merge(stations[["station_id", "station_term_time"]], on="station_id")
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
    reference_t0 = config["reference_t0"]
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
    picks_full = picks_full.merge(
        stations[["station_id", "station_term_time", "x_km", "y_km", "z_km"]], on="station_id"
    )
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
    mseed_path,
    events,
    picks,
    stations,
    config,
    lock,
):

    reference_t0 = config["reference_t0"]

    template_array = np.memmap(template_fname, dtype=np.float32, mode="r+", shape=tuple(config["template_shape"]))
    traveltime_array = np.memmap(traveltime_fname, dtype=np.float32, mode="r+", shape=tuple(config["traveltime_shape"]))
    traveltime_index_array = np.memmap(
        traveltime_index_fname, dtype=np.float32, mode="r+", shape=tuple(config["traveltime_shape"])
    )
    traveltime_mask = np.memmap(traveltime_mask_fname, dtype=bool, mode="r+", shape=tuple(config["traveltime_shape"]))

    ## Load waveforms
    waveforms_dict = {}
    for i, station in stations.iterrows():
        station_id = station["station_id"]
        for c in station["component"]:
            mseed_name = f"{mseed_path}/{station_id.lstrip('.').rstrip('.')}.{c}.sac"
            if "WJMF" in station_id:
                mseed_name = f"{mseed_path}/{station_id.lstrip('.').rstrip('.')}.{c}B.sac"  ## FIXME
            if os.path.exists(mseed_name):
                try:
                    stream = obspy.read(mseed_name)
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
                    waveforms_dict[f"{station_id}{c}"] = trace
                except Exception as e:
                    print(e)
                    continue

    ## Cut templates
    for (idx_eve, idx_sta, phase_type), pick in picks.iterrows():

        idx_pick = pick["idx_pick"]
        phase_timestamp = pick["phase_timestamp"]

        station = stations.loc[idx_sta]
        station_id = station["station_id"]
        event = events.loc[idx_eve]

        for c in station["component"]:
            ic = config["component_mapping"][c]  # 012 for P, 345 for S

            if f"{station_id}{c}" in waveforms_dict:
                trace = waveforms_dict[f"{station_id}{c}"]
                trace_starttime = (
                    pd.to_datetime(trace.stats.starttime.datetime, utc=True) - reference_t0
                ).total_seconds()
            else:
                continue

            begin_time = phase_timestamp - trace_starttime - config[f"time_before_{phase_type.lower()}"]
            end_time = phase_timestamp - trace_starttime + config[f"time_after_{phase_type.lower()}"]

            if phase_type == "P" and ((idx_eve, idx_sta, "S") in picks.index):
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
            template_array[idx_pick, ic, 0, : len(trace_data)] = trace_data

    if lock is not None:
        with lock:
            template_array.flush()
            traveltime_array.flush()
            traveltime_index_array.flush()
            traveltime_mask.flush()

    return mseed_path


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
def cut_templates(root_path, region, config):

    # %%
    data_path = f"{region}/adloc"
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

    time_window = max((time_before_p + time_after_p), (time_before_s + time_after_s))
    nt = int(round(time_window * sampling_rate))
    max_epicenter_dist = 200.0
    max_pair_dist = 10
    max_neighbors = 50
    min_cc_score = 0.5
    min_obs = 8
    max_obs = 100
    component_mapping = {
        "123": {"3": 0, "2": 1, "1": 2},
        "12Z": {"1": 0, "2": 1, "Z": 2},
        "ENU": {"E": 0, "N": 1, "Z": 2},
        "ENZ": {"E": 0, "N": 1, "Z": 2},
    }
    component_mapping = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2, "U": 2}

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
    stations = pd.read_csv(f"{root_path}/{data_path}/ransac_stations.csv")
    stations.sort_values(by=["latitude", "longitude"], inplace=True)
    stations["network"] = stations["station_id"].apply(lambda x: x.split(".")[1])  ## Hard-coded
    stations["station"] = stations["station_id"].apply(lambda x: x.split(".")[2])  ## Hard-coded
    stations["location"] = ""  ## Hard-coded
    stations["instrument"] = ""  ## Hard-coded
    stations["component"] = "ENU"  ## Hard-coded
    stations.fillna("", inplace=True)
    print(f"{len(stations) = }")
    print(stations.iloc[:5])

    # %%
    events = pd.read_csv(f"{root_path}/{data_path}/ransac_events.csv", parse_dates=["time"])
    events.rename(columns={"time": "event_time"}, inplace=True)
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
    reference_t0 = events["event_time"].min()
    events["event_timestamp"] = events["event_time"].apply(lambda x: (x - reference_t0).total_seconds())
    print(f"{len(events) = }")
    print(events.iloc[:5])

    # %%
    lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
    lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")

    # %%
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z_km"] = stations["elevation_m"].apply(lambda x: -x / 1e3)

    events[["x_km", "y_km"]] = events.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events["z_km"] = events["depth_km"]

    xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
    xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
    zmin, zmax = config["mindepth"], config["maxdepth"]
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
    vel = {"Z": zz, "P": vp, "S": vs}
    eikonal = {
        "vel": vel,
        "h": h,
        "xlim_km": config["xlim_km"],
        "ylim_km": config["ylim_km"],
        "zlim_km": config["zlim_km"],
    }
    eikonal = init_eikonal2d(eikonal)

    # %%
    picks = pd.read_csv(f"{root_path}/{region}/adloc/ransac_picks.csv")

    # events = events[events["event_time"] >= pd.to_datetime("2024-08-01", utc=True)]
    # picks = picks[picks["event_index"].isin(events["event_index"])]

    picks = picks[picks["adloc_mask"] == 1]
    picks["phase_time"] = pd.to_datetime(picks["phase_time"], utc=True)
    min_phase_score = picks["phase_score"].min()
    picks["phase_type"] = picks["phase_type"].apply(lambda x: "P" if x == 0 else "S")  ## FIXME

    picks = picks.merge(events[["event_index", "event_timestamp"]], on="event_index")
    picks = picks.merge(stations[["station_id", "station_term_time"]], on="station_id")
    picks["phase_timestamp"] = picks["phase_time"].apply(lambda x: (x - reference_t0).total_seconds())
    picks["traveltime"] = picks["phase_timestamp"] - picks["event_timestamp"] - picks["station_term_time"]

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

    # %% Reindex event and station to make it continuous
    stations["idx_sta"] = np.arange(len(stations))
    events["idx_eve"] = np.arange(len(events))
    picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    # %%
    stations.to_csv(f"{root_path}/{result_path}/cctorch_stations.csv", index=False)
    events.to_csv(f"{root_path}/{result_path}/cctorch_events.csv", index=False)
    picks.to_csv(f"{root_path}/{result_path}/cctorch_picks.csv", index=False)

    # %%
    pair_fname = f"{root_path}/{result_path}/pairs.txt"
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

    template_fname = f"{root_path}/{result_path}/template.dat"
    traveltime_fname = f"{root_path}/{result_path}/traveltime.dat"
    traveltime_index_fname = f"{root_path}/{result_path}/traveltime_index.dat"
    traveltime_mask_fname = f"{root_path}/{result_path}/traveltime_mask.dat"
    config["template_file"] = template_fname
    config["traveltime_file"] = traveltime_fname
    config["traveltime_index_file"] = traveltime_index_fname
    config["traveltime_mask_file"] = traveltime_mask_fname

    template_array = np.memmap(template_fname, dtype=np.float32, mode="w+", shape=template_shape)
    traveltime_array = np.memmap(traveltime_fname, dtype=np.float32, mode="w+", shape=traveltime_shape)
    traveltime_index_array = np.memmap(traveltime_index_fname, dtype=np.float32, mode="w+", shape=traveltime_shape)
    traveltime_mask = np.memmap(traveltime_mask_fname, dtype=bool, mode="w+", shape=traveltime_shape)

    with open(f"{root_path}/{result_path}/config.json", "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    # %%
    config["reference_t0"] = reference_t0
    events = events[["idx_eve", "x_km", "y_km", "z_km", "event_index", "event_time", "event_timestamp"]]
    stations = stations[["idx_sta", "x_km", "y_km", "z_km", "station_id", "component", "network", "station"]]
    picks = picks[["idx_eve", "idx_sta", "phase_type", "phase_score", "phase_time", "phase_timestamp", "phase_source"]]
    events.set_index("idx_eve", inplace=True)
    stations.set_index("idx_sta", inplace=True)
    picks.sort_values(by=["idx_eve", "idx_sta", "phase_type"], inplace=True)
    picks["idx_pick"] = np.arange(len(picks))

    picks.to_csv(f"{root_path}/{result_path}/cctorch_picks.csv", index=False)

    # dirs = sorted(glob(f"{root_path}/{region}/waveforms/????/???/??"))
    dirs = sorted(glob(f"{root_path}/{region}/waveforms/????-???/??"), reverse=True)
    # # # # 08/01 to jday
    # dirs = sorted(glob(f"{root_path}/{region}/waveforms/2024-???/??"))
    # jday = pd.to_datetime("2024-08-01", utc=True).strftime("%j")
    # dirs = [d for d in dirs if d.split("/")[-2].split("-")[-1] >= jday]  ## FIXME

    ncpu = min(32, mp.cpu_count())
    print(f"Using {ncpu} cores")

    pbar = tqdm(total=len(dirs), desc="Cutting templates")

    def pbar_update(x):
        """
        x: the return value of extract_template_numpy
        """
        pbar.update()
        pbar.set_description(f"Cutting templates: {'/'.join(x.split('/')[-3:])}")

    ctx = mp.get_context("spawn")
    picks_group = picks.copy()
    picks_group["year_jday_hour"] = picks_group["phase_time"].dt.strftime("%Y-%jT%H")
    picks_group = picks_group.groupby("year_jday_hour")

    with ctx.Manager() as manager:
        lock = manager.Lock()
        with ctx.Pool(ncpu) as pool:
            jobs = []
            for d in dirs:

                tmp = d.split("/")
                year_jday, hour = tmp[-2], tmp[-1]
                year, jday = year_jday.split("-")
                # begin_time = np.datetime64(datetime.strptime(f"{year}-{jday}T{hour}", "%Y-%jT%H"))
                # end_time = begin_time + np.timedelta64(1, "h")

                # picks_ = picks[(phase_time >= begin_time) & (phase_time < end_time)]
                # picks_ = (
                #     picks_group.get_group(f"{year}-{jday}T{hour}")
                #     if f"{year}-{jday}T{hour}" in picks_group.groups
                #     else pd.DataFrame()
                # )
                if f"{year}-{jday}T{hour}" not in picks_group.groups:
                    pbar_update(d)
                    continue
                picks_ = picks_group.get_group(f"{year}-{jday}T{hour}")
                events_ = events.loc[picks_["idx_eve"].unique()]  # index is idx_eve
                picks_ = picks_.set_index(["idx_eve", "idx_sta", "phase_type"])

                job = pool.apply_async(
                    extract_template_numpy,
                    (
                        template_fname,
                        traveltime_fname,
                        traveltime_index_fname,
                        traveltime_mask_fname,
                        d,
                        events_,
                        picks_,
                        stations,
                        config,
                        lock,
                    ),
                    callback=pbar_update,
                )
                jobs.append(job)
            pool.close()
            pool.join()
            # for job in jobs:
            #     out = job.get()
            #     if out is not None:
            #         print(out)

    pbar.close()

    pairs = generate_pairs(
        picks,
        events,
        stations,
        max_pair_dist=config["max_pair_dist_km"],
        fname=config["pair_file"],
    )


# %%
if __name__ == "__main__":

    # %%
    root_path = "local"
    region = "hinet"

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    # config.update(config["cctorch"])
    config["mindepth"] = 0.0
    config["maxdepth"] = 30.0

    # %%
    print(json.dumps(config, indent=4, sort_keys=True))

    cut_templates(root_path, region, config)

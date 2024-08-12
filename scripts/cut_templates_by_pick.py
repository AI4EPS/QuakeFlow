# %%
import json
import multiprocessing as mp
import os
import sys

from datetime import timezone
from glob import glob

import numpy as np
import obspy
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from adloc.eikonal2d import init_eikonal2d, calc_traveltime
from pyproj import Proj
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from obspy.signal.cross_correlation import correlate
from scipy.interpolate import interp1d

np.random.seed(42)


# %%
def fillin_missing_picks(picks, events, stations, config):

    reference_t0 = config["reference_t0"]
    vp_vs_ratio = config["vp_vs_ratio"]
    min_phase_score = config["min_phase_score"]

    ############################################ Fillin missing P and S picks ############################################
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

    ############################################ Fillin missing P and S picks ############################################
    return picks


# %%
def predict_full_picks(picks, events, stations, config):

    vp_vs_ratio = config["vp_vs_ratio"]
    reference_t0 = config["reference_t0"]
    eikonal = config["eikonal"]
    min_phase_score = config["min_phase_score"]

    ############################################ Generate full picks ############################################
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
    ############################################ Generate full picks ############################################
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

    tmp = mseed_path.split("/")
    year, jday, hour = tmp[-3], tmp[-2], tmp[-1]
    begin_time = pd.to_datetime(f"{year}-{jday}T{hour}", utc=True, format="%Y-%jT%H")
    # TODO: make timedelta a parameter
    end_time = begin_time + pd.Timedelta(hours=1)

    picks = picks[(picks["phase_time"] >= begin_time) & (picks["phase_time"] < end_time)]
    picks = picks.set_index(["idx_eve", "idx_sta", "phase_type"])

    if len(picks) == 0:
        # print(f"No events in {mseed_path}")
        return mseed_path

    ## Load waveforms
    waveforms_dict = {}
    for i, station in stations.iterrows():
        station_id = station["station_id"]
        for c in station["component"]:
            if os.path.exists(f"{mseed_path}/{station_id}{c}.mseed"):
                try:
                    stream = obspy.read(f"{mseed_path}/{station_id}{c}.mseed")
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

                # print(trace.data)
                # raise

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
def generate_pairs(picks, events, stations, max_pair_dist=10, max_neighbors=50, fname="event_pairs.txt"):
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
        # event_pairs.extend([[i, j] for j in neighs if i < j])
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
def correlate_traces(trace1, trace2, chan_inds, max_shift=85, min_cc=0.6, delta_tt=None):
    # init value of best_sub_shift doesn't matter because we are checking the weight
    best_weight = -1.0
    best_sub_shift = -1.0
    best_i = 0
    for i in chan_inds:
        fcc = np.abs(correlate(trace1[i], trace2[i], max_shift, normalize=None))
        # height has to be small enough to get the second peak (if any)
        peak_inds, _ = find_peaks(fcc, height=0.1)
        # need at least two peaks to compute weight
        if len(peak_inds) < 2:
            continue
        # get the two best peaks
        inds = np.argsort(fcc[peak_inds])[-2:][::-1]
        top_ind = peak_inds[inds[0]]
        # continue if the peak is at/near the edge
        if not 2 < top_ind < len(fcc) - 2:
            continue
        top_cc = fcc[top_ind]
        # cross-correlation coefficient is too low
        if top_cc < min_cc:
            continue
        sec_ind = peak_inds[inds[1]]
        sec_cc = fcc[sec_ind]
        shift = top_ind - max_shift
        # get subsample rate precision
        x = np.array([-1, 0, 1]) + shift
        y = np.array(fcc[top_ind - 1 : top_ind + 2])
        f = interp1d(x, y, kind="quadratic")
        xnew = np.arange(-0.999, 1.0, 0.001) + shift
        ynew = f(xnew)
        sub_ind = np.argmax(ynew)
        sub_shift = xnew[sub_ind]
        # compute weight
        diff = top_cc - sec_cc
        weight = (0.1 + 3.0 * diff) * top_cc**2
        if weight > best_weight:
            best_weight = weight
            best_sub_shift = sub_shift + delta_tt[i]
            best_i = i

    # if best_weight > -1.0:
    #     print(f"{sub_shift = :.3f}, {weight = :.3f}, {top_cc = :.3f}, {diff = :.3f}{delta_tt[best_i] = :.3f}")

    return best_weight, best_sub_shift


# %%
def cut_templates(root_path, region, config):

    # %%
    root_path = "local"
    region = "Ridgecrest"
    result_path = f"cctorch"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    # config["time_before_p"] = 0.3
    # config["time_after_p"] = 2.5 - config["time_before_p"]
    # config["time_before_s"] = 0.3
    # config["time_after_s"] = 4.0 - config["time_before_s"]
    # config["time_window"] = max(
    #     (config["time_before_p"] + config["time_after_p"]), (config["time_before_s"] + config["time_after_s"])
    # )
    # config["nt"] = int(round(config["time_window"] * config["sampling_rate"]))
    # config["max_epicenter_dist"] = 200.0
    time_before_p = 0.3
    time_after_p = 2.5 - time_before_p
    time_before_s = 0.3
    time_after_s = 4.0 - time_before_s
    time_window = max((time_before_p + time_after_p), (time_before_s + time_after_s))
    nt = int(round(time_window * config["sampling_rate"]))
    max_epicenter_dist = 200.0
    max_pair_dist = 10
    max_neighbors = 50
    min_cc_score = 0.5
    min_obs = 8
    max_obs = 100
    sampling_rate = 100.0

    config["cctorch"].update(
        {
            "time_before_p": time_before_p,
            "time_after_p": time_after_p,
            "time_before_s": time_before_s,
            "time_after_s": time_after_s,
            "time_window": time_window,
            "nt": nt,
            "max_epicenter_dist": max_epicenter_dist,
            "max_pair_dist": max_pair_dist,
            "max_neighbors": max_neighbors,
            "min_cc_score": min_cc_score,
            "min_obs": min_obs,
            "max_obs": max_obs,
            "sampling_rate": sampling_rate,
        }
    )

    # %%
    stations = pd.read_csv(f"{root_path}/{region}/adloc/ransac_stations_sst.csv")
    stations = stations[stations["network"] == "CI"]
    stations.sort_values(by=["latitude", "longitude"], inplace=True)
    # stations = stations[stations["station_id"].isin(picks["station_id"].unique())]
    print(f"{len(stations) = }")
    print(stations.iloc[:5])

    # %%
    # events = pd.read_csv(f"{root_path}/{region}/adloc/ransac_events_sst.csv", parse_dates=["time"])
    events = pd.read_csv(f"tests/events.csv")
    events.rename(columns={"time": "event_time"}, inplace=True)
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
    reference_t0 = events["event_time"].min()
    events["event_timestamp"] = events["event_time"].apply(lambda x: (x - reference_t0).total_seconds())
    print(f"{len(events) = }")
    print(events.iloc[:5])

    plt.figure()
    plt.scatter(events["longitude"], events["latitude"], s=2)
    # plt.plot(stations["longitude"], stations["latitude"], "v")
    # plt.plot(picks["longitude"], picks["latitude"], "x")
    plt.title(f"{len(events)} events")
    plt.savefig("debug_events.png")
    # raise

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
    # zz = [0.0, 5.5, 16.0, 32.0]
    # vp = [5.5, 5.5, 6.7, 7.8]
    zz = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30]
    vp = [4.74, 5.01, 5.35, 5.71, 6.07, 6.17, 6.27, 6.34, 6.39, 7.8]
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
    picks = pd.read_csv(f"{root_path}/{region}/adloc/ransac_picks_sst.csv")
    picks = picks[picks["adloc_mask"] == 1]
    picks["phase_time"] = pd.to_datetime(picks["phase_time"], utc=True)
    min_phase_score = picks["phase_score"].min()

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
    picks = picks[picks["dist_km"] < config["cctorch"]["max_epicenter_dist"]]
    print(f"{len(picks) = } with dist_km < {config['cctorch']['max_epicenter_dist']} km")

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
    config["cctorch"]["pair_file"] = pair_fname

    # %%
    nch = 3  ## [E,N,Z] components
    nt = config["cctorch"]["nt"]
    npk = len(picks)
    nev = len(events)
    nst = len(stations)
    print(f"npk: {npk}, nev: {nev}, nch: {nch}, nst: {nst}, nt: {nt}")
    template_shape = (npk, 3, 1, nt)
    traveltime_shape = (npk, 3, 1)
    config["cctorch"]["template_shape"] = template_shape
    config["cctorch"]["traveltime_shape"] = traveltime_shape

    template_fname = f"{root_path}/{result_path}/template.dat"
    traveltime_fname = f"{root_path}/{result_path}/traveltime.dat"
    traveltime_index_fname = f"{root_path}/{result_path}/traveltime_index.dat"
    traveltime_mask_fname = f"{root_path}/{result_path}/traveltime_mask.dat"
    config["cctorch"]["template_file"] = template_fname
    config["cctorch"]["traveltime_file"] = traveltime_fname
    config["cctorch"]["traveltime_index_file"] = traveltime_index_fname
    config["cctorch"]["traveltime_mask_file"] = traveltime_mask_fname

    template_array = np.memmap(template_fname, dtype=np.float32, mode="w+", shape=template_shape)
    traveltime_array = np.memmap(traveltime_fname, dtype=np.float32, mode="w+", shape=traveltime_shape)
    traveltime_index_array = np.memmap(traveltime_index_fname, dtype=np.float32, mode="w+", shape=traveltime_shape)
    traveltime_mask = np.memmap(traveltime_mask_fname, dtype=bool, mode="w+", shape=traveltime_shape)

    with open(f"{root_path}/{result_path}/config.json", "w") as f:
        json.dump(config["cctorch"], f, indent=4, sort_keys=True)

    # %%
    events = events[["idx_eve", "x_km", "y_km", "z_km", "event_index", "event_time", "event_timestamp"]]
    stations = stations[["idx_sta", "x_km", "y_km", "z_km", "station_id", "component", "network", "station"]]
    picks = picks[["idx_eve", "idx_sta", "phase_type", "phase_score", "phase_time", "phase_timestamp", "phase_source"]]
    events.set_index("idx_eve", inplace=True)
    stations.set_index("idx_sta", inplace=True)
    picks.sort_values(by=["idx_eve", "idx_sta", "phase_type"], inplace=True)
    picks["idx_pick"] = np.arange(len(picks))
    config["cctorch"]["reference_t0"] = reference_t0

    dirs = sorted(glob(f"{root_path}/{region}/waveforms/????/???/??"))
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
    with ctx.Manager() as manager:
        lock = manager.Lock()
        with ctx.Pool(ncpu) as pool:
            jobs = []
            for d in dirs:
                job = pool.apply_async(
                    extract_template_numpy,
                    (
                        template_fname,
                        traveltime_fname,
                        traveltime_index_fname,
                        traveltime_mask_fname,
                        d,
                        events,
                        picks,
                        stations,
                        config["cctorch"],
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
        max_pair_dist=config["cctorch"]["min_pair_dist_km"],
        fname=config["cctorch"]["pair_file"],
    )

    # %%
    # i = 0
    # normalize2d = lambda x: (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6) / 6
    # for (idx_eve, phase_type), picks_ in picks.groupby(["idx_eve", "phase_type"]):
    #     idx_pick = picks_["idx_pick"].values
    #     ic = 2
    #     data = normalize2d(template_array[idx_pick, ic, 0, :])
    #     vmax = np.abs(data).max()
    #     plt.figure()
    #     plt.imshow(data, aspect="auto", cmap="bwr", interpolation="none", vmin=-vmax, vmax=vmax)
    #     plt.colorbar()
    #     plt.savefig("debug_template.png")
    #     plt.show()
    #     i += 1
    #     if i > 10:
    #         break

    # # %%
    # neigh = NearestNeighbors(radius=config["cctorch"]["max_pair_dist"], n_jobs=-1)
    # neigh.fit(events[["x_km", "y_km", "z_km"]].values)
    # pairs = set()
    # neigh_ind = neigh.radius_neighbors(sort_results=True)[1]

    # # idx_sta_dict = picks.groupby("idx_eve")["idx_sta"].apply(set).to_dict()
    # for i, neighs in enumerate(tqdm(neigh_ind, desc="Generating pairs")):
    #     # n = 0
    #     for j in neighs[: config["cctorch"]["max_neighbors"]]:
    #         # if i not in idx_sta_dict or j not in idx_sta_dict:
    #         #     continue
    #         # if len(idx_sta_dict[i] & idx_sta_dict[j]) < MIN_LINKS:
    #         #     continue

    #         if i > j:
    #             pairs.add((j, i))
    #         else:
    #             pairs.add((i, j))

    #         # n += 1
    #         # if n >= MAX_NEIGHBORS:
    #         #     break

    # pairs = list(pairs)
    # print(f"{len(pairs) = } {len(events) = }")

    # # raise

    # %%
    normalize2d = lambda x: (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6) / 6
    cc_array = []
    dt_array = []
    max_dt = {"P": 0.5, "S": 0.85}

    picks.to_csv(f"{root_path}/{result_path}/cctorch_picks.csv", index=False)
    # picks.set_index(["idx_eve", "idx_sta", "phase_type"], inplace=True)
    picks.set_index(["idx_pick"], inplace=True)
    # idx_pick_dict = picks["idx_pick"].to_dict()  ## much faster than using .loc
    # idx_pick_dict = picks.index.to_dict()

    # for ii, (i, j) in enumerate(tqdm(pairs)):
    #     for idx_sta in stations.index:
    #         for phase_type in ["P", "S"]:

    #             # if (i, idx_sta, phase_type) not in picks.index or (j, idx_sta, phase_type) not in picks.index:
    #             #     continue
    #             # tt1 = traveltime_array[picks.loc[(i, idx_sta, phase_type), "idx_pick"], :, 0]
    #             # tt2 = traveltime_array[picks.loc[(j, idx_sta, phase_type), "idx_pick"], :, 0]

    #             if (i, idx_sta, phase_type) not in idx_pick_dict or (j, idx_sta, phase_type) not in idx_pick_dict:
    #                 continue

    idx_eve_dict = picks["idx_eve"].to_dict()
    idx_sta_dict = picks["idx_sta"].to_dict()
    phase_type_dict = picks["phase_type"].to_dict()

    # for idx_pick, jdx_pick in tqdm(pairs, desc="Correlating traces"):
    for iii, (idx_pick, jdx_pick) in enumerate(tqdm(pairs, desc="Correlating traces")):
        # idx_eve1 = picks.loc[idx_pick, "idx_eve"]
        # idx_eve2 = picks.loc[jdx_pick, "idx_eve"]
        # idx_sta = picks.loc[idx_pick, "idx_sta"]
        # phase_type = picks.loc[idx_pick, "phase_type"]
        idx_eve1 = idx_eve_dict[idx_pick]
        idx_eve2 = idx_eve_dict[jdx_pick]
        idx_sta = idx_sta_dict[idx_pick]
        phase_type = phase_type_dict[idx_pick]

        # idx_pick = idx_pick_dict[(i, idx_sta, phase_type)]
        data1 = normalize2d(template_array[idx_pick, :, 0, :])
        # jdx_pick = idx_pick_dict[(j, idx_sta, phase_type)]
        data2 = normalize2d(template_array[jdx_pick, :, 0, :])

        # print(f"{idx_pick = }, {jdx_pick = }")
        # print(f"{idx_eve1 = }, {idx_eve2 = }, {idx_sta = }, {phase_type = }")
        # print(f"{template_array[idx_pick, :, 0, :][:10] = }, {template_array[jdx_pick, :, 0, :][:10] = }")
        # raise

        norm = np.sqrt(np.sum(data1**2, axis=-1, keepdims=True) * np.sum(data2**2, axis=-1, keepdims=True))
        if norm.all() == 0:
            continue

        # mask1 = traveltime_mask[idx_pick_dict[(i, idx_sta, phase_type)], :, 0]
        # mask2 = traveltime_mask[idx_pick_dict[(j, idx_sta, phase_type)], :, 0]
        mask1 = traveltime_mask[idx_pick, :, 0]
        mask2 = traveltime_mask[jdx_pick, :, 0]
        if not (mask1.any() and mask2.any()):
            continue

        ich = np.arange(3)[mask1 & mask2].tolist()
        if len(ich) == 0:
            continue

        # tt1 = traveltime_array[idx_pick_dict[(i, idx_sta, phase_type)], ich, 0]
        # tt2 = traveltime_array[idx_pick_dict[(j, idx_sta, phase_type)], ich, 0]
        tt1 = traveltime_array[idx_pick, ich, 0]
        tt2 = traveltime_array[jdx_pick, ich, 0]

        # phase_type = picks.loc[idx_pick, "phase_type"]

        cc_weight, cc_shift = correlate_traces(
            data1,
            data2,
            ich,
            max_shift=int(max_dt[phase_type] * config["sampling_rate"]),
            min_cc=config["cctorch"]["min_cc_score"],
            delta_tt=(tt1 - tt2) * config["sampling_rate"],
        )
        if (cc_weight == -1.0) or (cc_shift == -1.0):
            continue
        cc_shift = cc_shift / config["sampling_rate"]

        # if cc_weight < MIN_CC_WEIGHT:
        #     continue

        # cc_weight = np.random.uniform(0.0, 1.0)
        # cc_shift = tt1[-1] - tt2[-1]

        # assert tt1[0] == tt1[1]

        dt_array.append(
            {
                # "idx_eve1": i,
                # "idx_eve2": j,
                # "idx_sta": idx_sta,
                # "idx_eve1": picks.loc[idx_pick, "idx_eve"],
                # "idx_eve2": picks.loc[jdx_pick, "idx_eve"],
                # "idx_sta": picks.loc[idx_pick, "idx_sta"],
                "idx_eve1": idx_eve1,
                "idx_eve2": idx_eve2,
                "idx_sta": idx_sta,
                "phase_type": phase_type,
                "dt": cc_shift,
                "weight": cc_weight,
            }
        )

        if iii > 100:
            break

    # %%
    dt_array = pd.DataFrame(dt_array)
    print(dt_array)
    plt.figure()
    plt.hist(dt_array[dt_array["phase_type"] == "P"]["dt"], bins=100, alpha=0.5)
    plt.hist(dt_array[dt_array["phase_type"] == "S"]["dt"], bins=100, alpha=0.5)
    # bins = np.linspace(-20, 20, 41)
    # plt.hist(dt_array[dt_array["phase_type"] == "P"]["dt"], bins=bins, alpha=0.5)
    # plt.hist(dt_array[dt_array["phase_type"] == "S"]["dt"], bins=bins, alpha=0.5)
    plt.yscale("log")
    plt.savefig("debug_dt.png")

    plt.figure()
    plt.hist(dt_array[dt_array["phase_type"] == "P"]["weight"], bins=100, alpha=0.5)
    plt.hist(dt_array[dt_array["phase_type"] == "S"]["weight"], bins=100, alpha=0.5)
    # bins = np.linspace(-20, 20, 41)
    # plt.hist(dt_array[dt_array["phase_type"] == "P"]["dt"], bins=bins, alpha=0.5)
    # plt.hist(dt_array[dt_array["phase_type"] == "S"]["dt"], bins=bins, alpha=0.5)
    plt.yscale("log")
    plt.savefig("debug_weight.png")

    # %%
    stations.reset_index(inplace=True)
    stations["network_station"] = stations["network"] + "." + stations["station"]
    dt_array = dt_array.merge(stations[["network_station", "idx_sta"]], on="idx_sta", how="left")
    # dt_array = (
    #     dt_array.groupby(["idx_eve1", "idx_eve2", "network_station", "phase_type"])
    #     .apply(lambda x: x.nlargest(1, "weight"))
    #     .reset_index(drop=True)
    # )
    dt_array.sort_values("weight", ascending=False, inplace=True)
    dt_array = dt_array.groupby(["idx_eve1", "idx_eve2", "network_station", "phase_type"]).first().reset_index()
    dt_array.drop(columns=["network_station"], inplace=True)
    stations.set_index("idx_sta", inplace=True)

    # %%
    # fitler (idx_eve1, idx_eve2) pairs with at least MIN_OBS observations, and select MAX_OBS observations with the highest weight
    dt_array = (
        dt_array.groupby(["idx_eve1", "idx_eve2"])
        .apply(
            lambda x: (
                x.nlargest(config["cctorch"]["max_obs"], "weight") if len(x) >= config["cctorch"]["min_obs"] else None
            )
        )
        .reset_index(drop=True)
    )

    # %%
    # dt_array = dt_array.groupby(["idx_eve1", "idx_eve2"]).filter(
    #     lambda x: (x["phase_type"] == "P").sum() >= MIN_CC_NUM_P
    #     and (x["phase_type"] == "S").sum() >= MIN_CC_NUM_S
    #     and len(x) >= MIN_CC_NUM
    # )

    plt.figure()
    plt.hist(dt_array[dt_array["phase_type"] == "P"]["dt"], bins=100, alpha=0.5)
    plt.hist(dt_array[dt_array["phase_type"] == "S"]["dt"], bins=100, alpha=0.5)
    plt.yscale("log")
    plt.savefig("debug_dt_2.png")
    plt.show()

    plt.figure()
    plt.hist(dt_array[dt_array["phase_type"] == "P"]["weight"], bins=100, alpha=0.5)
    plt.hist(dt_array[dt_array["phase_type"] == "S"]["weight"], bins=100, alpha=0.5)
    plt.yscale("log")
    plt.savefig("debug_weight_2.png")
    plt.show()

    # %%
    event_idx_dict = events["event_index"].to_dict()  ##  faster than using .loc
    # station_id_dict = (stations["station"] + stations["instrument"]).to_dict()
    station_id_dict = stations["station"].to_dict()
    # with open(f"cctorch/dt.cc", "w") as fp:
    with open(f"{root_path}/{result_path}/dt.cc", "w") as fp:

        for (i, j), record in tqdm(dt_array.groupby(["idx_eve1", "idx_eve2"])):
            # event_id1 = events.loc[i]["event_id"]
            # event_id2 = events.loc[j]["event_id"]
            event_idx1 = event_idx_dict[i]
            event_idx2 = event_idx_dict[j]
            fp.write(f"# {event_idx1} {event_idx2} 0.000\n")
            for k, record_ in record.iterrows():
                idx_sta = record_["idx_sta"]
                # station_id = stations.loc[idx_sta]["station"] + stations.loc[idx_sta]["instrument"]
                station_id = station_id_dict[idx_sta]
                phase_type = record_["phase_type"]
                fp.write(f"{station_id} {record_['dt']: .4f} {record_['weight']:.4f} {phase_type}\n")

    # # %%
    # generate_pairs(
    #     events, min_pair_dist=config["cctorch"]["min_pair_dist_km"], fname=config["cctorch"]["pair_file"]
    # )


# %%
if __name__ == "__main__":

    # %%
    root_path = "local"
    # region = "demo"
    region = "debug"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]
    # with open(f"{root_path}/{region}/config.json", "r") as fp:
    #     config = json.load(fp)
    config = None

    # %%
    result_path = f"{region}/cctorch"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    print(json.dumps(config, indent=4, sort_keys=True))

    cut_templates(root_path, region, config)

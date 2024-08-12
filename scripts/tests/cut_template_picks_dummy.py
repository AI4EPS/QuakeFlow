# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pyproj import Proj
from adloc.eikonal2d import init_eikonal2d, calc_traveltime
from glob import glob
import obspy
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
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
root_path = "."
region = ""
result_path = "cctorch"
if not os.path.exists(f"{root_path}/{region}/{result_path}"):
    os.makedirs(f"{root_path}/{region}/{result_path}")

# %%
# MAX_EPICENTER_DIST = 50
MAX_EPICENTER_DIST = 200
MAX_PAIR_DIST = 10
MAX_NEIGHBORS = 50
MIN_WEIGHT = 0.0
MIN_OBS = 8  # based on pick counts
MAX_OBS = 100
MIN_CC_SCORE = 0.5
# MIN_CC_WEIGHT = 0.1  # 0.25**2 * 0.1
# MIN_CC_NUM_P = 10
# MIN_CC_NUM_S = 10
# MIN_CC_NUM = 20
# MIN_NEIGHBORS = 5

# %%
with open(f"../local/Ridgecrest/config.json", "r") as fp:
    config = json.load(fp)
    config.update(config["cctorch"])

print(json.dumps(config, indent=4))
config["time_before_p"] = 0.3
config["time_after_p"] = 2.5 - config["time_before_p"]
config["time_before_s"] = 0.3
config["time_after_s"] = 4.0 - config["time_before_s"]
config["time_window"] = max(
    (config["time_before_p"] + config["time_after_p"]), (config["time_before_s"] + config["time_after_s"])
)
config["nt"] = int(round(config["time_window"] * config["sampling_rate"]))

# %%
stations = pd.read_csv(f"../local/Ridgecrest/adloc/ransac_stations_sst.csv")
stations = stations[stations["network"] == "CI"]
stations.sort_values(by=["latitude", "longitude"], inplace=True)
print(f"{len(stations) = }")
print(stations.iloc[:5])

# %%
events = pd.read_csv(f"events.csv")
events.rename(columns={"time": "event_time"}, inplace=True)
events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
# events = events[
#     (events["event_time"] > pd.to_datetime(config["starttime"], utc=True))
#     & (events["event_time"] < pd.to_datetime(config["endtime"], utc=True))
# ]
# dx = np.random.uniform(-0.01, 0.01, len(events))
# dy = np.random.uniform(-0.01, 0.01, len(events))
# events["longitude"] += dx
# events["latitude"] += dy
print(f"{len(events) = }")
print(events.iloc[:5])

# %%
reference_t0 = events["event_time"].min()
events["event_timestamp"] = events["event_time"].apply(lambda x: (x - reference_t0).total_seconds())
min_phase_score = 0.1

# %%
lon0 = (config["minlongitude"] + config["maxlongitude"]) / 2
lat0 = (config["minlatitude"] + config["maxlatitude"]) / 2
proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")

# %%
stations[["x_km", "y_km"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
stations["z_km"] = stations["elevation_m"].apply(lambda x: -x / 1e3)
# stations["station_term_time"] = 0.0

events[["x_km", "y_km"]] = events.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
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
## QuakeFlow example
# picks = pd.read_csv(f"../local/Ridgecrest/adloc/ransac_picks_sst.csv")
# picks = picks[picks["adloc_mask"] == 1]
# picks = picks.merge(events[["event_index", "event_timestamp"]], on="event_index")
# picks = picks.merge(stations[["station_id", "station_term_time"]], on="station_id")
# picks["phase_time"] = pd.to_datetime(picks["phase_time"], utc=True)
# picks["phase_timestamp"] = picks["phase_time"].apply(lambda x: (x - reference_t0).total_seconds())
# picks["traveltime"] = picks["phase_timestamp"] - picks["event_timestamp"] - picks["station_term_time"]

# picks = fillin_missing_picks(
#     picks,
#     events,
#     stations,
#     config={"reference_t0": reference_t0, "vp_vs_ratio": vp_vs_ratio, "min_phase_score": min_phase_score},
# )

# if "dist_km" not in picks:
#     picks = picks.merge(stations[["station_id", "x_km", "y_km", "z_km"]], on="station_id")
#     picks.rename(columns={"x_km": "station_x_km", "y_km": "station_y_km", "z_km": "station_z_km"}, inplace=True)
#     picks = picks.merge(events[["event_index", "x_km", "y_km", "z_km"]], on="event_index")
#     picks.rename(columns={"x_km": "event_x_km", "y_km": "event_y_km", "z_km": "event_z_km"}, inplace=True)
#     picks["dist_km"] = np.linalg.norm(
#         picks[["event_x_km", "event_y_km", "event_z_km"]].values
#         - picks[["station_x_km", "station_y_km", "station_z_km"]].values,
#         axis=-1,
#     )
#     picks.drop(
#         columns=["station_x_km", "station_y_km", "station_z_km", "event_x_km", "event_y_km", "event_z_km"], inplace=True
#     )

# ## Shelley example
picks = predict_full_picks(
    None,
    events,
    stations,
    config={
        "eikonal": eikonal,
        "vp_vs_ratio": vp_vs_ratio,
        "reference_t0": reference_t0,
        "min_phase_score": min_phase_score,
    },
)


print(f"{len(picks) = }")
picks = picks[picks["dist_km"] < MAX_EPICENTER_DIST]
print(f"{len(picks) = } with dist_km < {MAX_EPICENTER_DIST}")

# %%
stations["idx_sta"] = np.arange(len(stations))
events["idx_eve"] = np.arange(len(events))
picks = picks.merge(events[["event_index", "idx_eve"]], on="event_index")
picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")
if "magnitude" not in events:
    events["magnitude"] = 0.0
events = events[
    [
        "idx_eve",
        "x_km",
        "y_km",
        "z_km",
        "event_time",
        "event_timestamp",
        "event_index",
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
    ]
]
stations = stations[
    [
        "idx_sta",
        "x_km",
        "y_km",
        "z_km",
        "station_id",
        "component",
        "network",
        "station",
        "instrument",
        "latitude",
        "longitude",
        "elevation_m",
    ]
]
picks = picks[
    [
        "idx_eve",
        "idx_sta",
        "phase_type",
        "phase_score",
        "phase_time",
        "phase_timestamp",
        "phase_source",
    ]
]
picks.sort_values(by=["idx_eve", "idx_sta", "phase_type"], inplace=True)
picks["idx_pick"] = np.arange(len(picks))

events.to_csv(f"cctorch/cctorch_events.csv", index=False)
stations.to_csv(f"cctorch/cctorch_stations.csv", index=False)
picks.to_csv(f"cctorch/cctorch_picks.csv", index=False)

dirs = sorted(glob(f"../local/Ridgecrest/waveforms/????/???/??"))

# %%
events.set_index("idx_eve", inplace=True)
stations.set_index("idx_sta", inplace=True)
# picks.set_index(["idx_eve", "idx_sta", "phase_type"], inplace=True)

# %%
traveltime_array = np.zeros((len(picks), 3, 1), dtype=np.float32)
traveltime_index_array = np.zeros((len(picks), 3, 1), dtype=int)

# %%
for mseed_path in tqdm(dirs):

    tmp = mseed_path.split("/")
    year, jday, hour = tmp[-3], tmp[-2], tmp[-1]
    begin_time = pd.to_datetime(f"{year}-{jday}T{hour}", utc=True, format="%Y-%jT%H")
    end_time = begin_time + pd.Timedelta(hours=1)

    picks_ = picks[(picks["phase_time"] >= begin_time) & (picks["phase_time"] < end_time)]
    picks_.set_index(["idx_eve", "idx_sta", "phase_type"], inplace=True)

    if len(picks_) == 0:
        # print(f"No picks_ in {mseed_path}")
        continue

    normalize = lambda x: (x - x.mean()) / (x.std() + 1e-6) / 6

    for (idx_eve, idx_sta, phase_type), pick in picks_.iterrows():

        idx_pick = pick["idx_pick"]
        phase_timestamp = pick["phase_timestamp"]

        station = stations.loc[idx_sta]
        station_id = station["station_id"]
        event = events.loc[idx_eve]

        for c in station["component"]:
            ic = config["component_mapping"][c]  # 012 for P, 345 for S

            trace_starttime = 0.0
            begin_time = phase_timestamp - trace_starttime - config[f"time_before_{phase_type.lower()}"]
            end_time = phase_timestamp - trace_starttime + config[f"time_after_{phase_type.lower()}"]

            if phase_type == "P" and ((idx_eve, idx_sta, "S") in picks_.index):
                s_begin_time = (
                    picks_.loc[idx_eve, idx_sta, "S"]["phase_timestamp"] - trace_starttime - config[f"time_before_s"]
                )
                end_time = min(end_time, s_begin_time)

            begin_time_index = max(0, int(round(begin_time * config["sampling_rate"])))
            end_time_index = max(0, int(round(end_time * config["sampling_rate"])))

            ## define traveltime at the exact data point of event origin time
            traveltime_array[idx_pick, ic, 0] = begin_time_index / config["sampling_rate"] - (
                event["event_timestamp"] - trace_starttime - config[f"time_before_{phase_type.lower()}"]
            )
            # traveltime_array[idx_pick, ic, 0] = begin_time - (
            #     event["event_timestamp"] - trace_starttime - config[f"time_before_{phase_type.lower()}"]
            # )  ## should have same effect as super resolution
            traveltime_index_array[idx_pick, ic, 0] = begin_time_index - int(
                (event["event_timestamp"] - trace_starttime - config[f"time_before_{phase_type.lower()}"])
                * config["sampling_rate"]
            )

# %%
neigh = NearestNeighbors(radius=MAX_PAIR_DIST, n_jobs=-1)
neigh.fit(events[["x_km", "y_km", "z_km"]].values)
pairs = set()
neigh_ind = neigh.radius_neighbors(sort_results=True)[1]

# idx_sta_dict = picks.groupby("idx_eve")["idx_sta"].apply(set).to_dict()
for i, neighs in enumerate(tqdm(neigh_ind, desc="Generating pairs")):
    # n = 0
    for j in neighs[:MAX_NEIGHBORS]:
        # if i not in idx_sta_dict or j not in idx_sta_dict:
        #     continue
        # if len(idx_sta_dict[i] & idx_sta_dict[j]) < MIN_LINKS:
        #     continue

        if i > j:
            pairs.add((j, i))
        else:
            pairs.add((i, j))

        # n += 1
        # if n >= MAX_NEIGHBORS:
        #     break

pairs = list(pairs)
print(f"{len(pairs) = } {len(events) = }")


# %%
cc_array = []
dt_array = []
max_dt = {"P": 0.5, "S": 0.85}
picks.set_index(["idx_eve", "idx_sta", "phase_type"], inplace=True)
idx_pick_dict = picks["idx_pick"].to_dict()  ## much faster than using .loc

for ii, (i, j) in enumerate(tqdm(pairs)):
    for idx_sta in stations.index:
        for phase_type in ["P", "S"]:

            # if (i, idx_sta, phase_type) not in picks.index or (j, idx_sta, phase_type) not in picks.index:
            #     continue
            # tt1 = traveltime_array[picks.loc[(i, idx_sta, phase_type), "idx_pick"], :, 0]
            # tt2 = traveltime_array[picks.loc[(j, idx_sta, phase_type), "idx_pick"], :, 0]

            if (i, idx_sta, phase_type) not in idx_pick_dict or (j, idx_sta, phase_type) not in idx_pick_dict:
                continue
            tt1 = traveltime_array[idx_pick_dict[(i, idx_sta, phase_type)], :, 0]
            tt2 = traveltime_array[idx_pick_dict[(j, idx_sta, phase_type)], :, 0]

            cc_weight = np.random.uniform(0.0, 1.0)
            cc_shift = tt1[-1] - tt2[-1]

            # assert tt1[0] == tt1[1]

            dt_array.append(
                {
                    "idx_eve1": i,
                    "idx_eve2": j,
                    "idx_sta": idx_sta,
                    "phase_type": phase_type,
                    "dt": cc_shift,
                    "weight": cc_weight,
                }
            )
picks.reset_index(inplace=True)


# %%
dt_array = pd.DataFrame(dt_array)
plt.figure()
plt.hist(dt_array[dt_array["phase_type"] == "P"]["dt"], bins=100, alpha=0.5)
plt.hist(dt_array[dt_array["phase_type"] == "S"]["dt"], bins=100, alpha=0.5)
# bins = np.linspace(-20, 20, 41)
# plt.hist(dt_array[dt_array["phase_type"] == "P"]["dt"], bins=bins, alpha=0.5)
# plt.hist(dt_array[dt_array["phase_type"] == "S"]["dt"], bins=bins, alpha=0.5)
plt.yscale("log")
plt.show()

plt.figure()
plt.hist(dt_array[dt_array["phase_type"] == "P"]["weight"], bins=100, alpha=0.5)
plt.hist(dt_array[dt_array["phase_type"] == "S"]["weight"], bins=100, alpha=0.5)
# bins = np.linspace(-20, 20, 41)
# plt.hist(dt_array[dt_array["phase_type"] == "P"]["dt"], bins=bins, alpha=0.5)
# plt.hist(dt_array[dt_array["phase_type"] == "S"]["dt"], bins=bins, alpha=0.5)
plt.yscale("log")
plt.show()

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
    .apply(lambda x: x.nlargest(MAX_OBS, "weight") if len(x) >= MIN_OBS else None)
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
plt.show()

plt.figure()
plt.hist(dt_array[dt_array["phase_type"] == "P"]["weight"], bins=100, alpha=0.5)
plt.hist(dt_array[dt_array["phase_type"] == "S"]["weight"], bins=100, alpha=0.5)
plt.yscale("log")
plt.show()

# %%
event_idx_dict = events["event_index"].to_dict()  ##  faster than using .loc
# station_id_dict = (stations["station"] + stations["instrument"]).to_dict()
station_id_dict = stations["station"].to_dict()
with open(f"cctorch/dt.cc", "w") as fp:

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

# %%

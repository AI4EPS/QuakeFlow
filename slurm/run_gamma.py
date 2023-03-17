# Applying GaMMA to associate PhaseNet picks

# %%
import pandas as pd
from datetime import datetime, timedelta
from gamma import BayesianGaussianMixture, GaussianMixture
from gamma.utils import convert_picks_csv, association, from_seconds
import numpy as np
from sklearn.cluster import DBSCAN 
from datetime import datetime, timedelta
import os
import json
import pickle
from pyproj import Proj
from tqdm import tqdm

# %%
data_dir = lambda x: os.path.join("results", x)
# station_csv = data_dir("stations.csv")
station_json = data_dir("stations.json")
pick_csv = data_dir("picks.csv")
catalog_csv = data_dir("gamma_catalog.csv")
picks_csv = data_dir("gamma_picks.csv")
if not os.path.exists("figures"):
    os.makedirs("figures")
figure_dir = lambda x: os.path.join("figures", x)

# %%
with open(data_dir("config.json"), "r") as f:
    config = json.load(f)

## read picks
picks = pd.read_csv(pick_csv)
picks["id"] = picks["station_id"]
picks["timestamp"] = picks["phase_time"]
if "phase_amp" in picks.columns:
    picks["amp"] = picks["phase_amp"]
    picks["phase_amplitude"] = picks["phase_amp"]
if "phase_amplitude" in picks.columns:
    picks["amp"] = picks["phase_amplitude"]
picks["type"] = picks["phase_type"]
picks["prob"] = picks["phase_score"]

## read stations
# stations = pd.read_csv(station_csv, delimiter="\t")
stations = pd.read_json(station_json, orient="index")
# stations = stations.rename(columns={"station":"id"})
stations["id"] = stations.index
proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
stations[["x(km)", "y(km)"]] = stations.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x/1e3)

### setting GMMA configs
config["use_dbscan"] = True
config["use_amplitude"] = True
config["method"] = "BGMM"  
if config["method"] == "BGMM": ## BayesianGaussianMixture
    config["oversample_factor"] = 4
if config["method"] == "GMM": ## GaussianMixture
    config["oversample_factor"] = 1

# earthquake location
config["vel"] = {"p": 6.0, "s": 6.0 / 1.75}
config["dims"] = ['x(km)', 'y(km)', 'z(km)']
config["x(km)"] = (np.array(config["xlim_degree"])-np.array(config["center"][0]))*config["degree2km"]*np.cos(np.deg2rad(config["center"][1]))
config["y(km)"] = (np.array(config["ylim_degree"])-np.array(config["center"][1]))*config["degree2km"]
config["z(km)"] = (0, 20)
config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # z
    (None, None),  # t
)

# DBSCAN
config["dbscan_eps"] = 10 #s
# config["dbscan_eps"] = 6 #s
config["dbscan_min_samples"] = 3

## Eikonal for 1D velocity model
# zz = [0.0, 5.5, 16.0, 32.0]
# vp = [5.5, 5.5,  6.7,  7.8]
# vp_vs_ratio = 1.73
# vs = [v / vp_vs_ratio for v in vp]
# h = 0.3
# # h = 3
# vel = {"z": zz, "p": vp, "s": vs}
# config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}


# filtering
config["min_picks_per_eq"] = min(10, len(stations) // 2)
config["min_p_picks_per_eq"] = 0
config["min_s_picks_per_eq"] = 0
config["max_sigma11"] = 2.0 # s
config["max_sigma22"] = 1.0 # log10(m/s)
config["max_sigma12"] = 1.0 # covariance

## filter picks without amplitude measurements
if config["use_amplitude"]:
    picks = picks[picks["amp"] != -1]

for k, v in config.items():
    print(f"{k}: {v}")


# %%
event_idx0 = 0 ## current earthquake index
assignments = []
catalogs, assignments = association(picks, stations, config, event_idx0, config["method"])
event_idx0 += len(catalogs)

## create catalog
catalogs = pd.DataFrame(catalogs, columns=["time"]+config["dims"]+["magnitude", "sigma_time", "sigma_amp", "cov_time_amp",  "event_index", "gamma_score"])
catalogs[["longitude","latitude"]] = catalogs.apply(lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1)
catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: x*1e3)
catalogs.sort_values("time", inplace=True)
with open(catalog_csv, 'w') as fp:
    catalogs.to_csv(fp, index=False, 
                    float_format="%.3f",
                    date_format='%Y-%m-%dT%H:%M:%S.%f',
                    columns=["time", "magnitude", "longitude", "latitude", "depth(m)", "sigma_time", "sigma_amp", "cov_time_amp", "event_index", "gamma_score"])
# catalogs = catalogs[['time', 'magnitude', 'longitude', 'latitude', 'depth(m)', 'sigma_time', 'sigma_amp', 'gamma_score']]

## add assignment to picks
assignments = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
picks = picks.join(assignments.set_index("pick_index")).fillna(-1).astype({'event_index': int})
with open(picks_csv, 'w') as fp:
    picks.to_csv(fp, index=False, 
                    date_format='%Y-%m-%dT%H:%M:%S.%f',
                    columns=["station_id", "phase_time", "phase_type", "phase_score", "phase_amplitude", "event_index", "gamma_score"])



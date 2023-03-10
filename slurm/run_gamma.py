# %% [markdown]
# # PhaseNet Example
# 
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
picks["amp"] = picks["phase_amp"]
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
                    columns=["station_id", "phase_time", "phase_type", "phase_score", "phase_amp", "event_index", "gamma_score"])


# # %%
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# result_label="GaMMA"
# catalog_label="SCSN"

# # %%
# stations = pd.read_json(station_json, orient="index")
# # stations = stations.rename(columns={"station":"id"})
# stations["id"] = stations.index
# # events = pd.read_csv(data_dir("events.csv"), delimiter="\t")
# # events["time"] = events["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))

# catalog = pd.read_csv(data_dir("catalog_gamma.csv"), delimiter="\t")
# catalog["time"] = catalog["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))

# plt.figure()
# plt.hist(catalog["time"], range=(config["starttime"], config["endtime"]), bins=24, edgecolor="k", alpha=1.0, linewidth=0.5, label=f"{result_label}: {len(catalog['time'])}")
# # plt.hist(events["time"], range=(config["starttime"], config["endtime"]), bins=24, edgecolor="k", alpha=0.6, linewidth=0.5, label=f"{catalog_label}: {len(events['time'])}")
# plt.ylabel("Frequency")
# plt.xlabel("Date")
# plt.gca().autoscale(enable=True, axis='x', tight=True)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d:%H'))
# plt.gcf().autofmt_xdate()
# plt.legend()
# plt.savefig(figure_dir("earthquake_number.png"), bbox_inches="tight", dpi=300)
# plt.savefig(figure_dir("earthquake_number.pdf"), bbox_inches="tight")
# plt.show();

# # %%
# fig = plt.figure(figsize=plt.rcParams["figure.figsize"]*np.array([1.5,1]))
# box = dict(boxstyle='round', facecolor='white', alpha=1)
# text_loc = [0.05, 0.92]
# grd = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1.5, 1], height_ratios=[1,1])
# fig.add_subplot(grd[:, 0])
# plt.plot(catalog["longitude"], catalog["latitude"], '.',markersize=2, alpha=1.0)
# # plt.plot(events["longitude"], events["latitude"], '.', markersize=2, alpha=0.6)
# plt.axis("scaled")
# plt.xlim(np.array(config["xlim_degree"])+np.array([0.2,-0.27]))
# plt.ylim(np.array(config["ylim_degree"])+np.array([0.2,-0.27]))
# plt.xlabel("Latitude")
# plt.ylabel("Longitude")
# plt.gca().set_prop_cycle(None)
# plt.plot(config["xlim_degree"][0]-10, config["ylim_degree"][0]-10, '.', markersize=10, label=f"{result_label}", rasterized=True)
# plt.plot(config["xlim_degree"][0]-10, config["ylim_degree"][0]-10, '.', markersize=10, label=f"{catalog_label}", rasterized=True)
# plt.plot(stations["longitude"], stations["latitude"], 'k^', markersize=5, alpha=0.7, label="Stations")
# plt.legend(loc="lower right")
# plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='left', verticalalignment="top", 
#          transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)

# fig.add_subplot(grd[0, 1])
# plt.plot(catalog["longitude"], catalog["depth(m)"]/1e3, '.', markersize=2, alpha=1.0, rasterized=True)
# # plt.plot(events["longitude"], events["depth(m)"]/1e3, '.', markersize=2, alpha=0.6, rasterized=True)
# # plt.axis("scaled")
# plt.xlim(np.array(config["xlim_degree"])+np.array([0.2,-0.27]))
# plt.ylim([0,21])
# plt.gca().invert_yaxis()
# plt.xlabel("Longitude")
# plt.ylabel("Depth (km)")
# plt.gca().set_prop_cycle(None)
# plt.plot(config["xlim_degree"][0]-10, 31, '.', markersize=10, label=f"{result_label}")
# plt.plot(31, 31, '.', markersize=10, label=f"{catalog_label}")
# plt.legend(loc="lower right")
# plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='left', verticalalignment="top", 
#          transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)

# fig.add_subplot(grd[1, 1])
# plt.plot(catalog["latitude"], catalog["depth(m)"]/1e3, '.', markersize=2, alpha=1.0, rasterized=True)
# # plt.plot(events["latitude"], events["depth(m)"]/1e3, '.', markersize=2, alpha=0.6, rasterized=True)
# # plt.axis("scaled")
# plt.xlim(np.array(config["ylim_degree"])+np.array([0.2,-0.27]))
# plt.ylim([0,21])
# plt.gca().invert_yaxis()
# plt.xlabel("Latitude")
# plt.ylabel("Depth (km)")
# plt.gca().set_prop_cycle(None)
# plt.plot(config["ylim_degree"][0]-10, 31, '.', markersize=10, label=f"{result_label}")
# plt.plot(31, 31, '.', markersize=10, label=f"{catalog_label}")
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='left', verticalalignment="top", 
#          transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)
# plt.savefig(figure_dir("earthquake_location.png"), bbox_inches="tight", dpi=300)
# plt.savefig(figure_dir("earthquake_location.pdf"), bbox_inches="tight", dpi=300)
# plt.show();

# # %%
# plt.figure()
# plt.hist(catalog["magnitude"], range=(-1., catalog["magnitude"].max()), bins=25, alpha=1.0,  edgecolor="k", linewidth=0.5, label=f"{result_label}: {len(catalog['magnitude'])}")
# # plt.hist(events["magnitude"], range=(-1., catalog["magnitude"].max()), bins=25, alpha=0.6,  edgecolor="k", linewidth=0.5, label=f"{catalog_label}: {len(events['magnitude'])}")
# plt.legend()
# # plt.figure()
# plt.xlim([-1,catalog["magnitude"].max()])
# plt.xlabel("Magnitude")
# plt.ylabel("Frequency")
# plt.gca().set_yscale('log')
# plt.savefig(figure_dir("earthquake_magnitude_frequency.png"), bbox_inches="tight", dpi=300)
# plt.savefig(figure_dir("earthquake_magnitude_frequency.pdf"), bbox_inches="tight")
# plt.show();

# # %%
# plt.figure()
# plt.plot(catalog["time"], catalog["magnitude"], '.', markersize=5, alpha=1.0, rasterized=True)
# # plt.plot(events["time"], events["magnitude"], '.', markersize=5, alpha=0.8, rasterized=True)
# plt.xlim(config["starttime"], config["endtime"])
# ylim = plt.ylim()
# plt.ylabel("Magnitude")
# # plt.xlabel("Date")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d:%H'))
# plt.gcf().autofmt_xdate()
# plt.gca().set_prop_cycle(None)
# plt.plot(config["starttime"], -10, '.', markersize=15, alpha=1.0, label=f"{result_label}: {len(catalog['magnitude'])}")
# plt.plot(config["starttime"], -10, '.', markersize=15, alpha=1.0, label=f"{catalog_label}: {len(events['magnitude'])}")
# plt.legend()
# plt.ylim(ylim)
# plt.grid()
# plt.savefig(figure_dir("earthquake_magnitude_time.png"), bbox_inches="tight", dpi=300)
# plt.savefig(figure_dir("earthquake_magnitude_time.pdf"), bbox_inches="tight", dpi=300)
# plt.show();

# # %%
# # covariance = np.array(catalog["covariance"].to_list())
# fig = plt.figure(figsize=plt.rcParams["figure.figsize"]*np.array([0.8,1.1]))
# box = dict(boxstyle='round', facecolor='white', alpha=1)
# text_loc = [0.05, 0.90]
# plt.subplot(311)
# plt.plot(catalog["time"], catalog["sigma_time"], '.', markersize=3.0, label="Travel-time")
# # plt.ylim([0, 3])
# plt.ylabel(r"$\sigma_{11}$ (s)")
# plt.legend(loc="upper right")
# plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='left', verticalalignment="top", 
#          transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)
# plt.subplot(312)
# plt.plot(catalog["time"], catalog["sigma_amp"], '.', markersize=3.0, label="Amplitude")
# # plt.ylim([0, 1])
# plt.ylabel(r"$\sigma_{22}$ ($\log10$ m/s)")
# plt.legend(loc="upper right")
# plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='left', verticalalignment="top", 
#          transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)
# plt.subplot(313)
# plt.plot(catalog["time"], catalog["cov_time_amp"], '.', markersize=3.0, label="Travel-time vs. Amplitude")
# plt.ylabel(r"$\Sigma_{12}$")
# plt.ylim([-0.5, 0.5])
# plt.legend(loc="upper right")
# plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='left', verticalalignment="top", 
#          transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d:%H'))
# plt.gcf().autofmt_xdate()
# # plt.suptitle(r"Covariance Matrix ($\Sigma$) Coefficients")
# plt.tight_layout()
# plt.gcf().align_labels()
# plt.savefig(figure_dir("covariance.png"), bbox_inches="tight", dpi=300)
# plt.savefig(figure_dir("covariance.pdf"), bbox_inches="tight")
# plt.show();



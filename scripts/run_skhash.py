# %%
import argparse
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy.imaging.beachball import beach
from sklearn.cluster import DBSCAN

# %%
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("root_path", nargs="?", type=str, default="local", help="root path")
#     parser.add_argument("region", nargs="?", type=str, default="demo", help="region")
#     return parser.parse_args()


# args = parse_args()
# root_path = args.root_path
# region = args.region

# %%
root_path = "local"
region = "demo"

result_path = f"{region}/skhash"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")

if not os.path.exists(f"{root_path}/{result_path}/IN"):
    os.makedirs(f"{root_path}/{result_path}/IN")

# %%
if not os.path.exists(f"../SKHASH"):
    os.system(f"git clone git@github.com:AI4EPS/SKHASH.git ../SKHASH")

# %%
stations = pd.read_json(f"{root_path}/{region}/results/data/stations.json", orient="index")
stations["station_id"] = stations.index
stations = stations[
    ["station_id", "network", "station", "location", "instrument", "latitude", "longitude", "elevation_m"]
]
stations.rename(columns={"elevation_m": "elevation", "instrument": "channel"}, inplace=True)
stations["location"] = stations.apply(lambda row: "--" if row["location"] == "" else row["location"], axis=1)
stations["channel"] = stations.apply(lambda row: row["channel"] + "Z", axis=1)
stations.to_csv(
    f"{root_path}/{result_path}/IN/stations.csv",
    index=False,
    columns=["network", "station", "location", "channel", "latitude", "longitude", "elevation"],
)

# %%
events = pd.read_csv(f"{root_path}/{region}/adloc/adloc_events.csv", parse_dates=["time"])
# events = pd.read_csv(f"{root_path}/{region}/adloc/debug_events.csv", parse_dates=["time"])
if "magnitude" not in events.columns:
    events["magnitude"] = pd.NA
events = events[["event_index", "time", "latitude", "longitude", "depth_km", "magnitude"]]
events.rename(columns={"event_index": "event_id", "depth_km": "depth", "magnitude": "mag"}, inplace=True)
events["time"] = events["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
events["horz_uncert_km"] = 0.0
events["vert_uncert_km"] = 0.0
events = events[events["depth"] >= 0]
events.to_csv(f"{root_path}/{result_path}/IN/catalog.csv", index=None)

# %%
picks = pd.read_csv(f"{root_path}/{region}/adloc/adloc_picks.csv", parse_dates=["phase_time"])
if "adloc_mask" in picks.columns:
    picks = picks[picks["adloc_mask"] == 1.0]
# picks = pd.read_csv(f"{root_path}/{region}/adloc/debug_picks.csv", parse_dates=["phase_time"])
picks.sort_values("phase_index", inplace=True)
picks.rename(columns={"event_index": "event_id"}, inplace=True)

# TODO: check why ADLOC keep picks that are not in events
picks = picks.merge(events[["event_id"]], on="event_id", how="right")


# TODO: do the filtering in EQNet
def filter_duplicates(picks):
    picks_filt = []
    picks["t"] = (picks["phase_time"] - picks["phase_time"].min()).dt.total_seconds()
    print(f"before {len(picks)} picks")
    dbscan = DBSCAN(eps=0.10, min_samples=1)
    for station_id, picks_ in picks.groupby("station_id"):
        dbscan.fit(picks_[["t"]])
        picks_["label"] = dbscan.labels_
        picks_ = picks_.groupby("label").first().reset_index()
        picks_.drop(columns=["label", "t"], inplace=True)
        picks_filt.append(picks_)
    picks_filt = pd.concat(picks_filt)
    print(f"after {len(picks_filt)} picks")
    return picks_filt


picks = filter_duplicates(picks)

picks = picks[picks["event_id"] != -1]
picks = picks.merge(
    stations[["station_id", "network", "station", "location", "channel"]],
    on=["station_id"],
    how="left",
)
ppicks = picks[picks["phase_type"] == "P"].copy()
ppicks.rename(columns={"phase_polarity": "p_polarity"}, inplace=True)
ppicks = ppicks[["event_id", "network", "station", "location", "channel", "p_polarity"]]
ppicks.sort_values(["event_id", "network", "station", "location", "channel"], inplace=True)
ppicks.to_csv(f"{root_path}/{result_path}/IN/pol.csv", index=None)

# %%
picks["station_id"] = picks["network"] + "." + picks["station"] + "." + picks["location"] + "." + picks["channel"]
stations["station_id"] = (
    stations["network"] + "." + stations["station"] + "." + stations["location"] + "." + stations["channel"]
)
sp_ratio = []
for (event_id, staton_id), picks_ in picks.groupby(["event_id", "station_id"]):
    if len(picks_["phase_type"].unique()) < 2:  ## assume P and S
        continue
    if picks_["phase_score"].max() < 0.3:
        continue
    ratio = (
        picks_[picks_["phase_type"] == "S"]["phase_amplitude"].values[0]
        / picks_[picks_["phase_type"] == "P"]["phase_amplitude"].values[0]
    )
    network, station, location, channel = staton_id.split(".")
    sp_ratio.append(
        {
            "event_id": event_id,
            "network": network,
            "station": station,
            "location": location,
            "channel": channel,
            "sp_ratio": ratio,
        }
    )
sp_ratio = pd.DataFrame(sp_ratio)
sp_ratio.to_csv(f"{root_path}/{result_path}/IN/amp.csv", index=None)

# %%
# 1D model from Shelly (2020)
velocity_model = """# 1D model from Shelly (2020)
# Depth(km), Vp(km/s)
0.00, 4.74000
1.00, 5.01000
2.00, 5.35000
3.00, 5.71000
4.00, 6.07000
5.00, 6.17000
6.00, 6.27000
7.00, 6.34000
8.00, 6.39000
30.00, 7.80000
"""
# with open(f"./IN/ridgecrest.txt", "w") as fp:
#     fp.writelines(velocity_model)
with open(f"{root_path}/{result_path}/IN/ridgecrest.txt", "w") as fp:
    fp.writelines(velocity_model)

# %%
control_params = f"""## Control file for SKHASH_ridgecrest
$stfile        # station filepath
{root_path}/{result_path}/IN/stations.csv

$catfile       # earthquake catalog filepath
{root_path}/{result_path}/IN/catalog.csv

$vmodel_paths
{root_path}/{result_path}/IN/ridgecrest.txt

$fpfile        # P-polarity input filepath
{root_path}/{result_path}/IN/pol.csv

$ampfile       # amplitude input filepath
{root_path}/{result_path}/IN/amp.csv

$outfile1      # focal mechanisms output filepath
{root_path}/{result_path}/OUT/out.csv

$outfile_pol_agree  # record of polarity (dis)agreeement output filepath
{root_path}/{result_path}/OUT/out_polagree.csv

$outfile_sp_agree
{root_path}/{result_path}/OUT/out_spagree.csv

$outfile_pol_info
{root_path}/{result_path}/OUT/out_polinfo.csv

$outfolder_plots
{root_path}/{result_path}/OUT/figures

$npolmin       # mininum number of polarity data (e.g., 8)
8

$min_polarity_weight  # Any polarities with a abs(weight) < min_polarity_weight will be ignored
0.1

$nmc           # number of trials (e.g., 30)
50

$maxout        # max num of acceptable focal mech. outputs (e.g., 500)
500

$badfrac       # fraction polarities assumed bad
0.1

$qbadfrac      # assumed noise in amplitude ratios, log10 (e.g. 0.3 for a factor of 2)
0.2

$delmax        # maximum allowed source-receiver distance in km.
100

$prob_max      # probability threshold for multiples (e.g., 0.1)
0.2

$max_agap      # maximum azimuthal gap between stations in degree
150

$max_pgap      # maximum "plungal" gap between stations in degree
90

$cangle        # cutoff angle for computing focal mechanism probability
45

$num_cpus      # number of cpus for computing
50
"""
with open(f"{root_path}/{result_path}/control_file.txt", "w") as fp:
    fp.writelines(control_params)

# %%
# ! python SKHASH.py control_file.txt
os.system(f"python ../SKHASH/SKHASH/SKHASH.py {root_path}/{result_path}/control_file.txt")

# %%
fm_sol = pd.read_csv(f"{root_path}/{result_path}/OUT/out.csv")
fm_sol = fm_sol.merge(events[["event_id", "mag"]], on="event_id")
fm_sol["mag"] = fm_sol["mag"].fillna(3)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=100)
for i in fm_sol.index:
    strike, dip, rake, q = (
        fm_sol.loc[i, "strike"],
        fm_sol.loc[i, "dip"],
        fm_sol.loc[i, "rake"],
        fm_sol.loc[i, "quality"],
    )
    evdp, evla, evlo, mag = (
        fm_sol.loc[i, "origin_depth_km"],
        fm_sol.loc[i, "origin_lat"],
        fm_sol.loc[i, "origin_lon"],
        fm_sol.loc[i, "mag"],
    )
    if q != "A":
        bball = beach(fm=[strike, dip, rake], linewidth=0.5, facecolor="k", xy=(evlo, evla), width=mag * 2e-3)
        ax.add_collection(bball)
    else:
        bball = beach(fm=[strike, dip, rake], linewidth=0.5, facecolor="r", xy=(evlo, evla), width=mag * 2e-3)
        ax.add_collection(bball)
ax.set_xlim(-117.65, -117.45)
ax.set_ylim(35.58, 35.76)
fig.tight_layout()
plt.show
# %%

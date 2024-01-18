
# %%
from typing import Dict, List, NamedTuple

from kfp import dsl

@dsl.component(base_image="zhuwq0/quakeflow_plotting:latest")
def plot_gamma()
# %%
import json
from glob import glob
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pygmt
from matplotlib.colors import LightSource
from datetime import datetime

matplotlib.use("Agg")

# %%
root_path = "local"
region = "demo"
result_path = f"{region}/gamma"
plot_standard_catalog = True
use_pygmt = True

# %%
with open(f"{root_path}/{region}/config.json", "r") as f:
    config = json.load(f)

# %%
stations = pd.read_json(f"{root_path}/{region}/obspy/stations.json", orient="index")
stations["id"] = stations.index

# %%
gamma_events = []
for csv in glob(f"{root_path}/{region}/gamma/gamma_events*.csv"):
    print(csv)
    gamma_events.append(pd.read_csv(csv, parse_dates=["time"]))
gamma_events = pd.concat(gamma_events, ignore_index=True)
# gamma_events = pd.read_csv(f"{root_path}/{region}/gamma/gamma_events.csv", parse_dates=["time"])
if plot_standard_catalog:
    standard_catalog = pd.read_csv(f"{root_path}/{region}/obspy/events.csv", parse_dates=["time"])

# %%
gamma_label = "GaMMA"
standard_label = "Standard"

# %%
# unit = "D"
# starttime = np.datetime64(config["starttime"], unit)
# endtime = np.datetime64(config["endtime"], unit)
# bins = np.arange(starttime, endtime, np.timedelta64(1, "D"), dtype="datetime64[D]")

bins = pd.date_range(start=config["starttime"], end=config["endtime"], periods=30)
plt.figure(figsize=plt.rcParams["figure.figsize"] * np.array([1.5, 1.0]))
plt.hist(
    gamma_events["time"],
    bins=bins,
    edgecolor="k",
    alpha=1.0,
    linewidth=0.5,
    label=f"{gamma_label}: {len(gamma_events['time'])}",
)
if plot_standard_catalog:
    plt.hist(
        standard_catalog["time"],
        bins=bins,
        edgecolor="k",
        alpha=0.6,
        linewidth=0.5,
        label=f"{standard_label}: {len(standard_catalog['time'])}",
    )
plt.ylabel("Frequency")
plt.xlabel("Date")
plt.gca().autoscale(enable=True, axis="x", tight=True)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d:%H'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.grid(linestyle="--", linewidth=0.5, alpha=0.5)
plt.savefig(f"{root_path}/{result_path}/earthquake_number.png", bbox_inches="tight", dpi=300)
plt.savefig(f"{root_path}/{result_path}/earthquake_number.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%
if plot_standard_catalog:
    bins = np.arange(-1, standard_catalog["magnitude"].max() + 0.19, 0.2)
else:
    bins = np.arange(-1, gamma_events["magnitude"].max() + 0.19, 0.2)
plt.figure()
plt.hist(
    gamma_events["magnitude"],
    bins=bins,
    alpha=1.0,
    edgecolor="k",
    linewidth=0.5,
    label=f"{gamma_label}: {len(gamma_events['magnitude'])}",
)
if plot_standard_catalog:
    plt.hist(
        standard_catalog["magnitude"],
        bins=bins,
        alpha=0.6,
        edgecolor="k",
        linewidth=0.5,
        label=f"{standard_label}: {len(standard_catalog['magnitude'])}",
    )
plt.legend()
plt.xlabel("Magnitude")
plt.ylabel("Frequency")
plt.gca().set_yscale("log")
plt.savefig(f"{root_path}/{result_path}/earthquake_magnitude_frequency.png", bbox_inches="tight", dpi=300)
plt.savefig(f"{root_path}/{result_path}/earthquake_magnitude_frequency.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%
gamma_markersize = min(2, 1e5 / len(gamma_events))
standard_markersize = min(2, 1e4 / len(standard_catalog))
plt.figure(figsize=plt.rcParams["figure.figsize"] * np.array([1.5, 1.0]))

plt.scatter(
    gamma_events["time"], gamma_events["magnitude"], s=gamma_markersize, alpha=1.0, linewidth=0, rasterized=True
)
if plot_standard_catalog:
    plt.scatter(
        standard_catalog["time"],
        standard_catalog["magnitude"],
        s=standard_markersize,
        alpha=1.0,
        linewidth=0,
        rasterized=True,
    )

plt.ylabel("Magnitude")
plt.xlabel("Date")
plt.gcf().autofmt_xdate()

plt.gca().set_prop_cycle(None)
xlim = plt.xlim()
ylim = plt.ylim()
plt.plot(xlim[0] - 10, ylim[0] - 10, ".", markersize=15, alpha=1.0, label=f"{gamma_label}: {len(gamma_events)}")
plt.plot(xlim[0] - 10, ylim[0] - 10, ".", markersize=15, alpha=1.0, label=f"{standard_label}: {len(standard_catalog)}")
plt.legend(loc="lower right")
plt.xlim(xlim)
plt.ylim(ylim)
plt.grid(linestyle="--", linewidth=0.5, alpha=0.5)
plt.savefig(f"{root_path}/{result_path}/earthquake_magnitude_time.png", bbox_inches="tight", dpi=300)
plt.savefig(f"{root_path}/{result_path}/earthquake_magnitude_time.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%
gamma_markersize = min(2, 1e5 / len(gamma_events))
standard_markersize = min(2, 1e4 / len(standard_catalog))
fig = plt.figure(figsize=plt.rcParams["figure.figsize"] * np.array([0.8, 1.1]))
box = dict(boxstyle="round", facecolor="white", alpha=1)
text_loc = [0.05, 0.90]
plt.subplot(311)
plt.scatter(gamma_events["time"], gamma_events["sigma_time"], s=gamma_markersize, linewidth=0, label="Travel-time")
# plt.ylim([0, 3])
plt.ylabel(r"$\sigma_{11}$ (s)")
plt.legend(loc="upper right")
plt.text(
    text_loc[0],
    text_loc[1],
    "(i)",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
    fontsize="large",
    fontweight="normal",
    bbox=box,
)
plt.subplot(312)
plt.scatter(gamma_events["time"], gamma_events["sigma_amp"], s=gamma_markersize, linewidth=0, label="Amplitude")
# plt.ylim([0, 1])
plt.ylabel(r"$\sigma_{22}$ ($\log10$ m/s)")
plt.legend(loc="upper right")
plt.text(
    text_loc[0],
    text_loc[1],
    "(ii)",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
    fontsize="large",
    fontweight="normal",
    bbox=box,
)
plt.subplot(313)
plt.scatter(
    gamma_events["time"],
    gamma_events["cov_time_amp"],
    s=gamma_markersize,
    linewidth=0,
    label="Travel-time vs. Amplitude",
)
plt.ylabel(r"$\Sigma_{12}$")
plt.ylim([-0.5, 0.5])
plt.legend(loc="upper right")
plt.text(
    text_loc[0],
    text_loc[1],
    "(iii)",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
    fontsize="large",
    fontweight="normal",
    bbox=box,
)
plt.gcf().autofmt_xdate()
# plt.suptitle(r"Covariance Matrix ($\Sigma$) Coefficients")
plt.tight_layout()
plt.gcf().align_labels()
plt.savefig(f"{root_path}/{result_path}/covariance.png", bbox_inches="tight", dpi=300)
plt.savefig(f"{root_path}/{result_path}/covariance.pdf", bbox_inches="tight")
plt.show()


# %%
# plot_standard_catalog = False
gamma_events = gamma_events[gamma_events["gamma_score"] > 30]

fig = plt.figure(figsize=plt.rcParams["figure.figsize"] * np.array([1.5, 1]))
box = dict(boxstyle="round", facecolor="white", alpha=1)
text_loc = [0.05, 0.92]

gamma_markersize = min(5, 1e5 / len(gamma_events["latitude"]))
standard_markersize = min(5, 1e4 / len(standard_catalog["latitude"]))
alpha = 0.3
grd = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1.5, 1], height_ratios=[1, 1])
ax1 = fig.add_subplot(grd[:, 0], projection=ccrs.PlateCarree())
ax1.set_extent(
    [config["minlongitude"], config["maxlongitude"], config["minlatitude"], config["maxlatitude"]],
    crs=ccrs.PlateCarree(),
)
# terrain = cimgt.Stamen("terrain-background")
# ax1.add_image(terrain, 10, alpha=0.4)
topo = (
    pygmt.datasets.load_earth_relief(
        resolution="01s",
        region=[config["minlongitude"], config["maxlongitude"], config["minlatitude"], config["maxlatitude"]],
    ).to_numpy()
    / 1e3
)
topo = np.flipud(topo)
ls = LightSource()
ax1.imshow(
    ls.hillshade(topo, vert_exag=300, dx=1.0, dy=1.0),
    origin="upper",
    extent=[config["minlongitude"], config["maxlongitude"], config["minlatitude"], config["maxlatitude"]],
    cmap="gray",
    alpha=0.5,
    transform=ccrs.PlateCarree(),
)

ax1.coastlines(resolution="10m", color="gray", linewidth=0.5)
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
gl.top_labels = False
gl.right_labels = False
ax1.scatter(gamma_events["longitude"], gamma_events["latitude"], s=gamma_markersize, linewidth=0, alpha=alpha)
if plot_standard_catalog:
    ax1.scatter(
        standard_catalog["longitude"], standard_catalog["latitude"], s=standard_markersize, linewidth=0, alpha=alpha
    )
# ax1.axis("scaled")
# ax1.set_xlim([config["minlongitude"], config["maxlongitude"]])
# ax1.set_ylim([config["minlatitude"], config["maxlatitude"]])
# ax1.set_xlabel("Latitude")
# ax1.set_ylabel("Longitude")
ax1.set_prop_cycle(None)
ax1.plot(
    config["minlongitude"] - 10, config["minlatitude"] - 10, ".", markersize=10, label=f"{gamma_label}", rasterized=True
)
ax1.plot(
    config["minlongitude"] - 10,
    config["minlatitude"] - 10,
    ".",
    markersize=10,
    label=f"{standard_label}",
    rasterized=True,
)
ax1.plot(stations["longitude"], stations["latitude"], "k^", markersize=5, alpha=0.7, label="Stations")
ax1.text(
    text_loc[0],
    text_loc[1],
    "(i)",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
    fontsize="large",
    fontweight="normal",
    bbox=box,
)
plt.legend(loc="lower right")

ax2 = fig.add_subplot(grd[0, 1])
ax2.scatter(
    gamma_events["longitude"],
    gamma_events["depth(m)"] / 1e3,
    s=gamma_markersize,
    linewidth=0,
    alpha=alpha,
    rasterized=True,
)
if plot_standard_catalog:
    ax2.scatter(
        standard_catalog["longitude"],
        standard_catalog["depth_km"],
        s=standard_markersize,
        linewidth=0,
        alpha=alpha,
        rasterized=True,
    )
ax2.set_xlim([config["minlongitude"], config["maxlongitude"]])
ax2.set_ylim([0, 21])
ax2.invert_yaxis()
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Depth (km)")
ax2.set_prop_cycle(None)
ax2.plot(config["minlongitude"] - 10, 31, ".", markersize=10, label=f"{gamma_label}")
ax2.plot(31, 31, ".", markersize=10, label=f"{standard_label}")
ax2.text(
    text_loc[0],
    text_loc[1],
    "(ii)",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
    fontsize="large",
    fontweight="normal",
    bbox=box,
)
plt.legend(loc="lower right")


fig.add_subplot(grd[1, 1])
plt.scatter(
    gamma_events["latitude"],
    gamma_events["depth(m)"] / 1e3,
    s=gamma_markersize,
    linewidth=0,
    alpha=alpha,
    rasterized=True,
)
if plot_standard_catalog:
    plt.scatter(
        standard_catalog["latitude"],
        standard_catalog["depth_km"],
        s=standard_markersize,
        linewidth=0,
        alpha=alpha,
        rasterized=True,
    )
plt.xlim([config["minlatitude"], config["maxlatitude"]])
plt.ylim([0, 21])
plt.gca().invert_yaxis()
plt.xlabel("Latitude")
plt.ylabel("Depth (km)")
plt.gca().set_prop_cycle(None)
plt.plot(config["minlatitude"] - 10, 31, ".", markersize=10, label=f"{gamma_label}")
plt.plot(31, 31, ".", markersize=10, label=f"{standard_label}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.text(
    text_loc[0],
    text_loc[1],
    "(iii)",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
    fontsize="large",
    fontweight="normal",
    bbox=box,
)
plt.savefig(f"{root_path}/{result_path}/earthquake_location.png", bbox_inches="tight", dpi=300)
plt.savefig(f"{root_path}/{result_path}/earthquake_location.pdf", bbox_inches="tight", dpi=300)
plt.show()


# %%
def plot3d(x, y, z, config, fig_name):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=2.0,
                    color=z,
                    cmin=config["gamma"]["zmin_km"],
                    cmax=config["gamma"]["zmax_km"],
                    colorscale="Viridis",
                    opacity=0.6,
                ),
            )
        ],
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=[config["minlongitude"], config["maxlongitude"]],
            ),
            yaxis=dict(nticks=4, range=[config["minlatitude"], config["maxlatitude"]]),
            zaxis=dict(
                nticks=4,
                # range=[z.max(), z.min()],
                range=[config["gamma"]["zmax_km"], config["gamma"]["zmin_km"]],
            ),
            #         aspectratio = dict(x=(xlim[1]-xlim[0])/2, y=(ylim[1]-ylim[0])/2, z=1),
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        margin=dict(r=0, l=0, b=0, t=0),
    )
    fig.write_html(fig_name)


plot3d(
    gamma_events["longitude"],
    gamma_events["latitude"],
    gamma_events["depth(m)"] / 1e3,
    config,
    f"{root_path}/{result_path}/earthquake_location.html",
)
# %%

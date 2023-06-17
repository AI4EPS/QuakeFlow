# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import json
import plotly.graph_objects as go

# %%
region = "Kilauea"
root_path = Path(region)
result_path = root_path / "results"
figure_path = root_path / "figures"
if not result_path.exists():
    result_path.mkdir(parents=True)
if not figure_path.exists():
    figure_path.mkdir(parents=True)
plot_standard_catalog = True
use_pygmt = True

# %%
with open(root_path / "config.json", "r") as f:
    config = json.load(f)
config.pop("channel_priorities", None)
config.pop("location_priorities", None)
print(json.dumps(config, indent=4, sort_keys=True))

# %%
gamma_file = root_path / "gamma" / "gamma_catalog.csv"
gamma_exist = False
if gamma_file.exists():
    gamma_exist = True
    gamma_catalog = pd.read_csv(gamma_file, parse_dates=["time"])
    gamma_catalog["depth_km"] = gamma_catalog["depth(m)"]/1e3

# %%
hypodd_file = root_path / "hypodd" / "hypodd_ct_catalog.txt"
hypodd_ct_exist = False
if hypodd_file.exists():
    hypodd_ct_exist = True
    columns = ["ID", "LAT", "LON", "DEPTH", "X", "Y", "Z", "EX", "EY", "EZ", "YR", "MO", "DY", "HR", "MI", "SC", "MAG", "NCCP", "NCCS", "NCTP", "NCTS", "RCC", "RCT", "CID"]
    catalog_ct_hypodd = pd.read_csv(hypodd_file, delim_whitespace=True, header=None, names=columns, )
    catalog_ct_hypodd["time"] = catalog_ct_hypodd.apply(lambda x: f'{x["YR"]:04.0f}-{x["MO"]:02.0f}-{x["DY"]:02.0f}T{x["HR"]:02.0f}:{x["MI"]:02.0f}:{min(x["SC"], 59.999):05.3f}',axis=1)
    catalog_ct_hypodd["time"] = catalog_ct_hypodd["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
    catalog_ct_hypodd = catalog_ct_hypodd[catalog_ct_hypodd["DEPTH"] != "*********"]
    catalog_ct_hypodd["DEPTH"] = catalog_ct_hypodd["DEPTH"].astype(float)

# %%
hypodd_file = root_path / "hypodd" / "hypodd_cc_catalog.txt"
hypodd_cc_exist = False
if hypodd_file.exists():
    hypodd_cc_exist = True
    columns = ["ID", "LAT", "LON", "DEPTH", "X", "Y", "Z", "EX", "EY", "EZ", "YR", "MO", "DY", "HR", "MI", "SC", "MAG", "NCCP", "NCCS", "NCTP", "NCTS", "RCC", "RCT", "CID"]
    catalog_cc_hypodd = pd.read_csv(hypodd_file, delim_whitespace=True, header=None, names=columns, )
    catalog_cc_hypodd["time"] = catalog_cc_hypodd.apply(lambda x: f'{x["YR"]:04.0f}-{x["MO"]:02.0f}-{x["DY"]:02.0f}T{x["HR"]:02.0f}:{x["MI"]:02.0f}:{min(x["SC"], 59.999):05.3f}',axis=1)
    catalog_cc_hypodd["time"] = catalog_cc_hypodd["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
    catalog_cc_hypodd = catalog_cc_hypodd[catalog_cc_hypodd["DEPTH"] != "*********"]
    catalog_cc_hypodd["DEPTH"] = catalog_cc_hypodd["DEPTH"].astype(float)

# %%
growclust_file = root_path / "growclust" / "growclust_ct_catalog.txt"
growclust_ct_exist = False
if growclust_file.exists():
    growclust_ct_exist = True
    columns = ["yr", "mon", "day", "hr", "min", "sec", "evid", "latR", "lonR", "depR", "mag", "qID", "cID", "nbranch", "qnpair", "qndiffP", "qndiffS", "rmsP", "rmsS", "eh", "ez", "et", "latC", "lonC", "depC"]
    growclust_ct_catalog = pd.read_csv(growclust_file, delim_whitespace=True, header=None, names=columns)
    growclust_ct_catalog["time"] = growclust_ct_catalog.apply(lambda x: f'{x["yr"]:04.0f}-{x["mon"]:02.0f}-{x["day"]:02.0f}T{x["hr"]:02.0f}:{x["min"]:02.0f}:{min(x["sec"], 59.999):05.3f}',axis=1)
    growclust_ct_catalog["time"] = growclust_ct_catalog["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
    growclust_ct_catalog = growclust_ct_catalog[growclust_ct_catalog["nbranch"]> 1]

# %%
growclust_file = root_path / "growclust" / "growclust_cc_catalog.txt"
growclust_cc_exist = False
if growclust_file.exists():
    growclust_cc_exist = True
    columns = ["yr", "mon", "day", "hr", "min", "sec", "evid", "latR", "lonR", "depR", "mag", "qID", "cID", "nbranch", "qnpair", "qndiffP", "qndiffS", "rmsP", "rmsS", "eh", "ez", "et", "latC", "lonC", "depC"]
    growclust_cc_catalog = pd.read_csv(growclust_file, delim_whitespace=True, header=None, names=columns)
    growclust_cc_catalog["time"] = growclust_cc_catalog.apply(lambda x: f'{x["yr"]:04.0f}-{x["mon"]:02.0f}-{x["day"]:02.0f}T{x["hr"]:02.0f}:{x["min"]:02.0f}:{min(x["sec"], 59.999):05.3f}',axis=1)
    growclust_cc_catalog["time"] = growclust_cc_catalog["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
    growclust_cc_catalog = growclust_cc_catalog[growclust_cc_catalog["nbranch"]> 1]

# %%
fig, ax = plt.subplots(3, 2, squeeze=False, figsize=(15,15))
if gamma_exist and (len(gamma_catalog) > 0):
    ax[0, 0].scatter(gamma_catalog["longitude"], gamma_catalog["latitude"], s=800/len(gamma_catalog), alpha=1.0, linewidth=0)
    ax[0, 0].set_title(f"GaMMA: {len(gamma_catalog)}")
    xlim = ax[0, 0].get_xlim()
    ylim = ax[0, 0].get_ylim()
else:
    xlim = None
    ylim = None
if hypodd_ct_exist and (len(catalog_ct_hypodd) > 0):
    ax[1, 0].scatter(catalog_ct_hypodd["LON"], catalog_ct_hypodd["LAT"], s=800/len(catalog_ct_hypodd), alpha=1.0, linewidth=0)
    ax[1, 0].set_title(f"HypoDD (CT): {len(catalog_ct_hypodd)}")
    ax[1, 0].set_xlim(xlim)
    ax[1, 0].set_ylim(ylim)
if hypodd_cc_exist and (len(catalog_cc_hypodd) > 0):
    ax[1, 1].scatter(catalog_cc_hypodd["LON"], catalog_cc_hypodd["LAT"], s=800/len(catalog_cc_hypodd), alpha=1.0, linewidth=0)
    ax[1, 1].set_title(f"HypoDD (CC): {len(catalog_cc_hypodd)}")
    ax[1, 1].set_xlim(xlim)
    ax[1, 1].set_ylim(ylim)
if growclust_ct_exist and (len(growclust_ct_catalog) > 0):
    ax[2, 0].scatter(growclust_ct_catalog["lonR"], growclust_ct_catalog["latR"], s=800/len(growclust_ct_catalog), alpha=1.0, linewidth=0)
    ax[2, 0].set_title(f"GrowClust: {len(growclust_ct_catalog)}")
    ax[2, 0].set_xlim(xlim)
    ax[2, 0].set_ylim(ylim)
if growclust_cc_exist and (len(growclust_cc_catalog) > 0):
    ax[2, 1].scatter(growclust_cc_catalog["lonR"], growclust_cc_catalog["latR"], s=800/len(growclust_cc_catalog), alpha=1.0, linewidth=0)
    ax[2, 1].set_title(f"GrowClust: {len(growclust_cc_catalog)}")
    ax[2, 1].set_xlim(xlim)
    ax[2, 1].set_ylim(ylim)
plt.savefig(figure_path / "catalogs.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(5, 2, squeeze=False, figsize=(15,20))
cmin = 0
cmax = 10
if gamma_exist and (len(gamma_catalog) > 0):
    ax[0, 0].scatter(gamma_catalog["longitude"], gamma_catalog["depth_km"], c=gamma_catalog["depth_km"], s=8000/len(gamma_catalog), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
    ax[0, 0].set_title(f"GaMMA: {len(gamma_catalog)}")
    ax[0, 0].invert_yaxis()
    xlim = ax[0, 0].get_xlim()
    ylim = ax[0, 0].get_ylim()
else:
    xlim = None
    ylim = None
if hypodd_ct_exist and (len(catalog_ct_hypodd) > 0):
    ax[1, 0].scatter(catalog_ct_hypodd["LON"], catalog_ct_hypodd["DEPTH"], c=catalog_ct_hypodd["DEPTH"], s=8000/len(catalog_ct_hypodd), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
    ax[1, 0].set_title(f"HypoDD (CT): {len(catalog_ct_hypodd)}")
    ax[1, 0].set_xlim(xlim)
    ax[1, 0].set_ylim(ylim)
if hypodd_cc_exist and (len(catalog_cc_hypodd) > 0):
    ax[2, 0].scatter(catalog_cc_hypodd["LON"], catalog_cc_hypodd["DEPTH"], c=catalog_cc_hypodd["DEPTH"], s=8000/len(catalog_cc_hypodd), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
    ax[2, 0].set_title(f"HypoDD (CC): {len(catalog_cc_hypodd)}")
    ax[2, 0].set_xlim(xlim)
    ax[2, 0].set_ylim(ylim)
if growclust_ct_exist and (len(growclust_ct_catalog) > 0):
    ax[3, 0].scatter(growclust_ct_catalog["lonR"], growclust_ct_catalog["depR"], c=growclust_ct_catalog["depR"], s=8000/len(growclust_ct_catalog), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
    ax[3, 0].set_title(f"GrowClust (CT): {len(growclust_ct_catalog)}")
    ax[3, 0].set_xlim(xlim)
    ax[3, 0].set_ylim(ylim)
if growclust_cc_exist and (len(growclust_cc_catalog) > 0):
    ax[4, 0].scatter(growclust_cc_catalog["lonR"], growclust_cc_catalog["depR"], c=growclust_cc_catalog["depR"], s=8000/len(growclust_cc_catalog), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
    ax[4, 0].set_title(f"GrowClust (CC): {len(growclust_cc_catalog)}")
    ax[4, 0].set_xlim(xlim)
    ax[4, 0].set_ylim(ylim)

if gamma_exist and (len(gamma_catalog) > 0):
    ax[0, 1].scatter(gamma_catalog["latitude"], gamma_catalog["depth_km"], c=gamma_catalog["depth_km"], s=8000/len(gamma_catalog), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
    ax[0, 1].set_title(f"GaMMA: {len(gamma_catalog)}")
    ax[0, 1].invert_yaxis()
    xlim = ax[0, 1].get_xlim()
    ylim = ax[0, 1].get_ylim()
else:
    xlim = None
    ylim = None
if hypodd_ct_exist and (len(catalog_ct_hypodd) > 0):
    ax[1, 1].scatter(catalog_ct_hypodd["LAT"], catalog_ct_hypodd["DEPTH"], c=catalog_ct_hypodd["DEPTH"], s=8000/len(catalog_ct_hypodd), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
    ax[1, 1].set_title(f"HypoDD (CT): {len(catalog_ct_hypodd)}")
    ax[1, 1].set_xlim(xlim)
    ax[1, 1].set_ylim(ylim)
if hypodd_cc_exist and (len(catalog_cc_hypodd) > 0):
    ax[2, 1].scatter(catalog_cc_hypodd["LAT"], catalog_cc_hypodd["DEPTH"], c=catalog_cc_hypodd["DEPTH"], s=8000/len(catalog_cc_hypodd), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
    ax[2, 1].set_title(f"HypoDD (CC): {len(catalog_cc_hypodd)}")
    ax[2, 1].set_xlim(xlim)
    ax[2, 1].set_ylim(ylim)
if growclust_ct_exist and (len(growclust_ct_catalog) > 0):
    ax[3, 1].scatter(growclust_ct_catalog["latR"], growclust_ct_catalog["depR"], c=growclust_ct_catalog["depR"], s=8000/len(growclust_ct_catalog), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
    ax[3, 1].set_title(f"GrowClust (CT): {len(growclust_ct_catalog)}")
    ax[3, 1].set_xlim(xlim)
    ax[3, 1].set_ylim(ylim)
if growclust_cc_exist and (len(growclust_cc_catalog) > 0):
    ax[4, 1].scatter(growclust_cc_catalog["latR"], growclust_cc_catalog["depR"], c=growclust_cc_catalog["depR"], s=8000/len(growclust_cc_catalog), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
    ax[4, 1].set_title(f"GrowClust (CC): {len(growclust_cc_catalog)}")
    ax[4, 1].set_xlim(xlim)
    ax[4, 1].set_ylim(ylim)
plt.savefig(figure_path / "catalogs_depth.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(15,10))
if gamma_exist: 
    ax[0, 0].plot(gamma_catalog["time"], gamma_catalog["magnitude"], "o", markersize=2, alpha=0.5, label="GaMMA")
    ax[1, 0].plot(gamma_catalog["time"], gamma_catalog["magnitude"], "o", markersize=2, alpha=0.5, label="GaMMA")
if hypodd_ct_exist:
    ax[0, 0].plot(catalog_ct_hypodd["time"], catalog_ct_hypodd["MAG"], "o", markersize=2, alpha=0.5, label="HypoDD (CT)")
if hypodd_cc_exist:
    ax[1, 0].plot(catalog_cc_hypodd["time"], catalog_cc_hypodd["MAG"], "o", markersize=2, alpha=0.5, label="HypoDD (CC)")
if growclust_ct_exist:
    ax[0, 0].plot(growclust_ct_catalog["time"], growclust_ct_catalog["mag"], "o", markersize=2, alpha=0.5, label="GrowClust (CT)")
if growclust_cc_exist:
    ax[1, 0].plot(growclust_cc_catalog["time"], growclust_cc_catalog["mag"], "o", markersize=2, alpha=0.5, label="GrowClust (CC)")
ax[0, 0].legend()
ax[1, 0].legend()
plt.savefig(figure_path / "catalogs_magnitude_time.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(10,10))
xlim = [int(np.floor(gamma_catalog["magnitude"].min())), int(np.ceil(gamma_catalog["magnitude"].max()))]
bins = np.arange(xlim[0], xlim[1]+1, 0.2)
if gamma_exist:
    ax[0, 0].hist(gamma_catalog["magnitude"], bins = bins, alpha=0.5,label="GaMMA")
    ax[1, 0].hist(gamma_catalog["magnitude"], bins = bins, alpha=0.5,label="GaMMA")
if hypodd_ct_exist:
    ax[0, 0].hist(catalog_ct_hypodd["MAG"], bins = bins, alpha=0.5, label="HypoDD (CT)")
if hypodd_cc_exist:
    ax[1, 0].hist(catalog_cc_hypodd["MAG"], bins = bins, alpha=0.5, label="HypoDD (CC)")
if growclust_ct_exist:
    ax[0, 0].hist(growclust_ct_catalog["mag"], bins = bins, alpha=0.5, label="GrowClust (CT)")
if growclust_cc_exist:  
    ax[1, 0].hist(growclust_cc_catalog["mag"], bins = bins, alpha=0.5, label="GrowClust (CC)")
ax[0, 0].set_yscale("log")
ax[0, 0].legend()
ax[1, 0].set_yscale("log")
ax[1, 0].legend()
plt.savefig(figure_path / "catalogs_magnitude_hist.png", dpi=300)
plt.show()


# %%
# %%
def plot3d(x, y, z, config, fig_name):

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=1.0, autocolorscale=False, color=z, cmin=config["zrange"][0], cmax=config["zrange"][1], colorscale="Viridis_r", opacity=0.1),
            )
        ],
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=config["xrange"],
            ),
            yaxis=dict(
                nticks=4,
                range=config["yrange"],
            ),
            zaxis=dict(
                nticks=4,
                range=config["zrange"][::-1],
            ),
            #  aspectratio = dict(x=(xlim[1]-xlim[0])/2, y=(ylim[1]-ylim[0])/2, z=1),
            aspectratio=dict(x=1, y=1, z=0.3),
        ),
        margin=dict(r=0, l=0, b=0, t=0),
    )
    fig.write_html(fig_name)

config_plot3d = {
    "xrange": [config["minlongitude"], config["maxlongitude"]],
    "yrange": [config["minlatitude"], config["maxlatitude"]],
    # "zrange": [config["gamma"]["zmin_km"], config["gamma"]["zmax_km"]],
    "zrange": [0, 6]
}

if gamma_exist and len(gamma_catalog) > 0:
    plot3d(gamma_catalog["longitude"], gamma_catalog["latitude"], gamma_catalog["depth(m)"]/1e3, config_plot3d, result_path / "earthquake_location_gamma.html")

if hypodd_ct_exist and len(catalog_ct_hypodd) > 0:
    plot3d(catalog_ct_hypodd["lonR"], catalog_ct_hypodd["latR"], catalog_ct_hypodd["depR"], config_plot3d, result_path / "earthquake_location_hypodd_ct.html")

if hypodd_cc_exist and len(catalog_cc_hypodd) > 0:
    plot3d(catalog_cc_hypodd["lonR"], catalog_cc_hypodd["latR"], catalog_cc_hypodd["depR"], config_plot3d, result_path / "earthquake_location_hypodd_cc.html")

if growclust_ct_exist and len(growclust_ct_catalog) > 0:
    plot3d(growclust_ct_catalog["lonR"], growclust_ct_catalog["latR"], growclust_ct_catalog["depR"], config_plot3d, result_path / "earthquake_location_growclust_ct.html")

if growclust_cc_exist and len(growclust_cc_catalog) > 0:
    plot3d(growclust_cc_catalog["lonR"], growclust_cc_catalog["latR"], growclust_cc_catalog["depR"], config, result_path / "earthquake_location_growclust_cc.html")

# %%

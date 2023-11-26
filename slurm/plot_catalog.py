# %%
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", nargs="?", type=str, default="local", help="root path")
    parser.add_argument("region", nargs="?", type=str, default="demo", help="region")
    return parser.parse_args()


args = parse_args()

# %%
root_path = args.root_path
region = args.region

result_path = f"{region}/results"
figure_path = f"{region}/figures"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")
if not os.path.exists(f"{root_path}/{figure_path}"):
    os.makedirs(f"{root_path}/{figure_path}")
plot_3d = False

use_pygmt = True

# %%
with open(f"{root_path}/{region}/config.json", "r") as f:
    config = json.load(f)
print(json.dumps(config, indent=4, sort_keys=True))
xlim = [config["minlongitude"], config["maxlongitude"]]
ylim = [config["minlatitude"], config["maxlatitude"]]

# %%
# %%
routine_catalog = f"{root_path}/{region}/obspy/catalog.csv"
routine_exist = False
if os.path.exists(routine_catalog):
    routine_exist = True
    routine_catalog = pd.read_csv(routine_catalog, parse_dates=["time"])


# %%
gamma_file = f"{root_path}/{region}/gamma/gamma_events.csv"
gamma_exist = False
if os.path.exists(gamma_file):
    gamma_exist = True
    gamma_catalog = pd.read_csv(gamma_file, parse_dates=["time"])
    # gamma_catalog["depth_km"] = gamma_catalog["depth(m)"] / 1e3


# %%
adloc_file = f"{root_path}/{region}/adloc/adloc_events.csv"
adloc_exist = False
if os.path.exists(adloc_file):
    adloc_exist = True
    adloc_catalog = pd.read_csv(adloc_file, parse_dates=["time"])
    # gamma_catalog["depth_km"] = gamma_catalog["depth(m)"] / 1e3

# %%
hypodd_file = f"{root_path}/{region}/hypodd/hypodd_ct.reloc"
hypodd_ct_exist = False
if os.path.exists(hypodd_file):
    hypodd_ct_exist = True
    columns = [
        "ID",
        "LAT",
        "LON",
        "DEPTH",
        "X",
        "Y",
        "Z",
        "EX",
        "EY",
        "EZ",
        "YR",
        "MO",
        "DY",
        "HR",
        "MI",
        "SC",
        "MAG",
        "NCCP",
        "NCCS",
        "NCTP",
        "NCTS",
        "RCC",
        "RCT",
        "CID",
    ]
    catalog_ct_hypodd = pd.read_csv(
        hypodd_file,
        delim_whitespace=True,
        header=None,
        names=columns,
    )
    catalog_ct_hypodd["time"] = catalog_ct_hypodd.apply(
        lambda x: f'{x["YR"]:04.0f}-{x["MO"]:02.0f}-{x["DY"]:02.0f}T{x["HR"]:02.0f}:{x["MI"]:02.0f}:{min(x["SC"], 59.999):05.3f}',
        axis=1,
    )
    catalog_ct_hypodd["time"] = catalog_ct_hypodd["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
    catalog_ct_hypodd = catalog_ct_hypodd[catalog_ct_hypodd["DEPTH"] != "*********"]
    catalog_ct_hypodd["DEPTH"] = catalog_ct_hypodd["DEPTH"].astype(float)

# %%
hypodd_file = f"{root_path}/{region}/hypodd/hypodd_cc.reloc"
hypodd_cc_exist = False
if os.path.exists(hypodd_file):
    hypodd_cc_exist = True
    columns = [
        "ID",
        "LAT",
        "LON",
        "DEPTH",
        "X",
        "Y",
        "Z",
        "EX",
        "EY",
        "EZ",
        "YR",
        "MO",
        "DY",
        "HR",
        "MI",
        "SC",
        "MAG",
        "NCCP",
        "NCCS",
        "NCTP",
        "NCTS",
        "RCC",
        "RCT",
        "CID",
    ]
    catalog_cc_hypodd = pd.read_csv(
        hypodd_file,
        delim_whitespace=True,
        header=None,
        names=columns,
    )
    catalog_cc_hypodd["time"] = catalog_cc_hypodd.apply(
        lambda x: f'{x["YR"]:04.0f}-{x["MO"]:02.0f}-{x["DY"]:02.0f}T{x["HR"]:02.0f}:{x["MI"]:02.0f}:{min(x["SC"], 59.999):05.3f}',
        axis=1,
    )
    catalog_cc_hypodd["time"] = catalog_cc_hypodd["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
    catalog_cc_hypodd = catalog_cc_hypodd[catalog_cc_hypodd["DEPTH"] != "*********"]
    catalog_cc_hypodd["DEPTH"] = catalog_cc_hypodd["DEPTH"].astype(float)

# %%
growclust_file = f"{root_path}/{region}/growclust/growclust_ct_catalog.txt"
growclust_ct_exist = False
if os.path.exists(growclust_file):
    growclust_ct_exist = True
    columns = [
        "yr",
        "mon",
        "day",
        "hr",
        "min",
        "sec",
        "evid",
        "latR",
        "lonR",
        "depR",
        "mag",
        "qID",
        "cID",
        "nbranch",
        "qnpair",
        "qndiffP",
        "qndiffS",
        "rmsP",
        "rmsS",
        "eh",
        "ez",
        "et",
        "latC",
        "lonC",
        "depC",
    ]
    growclust_ct_catalog = pd.read_csv(growclust_file, delim_whitespace=True, header=None, names=columns)
    growclust_ct_catalog["time"] = growclust_ct_catalog.apply(
        lambda x: f'{x["yr"]:04.0f}-{x["mon"]:02.0f}-{x["day"]:02.0f}T{x["hr"]:02.0f}:{x["min"]:02.0f}:{min(x["sec"], 59.999):05.3f}',
        axis=1,
    )
    growclust_ct_catalog["time"] = growclust_ct_catalog["time"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f")
    )
    growclust_ct_catalog = growclust_ct_catalog[growclust_ct_catalog["nbranch"] > 1]

# %%
growclust_file = f"{root_path}/{region}/growclust/growclust_cc_catalog.txt"
growclust_cc_exist = False
if os.path.exists(growclust_file):
    growclust_cc_exist = True
    columns = [
        "yr",
        "mon",
        "day",
        "hr",
        "min",
        "sec",
        "evid",
        "latR",
        "lonR",
        "depR",
        "mag",
        "qID",
        "cID",
        "nbranch",
        "qnpair",
        "qndiffP",
        "qndiffS",
        "rmsP",
        "rmsS",
        "eh",
        "ez",
        "et",
        "latC",
        "lonC",
        "depC",
    ]
    growclust_cc_catalog = pd.read_csv(growclust_file, delim_whitespace=True, header=None, names=columns)
    growclust_cc_catalog["time"] = growclust_cc_catalog.apply(
        lambda x: f'{x["yr"]:04.0f}-{x["mon"]:02.0f}-{x["day"]:02.0f}T{x["hr"]:02.0f}:{x["min"]:02.0f}:{min(x["sec"], 59.999):05.3f}',
        axis=1,
    )
    growclust_cc_catalog["time"] = growclust_cc_catalog["time"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f")
    )
    growclust_cc_catalog = growclust_cc_catalog[growclust_cc_catalog["nbranch"] > 1]


# %% Debug
# def load_Shelly2020():
#     if not os.path.exists("Shelly2020.txt"):
#         os.system(
#             "wget -O Shelly2020.txt 'https://gsw.silverchair-cdn.com/gsw/Content_public/Journal/srl/91/4/10.1785_0220190309/3/srl-2019309_supplement_hypos_ridgecrest_srl_header_mnew.txt?Expires=1702369391&Signature=u5ppgVdpwgoZPZ2bpsiUS1Xi9k1JFMoZ3MD5WHJ8XoIb3BG5nzXd2YT3lEJSf~GSJ9Ag7e5nbQYBmdFYKNHVA41Fw8Pf9IuXs1kNVAV98Pkd1uI9xuGFmJBfIzdi9rKYfl~dnoWj7vcxUlmakd8nB3jUs0ZYoVBsGr1xWz1Wd77bkquwY6UKOpa9EftfDanj-NYRvTQdfqYFJzb8uiE15VzfGj53cwCGmTA~vOQwkjjHz5bfjYIxOGvqPX42vnBAXHvUOW-ZAD02bvnFJnHNSbJ~Lj-43QI-k8I9Q-jbuOKEi24x80RTUpzZvB0Ia0XPnXPU2PqDekryQwn3cZLThg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA'"
#         )

#     catalog = pd.read_csv(
#         "Shelly2020.txt",
#         sep="\s+",
#         header=24,
#         names=["yr", "mon", "day", "hr", "min", "sec", "lat", "lon", "dep", "x", "y", "z", "mag", "ID"],
#         dtype=str,
#     )

#     # print(catalog)
#     # raise

#     catalog["date"] = (
#         catalog["yr"]
#         + "-"
#         + catalog["mon"]
#         + "-"
#         + catalog["day"]
#         + "T"
#         + catalog["hr"]
#         + ":"
#         + catalog["min"]
#         + ":"
#         + catalog["sec"]
#     )
#     catalog["date"] = catalog["date"].map(datetime.fromisoformat)
#     catalog["time"] = catalog["date"]
#     catalog["mag"] = catalog["mag"].map(float)
#     catalog["magnitude"] = catalog["mag"].map(float)
#     catalog["latitude"] = catalog["lat"].map(float)
#     catalog["longitude"] = catalog["lon"].map(float)
#     catalog["lonR"] = catalog["lon"].map(float)
#     catalog["latR"] = catalog["lat"].map(float)
#     catalog["depR"] = catalog["dep"].map(float)

#     return catalog


# growclust_ct_exist = True
# growclust_ct_catalog = load_Shelly2020()
# growclust_ct_catalog = growclust_ct_catalog[
#     (
#         (growclust_ct_catalog["time"] > datetime.fromisoformat(config["starttime"]))
#         & (growclust_ct_catalog["time"] < datetime.fromisoformat(config["endtime"]))
#     )
# ]
# print(f"Sheely2020: {len(growclust_ct_catalog)}")


# %%
size_factor = 2600
fig, ax = plt.subplots(3, 2, squeeze=False, figsize=(10, 15), sharex=True, sharey=True)
for i in range(3):
    for j in range(2):
        # ax[i, j].set_xlim(xlim)
        # ax[i, j].set_ylim(ylim)
        # ax[i, j].set_aspect((ylim[1] - ylim[0]) / ((xlim[1] - xlim[0]) * np.cos(np.mean(ylim) * np.pi / 180)))
        # # ax[i, j].set_xlabel("Longitude")
        # # ax[i, j].set_ylabel("Latitude")
        # # ax[i, j].grid()
        ax[i, j].set_xlim([-117.70, -117.45])
        ax[i, j].set_ylim([35.55, 35.80])


if routine_exist and (len(routine_catalog) > 0):
    ax[0, 0].scatter(
        routine_catalog["longitude"],
        routine_catalog["latitude"],
        s=min(2, size_factor / len(routine_catalog)),
        alpha=1.0,
        linewidth=0,
    )
    ax[0, 0].set_title(f"Routine: {len(routine_catalog)}")
    # xlim = ax[0, 0].get_xlim()
    # ylim = ax[0, 0].get_ylim()

if gamma_exist and (len(gamma_catalog) > 0):
    ax[0, 1].scatter(
        gamma_catalog["longitude"],
        gamma_catalog["latitude"],
        s=min(2, size_factor / len(gamma_catalog)),
        alpha=1.0,
        linewidth=0,
        label=f"GaMMA: {len(gamma_catalog)}",
    )
    ax[0, 1].set_title(f"GaMMA: {len(gamma_catalog)}")
    # xlim = ax[0, 1].get_xlim()
    # ylim = ax[0, 1].get_ylim()

if adloc_exist and (len(adloc_catalog) > 0):
    ax[0, 1].scatter(
        adloc_catalog["longitude"],
        adloc_catalog["latitude"],
        s=min(2, size_factor / len(adloc_catalog)),
        alpha=1.0,
        linewidth=0,
        label=f"AdLoc: {len(adloc_catalog)}",
    )
    ax[0, 1].legend()
    # ax[0, 1].set_title(f"AdLoc: {len(adloc_catalog)}")

if hypodd_ct_exist and (len(catalog_ct_hypodd) > 0):
    ax[1, 0].scatter(
        catalog_ct_hypodd["LON"],
        catalog_ct_hypodd["LAT"],
        s=min(2, size_factor / len(catalog_ct_hypodd)),
        alpha=1.0,
        linewidth=0,
    )
    ax[1, 0].set_title(f"HypoDD (CT): {len(catalog_ct_hypodd)}")
    # ax[1, 0].set_xlim(xlim)
    # ax[1, 0].set_ylim(ylim)
    # ax[1, 0].set_aspect((ylim[1] - ylim[0]) / ((xlim[1] - xlim[0]) * np.cos(np.mean(ylim) * np.pi / 180)))

if hypodd_cc_exist and (len(catalog_cc_hypodd) > 0):
    ax[1, 1].scatter(
        catalog_cc_hypodd["LON"],
        catalog_cc_hypodd["LAT"],
        s=min(2, size_factor / len(catalog_cc_hypodd)),
        alpha=1.0,
        linewidth=0,
    )
    ax[1, 1].set_title(f"HypoDD (CC): {len(catalog_cc_hypodd)}")

if growclust_ct_exist and (len(growclust_ct_catalog) > 0):
    ax[2, 0].scatter(
        growclust_ct_catalog["lonR"],
        growclust_ct_catalog["latR"],
        s=min(2, size_factor / len(growclust_ct_catalog)),
        alpha=1.0,
        linewidth=0,
    )
    ax[2, 0].set_title(f"GrowClust: {len(growclust_ct_catalog)}")

if growclust_cc_exist and (len(growclust_cc_catalog) > 0):
    ax[2, 1].scatter(
        growclust_cc_catalog["lonR"],
        growclust_cc_catalog["latR"],
        s=min(2, size_factor / len(growclust_cc_catalog)),
        alpha=1.0,
        linewidth=0,
    )
    ax[2, 1].set_title(f"GrowClust: {len(growclust_cc_catalog)}")

fig.tight_layout()
plt.savefig(f"{root_path}/{figure_path}/catalogs_location.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(5, 2, squeeze=False, figsize=(15, 30))
cmin = 0
cmax = 10
if gamma_exist and (len(gamma_catalog) > 0):
    ax[0, 0].scatter(
        gamma_catalog["longitude"],
        gamma_catalog["depth_km"],
        c=gamma_catalog["depth_km"],
        s=8000 / len(gamma_catalog),
        alpha=1.0,
        linewidth=0,
        vmin=cmin,
        vmax=cmax,
        cmap="viridis_r",
        label=f"GaMMA: {len(gamma_catalog)}",
    )
    ax[0, 0].set_title(f"GaMMA: {len(gamma_catalog)}")
    ax[0, 0].invert_yaxis()
    xlim = ax[0, 0].get_xlim()
    ylim = ax[0, 0].get_ylim()
else:
    xlim = None
    ylim = None

if adloc_exist and (len(adloc_catalog) > 0):
    ax[0, 0].scatter(
        adloc_catalog["longitude"],
        adloc_catalog["depth_km"],
        # c=adloc_catalog["depth_km"],
        s=8000 / len(adloc_catalog),
        alpha=1.0,
        linewidth=0,
        # vmin=cmin,
        # vmax=cmax,
        # cmap="viridis_r",
        label=f"AdLoc: {len(adloc_catalog)}",
    )
    ax[0, 0].legend()
    # ax[0, 0].set_title(f"AdLoc: {len(adloc_catalog)}")
    ax[1, 0].set_xlim(xlim)
    ax[1, 0].set_ylim(ylim)
if hypodd_ct_exist and (len(catalog_ct_hypodd) > 0):
    ax[1, 0].scatter(
        catalog_ct_hypodd["LON"],
        catalog_ct_hypodd["DEPTH"],
        c=catalog_ct_hypodd["DEPTH"],
        s=8000 / len(catalog_ct_hypodd),
        alpha=1.0,
        linewidth=0,
        vmin=cmin,
        vmax=cmax,
        cmap="viridis_r",
    )
    ax[1, 0].set_title(f"HypoDD (CT): {len(catalog_ct_hypodd)}")
    ax[1, 0].set_xlim(xlim)
    ax[1, 0].set_ylim(ylim)
if hypodd_cc_exist and (len(catalog_cc_hypodd) > 0):
    ax[2, 0].scatter(
        catalog_cc_hypodd["LON"],
        catalog_cc_hypodd["DEPTH"],
        c=catalog_cc_hypodd["DEPTH"],
        s=8000 / len(catalog_cc_hypodd),
        alpha=1.0,
        linewidth=0,
        vmin=cmin,
        vmax=cmax,
        cmap="viridis_r",
    )
    ax[2, 0].set_title(f"HypoDD (CC): {len(catalog_cc_hypodd)}")
    ax[2, 0].set_xlim(xlim)
    ax[2, 0].set_ylim(ylim)
if growclust_ct_exist and (len(growclust_ct_catalog) > 0):
    ax[3, 0].scatter(
        growclust_ct_catalog["lonR"],
        growclust_ct_catalog["depR"],
        c=growclust_ct_catalog["depR"],
        s=8000 / len(growclust_ct_catalog),
        alpha=1.0,
        linewidth=0,
        vmin=cmin,
        vmax=cmax,
        cmap="viridis_r",
    )
    ax[3, 0].set_title(f"GrowClust (CT): {len(growclust_ct_catalog)}")
    ax[3, 0].set_xlim(xlim)
    ax[3, 0].set_ylim(ylim)
if growclust_cc_exist and (len(growclust_cc_catalog) > 0):
    ax[4, 0].scatter(
        growclust_cc_catalog["lonR"],
        growclust_cc_catalog["depR"],
        c=growclust_cc_catalog["depR"],
        s=8000 / len(growclust_cc_catalog),
        alpha=1.0,
        linewidth=0,
        vmin=cmin,
        vmax=cmax,
        cmap="viridis_r",
    )
    ax[4, 0].set_title(f"GrowClust (CC): {len(growclust_cc_catalog)}")
    ax[4, 0].set_xlim(xlim)
    ax[4, 0].set_ylim(ylim)

if gamma_exist and (len(gamma_catalog) > 0):
    ax[0, 1].scatter(
        gamma_catalog["latitude"],
        gamma_catalog["depth_km"],
        c=gamma_catalog["depth_km"],
        s=8000 / len(gamma_catalog),
        alpha=1.0,
        linewidth=0,
        vmin=cmin,
        vmax=cmax,
        cmap="viridis_r",
        label=f"GaMMA: {len(gamma_catalog)}",
    )
    ax[0, 1].set_title(f"GaMMA: {len(gamma_catalog)}")
    ax[0, 1].invert_yaxis()
    xlim = ax[0, 1].get_xlim()
    ylim = ax[0, 1].get_ylim()
else:
    xlim = None
    ylim = None
if adloc_exist and (len(adloc_catalog) > 0):
    ax[0, 1].scatter(
        adloc_catalog["latitude"],
        adloc_catalog["depth_km"],
        # c=adloc_catalog["depth_km"],
        s=8000 / len(adloc_catalog),
        alpha=1.0,
        linewidth=0,
        # vmin=cmin,
        # vmax=cmax,
        # cmap="viridis_r",
        label=f"AdLoc: {len(adloc_catalog)}",
    )
    ax[0, 1].legend()
    # ax[0, 1].set_title(f"AdLoc: {len(adloc_catalog)}")
    ax[0, 1].set_xlim(xlim)
    ax[0, 1].set_ylim(ylim)
if hypodd_ct_exist and (len(catalog_ct_hypodd) > 0):
    ax[1, 1].scatter(
        catalog_ct_hypodd["LAT"],
        catalog_ct_hypodd["DEPTH"],
        c=catalog_ct_hypodd["DEPTH"],
        s=8000 / len(catalog_ct_hypodd),
        alpha=1.0,
        linewidth=0,
        vmin=cmin,
        vmax=cmax,
        cmap="viridis_r",
    )
    ax[1, 1].set_title(f"HypoDD (CT): {len(catalog_ct_hypodd)}")
    ax[1, 1].set_xlim(xlim)
    ax[1, 1].set_ylim(ylim)
if hypodd_cc_exist and (len(catalog_cc_hypodd) > 0):
    ax[2, 1].scatter(
        catalog_cc_hypodd["LAT"],
        catalog_cc_hypodd["DEPTH"],
        c=catalog_cc_hypodd["DEPTH"],
        s=8000 / len(catalog_cc_hypodd),
        alpha=1.0,
        linewidth=0,
        vmin=cmin,
        vmax=cmax,
        cmap="viridis_r",
    )
    ax[2, 1].set_title(f"HypoDD (CC): {len(catalog_cc_hypodd)}")
    ax[2, 1].set_xlim(xlim)
    ax[2, 1].set_ylim(ylim)
if growclust_ct_exist and (len(growclust_ct_catalog) > 0):
    ax[3, 1].scatter(
        growclust_ct_catalog["latR"],
        growclust_ct_catalog["depR"],
        c=growclust_ct_catalog["depR"],
        s=8000 / len(growclust_ct_catalog),
        alpha=1.0,
        linewidth=0,
        vmin=cmin,
        vmax=cmax,
        cmap="viridis_r",
    )
    ax[3, 1].set_title(f"GrowClust (CT): {len(growclust_ct_catalog)}")
    ax[3, 1].set_xlim(xlim)
    ax[3, 1].set_ylim(ylim)
if growclust_cc_exist and (len(growclust_cc_catalog) > 0):
    ax[4, 1].scatter(
        growclust_cc_catalog["latR"],
        growclust_cc_catalog["depR"],
        c=growclust_cc_catalog["depR"],
        s=8000 / len(growclust_cc_catalog),
        alpha=1.0,
        linewidth=0,
        vmin=cmin,
        vmax=cmax,
        cmap="viridis_r",
    )
    ax[4, 1].set_title(f"GrowClust (CC): {len(growclust_cc_catalog)}")
    ax[4, 1].set_xlim(xlim)
    ax[4, 1].set_ylim(ylim)
plt.savefig(f"{root_path}/{figure_path}/catalogs_location_depth.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(15, 10))
if gamma_exist:
    ax[0, 0].plot(gamma_catalog["time"], gamma_catalog["magnitude"], "o", markersize=2, alpha=0.5, label="GaMMA")
    ax[1, 0].plot(gamma_catalog["time"], gamma_catalog["magnitude"], "o", markersize=2, alpha=0.5, label="GaMMA")
if adloc_exist:
    ax[0, 0].plot(adloc_catalog["time"], adloc_catalog["magnitude"], "o", markersize=2, alpha=0.5, label="AdLoc")
    ax[1, 0].plot(adloc_catalog["time"], adloc_catalog["magnitude"], "o", markersize=2, alpha=0.5, label="AdLoc")
if hypodd_ct_exist:
    ax[0, 0].plot(
        catalog_ct_hypodd["time"], catalog_ct_hypodd["MAG"], "o", markersize=2, alpha=0.5, label="HypoDD (CT)"
    )
if hypodd_cc_exist:
    ax[1, 0].plot(
        catalog_cc_hypodd["time"], catalog_cc_hypodd["MAG"], "o", markersize=2, alpha=0.5, label="HypoDD (CC)"
    )
if growclust_ct_exist:
    ax[0, 0].plot(
        growclust_ct_catalog["time"], growclust_ct_catalog["mag"], "o", markersize=2, alpha=0.5, label="GrowClust (CT)"
    )
if growclust_cc_exist:
    ax[1, 0].plot(
        growclust_cc_catalog["time"], growclust_cc_catalog["mag"], "o", markersize=2, alpha=0.5, label="GrowClust (CC)"
    )
ax[0, 0].legend()
ax[1, 0].legend()
plt.savefig(f"{root_path}/{figure_path}/catalogs_magnitude_time.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(10, 10))
xlim = [int(np.floor(gamma_catalog["magnitude"].min())), int(np.ceil(gamma_catalog["magnitude"].max()))]
bins = np.arange(xlim[0], xlim[1] + 1, 0.2)
if gamma_exist:
    ax[0, 0].hist(gamma_catalog["magnitude"], bins=bins, alpha=0.5, label="GaMMA")
    ax[1, 0].hist(gamma_catalog["magnitude"], bins=bins, alpha=0.5, label="GaMMA")
if adloc_exist:
    ax[0, 0].hist(adloc_catalog["magnitude"], bins=bins, alpha=0.5, label="AdLoc")
    ax[1, 0].hist(adloc_catalog["magnitude"], bins=bins, alpha=0.5, label="AdLoc")
if hypodd_ct_exist:
    ax[0, 0].hist(catalog_ct_hypodd["MAG"], bins=bins, alpha=0.5, label="HypoDD (CT)")
if hypodd_cc_exist:
    ax[1, 0].hist(catalog_cc_hypodd["MAG"], bins=bins, alpha=0.5, label="HypoDD (CC)")
if growclust_ct_exist:
    ax[0, 0].hist(growclust_ct_catalog["mag"], bins=bins, alpha=0.5, label="GrowClust (CT)")
if growclust_cc_exist:
    ax[1, 0].hist(growclust_cc_catalog["mag"], bins=bins, alpha=0.5, label="GrowClust (CC)")
ax[0, 0].set_yscale("log")
ax[0, 0].legend()
ax[1, 0].set_yscale("log")
ax[1, 0].legend()
plt.savefig(f"{root_path}/{figure_path}/catalogs_magnitude_histogram.png", dpi=300)
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
                marker=dict(
                    size=1.0,
                    autocolorscale=False,
                    color=z,
                    cmin=config["zrange"][0],
                    cmax=config["zrange"][1],
                    colorscale="Viridis_r",
                    opacity=0.1,
                ),
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


if plot3d:
    config_plot3d = {
        "xrange": [config["minlongitude"], config["maxlongitude"]],
        "yrange": [config["minlatitude"], config["maxlatitude"]],
        # "zrange": [config["gamma"]["zmin_km"], config["gamma"]["zmax_km"]],
        "zrange": [0, 6],
    }

    if gamma_exist and len(gamma_catalog) > 0:
        plot3d(
            gamma_catalog["longitude"],
            gamma_catalog["latitude"],
            # gamma_catalog["depth(m)"] / 1e3,
            gamma_catalog["depth_km"],
            config_plot3d,
            f"{root_path}/{result_path}/earthquake_location_gamma.html",
        )

    if adloc_exist and len(adloc_catalog) > 0:
        plot3d(
            adloc_catalog["longitude"],
            adloc_catalog["latitude"],
            # gamma_catalog["depth(m)"] / 1e3,
            adloc_catalog["depth_km"],
            config_plot3d,
            f"{root_path}/{result_path}/earthquake_location_adloc.html",
        )
    if hypodd_ct_exist and len(catalog_ct_hypodd) > 0:
        plot3d(
            catalog_ct_hypodd["LON"],
            catalog_ct_hypodd["LAT"],
            catalog_ct_hypodd["DEPTH"],
            config_plot3d,
            f"{root_path}/{result_path}/earthquake_location_hypodd_ct.html",
        )

    if hypodd_cc_exist and len(catalog_cc_hypodd) > 0:
        plot3d(
            catalog_cc_hypodd["LON"],
            catalog_cc_hypodd["LAT"],
            catalog_cc_hypodd["DEPTH"],
            config_plot3d,
            f"{root_path}/{result_path}/earthquake_location_hypodd_cc.html",
        )

    if growclust_ct_exist and len(growclust_ct_catalog) > 0:
        plot3d(
            growclust_ct_catalog["lonR"],
            growclust_ct_catalog["latR"],
            growclust_ct_catalog["depR"],
            config_plot3d,
            f"{root_path}/{result_path}/earthquake_location_growclust_ct.html",
        )

    if growclust_cc_exist and len(growclust_cc_catalog) > 0:
        plot3d(
            growclust_cc_catalog["lonR"],
            growclust_cc_catalog["latR"],
            growclust_cc_catalog["depR"],
            config_plot3d,
            f"{root_path}/{result_path}/earthquake_location_growclust_cc.html",
        )

# %%

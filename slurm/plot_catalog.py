# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# %%
result_path = Path("./results")

gamma_file = result_path / "gamma_catalog.csv"
gamma_catalog = pd.read_csv(gamma_file, parse_dates=["time"])
gamma_catalog["depth_km"] = gamma_catalog["depth(m)"]/1e3

# %%
hypodd_file = result_path / "hypodd_ct_catalog.txt"
columns = ["ID", "LAT", "LON", "DEPTH", "X", "Y", "Z", "EX", "EY", "EZ", "YR", "MO", "DY", "HR", "MI", "SC", "MAG", "NCCP", "NCCS", "NCTP", "NCTS", "RCC", "RCT", "CID"]
catalog_hypodd_ct = pd.read_csv(hypodd_file, delim_whitespace=True, header=None, names=columns, )
catalog_hypodd_ct["time"] = catalog_hypodd_ct.apply(lambda x: f'{x["YR"]:04.0f}-{x["MO"]:02.0f}-{x["DY"]:02.0f}T{x["HR"]:02.0f}:{x["MI"]:02.0f}:{min(x["SC"], 59.999):05.3f}',axis=1)
catalog_hypodd_ct["time"] = catalog_hypodd_ct["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))

# %%
hypodd_file = result_path / "hypodd_cc_catalog.txt"
columns = ["ID", "LAT", "LON", "DEPTH", "X", "Y", "Z", "EX", "EY", "EZ", "YR", "MO", "DY", "HR", "MI", "SC", "MAG", "NCCP", "NCCS", "NCTP", "NCTS", "RCC", "RCT", "CID"]
catalog_hypodd_cc = pd.read_csv(hypodd_file, delim_whitespace=True, header=None, names=columns, )
catalog_hypodd_cc["time"] = catalog_hypodd_cc.apply(lambda x: f'{x["YR"]:04.0f}-{x["MO"]:02.0f}-{x["DY"]:02.0f}T{x["HR"]:02.0f}:{x["MI"]:02.0f}:{min(x["SC"], 59.999):05.3f}',axis=1)
catalog_hypodd_cc["time"] = catalog_hypodd_cc["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))

# %%
growclust_file = result_path / "growclust_catalog.txt"
columns = ["yr", "mon", "day", "hr", "min", "sec", "evid", "latR", "lonR", "depR", "mag", "qID", "cID", "nbranch", "qnpair", "qndiffP", "qndiffS", "rmsP", "rmsS", "eh", "ez", "et", "latC", "lonC", "depC"]
growclust_catalog = pd.read_csv(growclust_file, delim_whitespace=True, header=None, names=columns)
growclust_catalog["time"] = growclust_catalog.apply(lambda x: f'{x["yr"]:04.0f}-{x["mon"]:02.0f}-{x["day"]:02.0f}T{x["hr"]:02.0f}:{x["min"]:02.0f}:{min(x["sec"], 59.999):05.3f}',axis=1)
growclust_catalog["time"] = growclust_catalog["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
growclust_catalog = growclust_catalog[growclust_catalog["nbranch"]> 1]

# %%
fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(15,15))
ax[0, 0].scatter(gamma_catalog["longitude"], gamma_catalog["latitude"], s=800/len(gamma_catalog), alpha=1.0, linewidth=0)
ax[0, 0].set_title(f"GaMMA: {len(gamma_catalog)}")
xlim = ax[0, 0].get_xlim()
ylim = ax[0, 0].get_ylim()
ax[0, 1].scatter(catalog_hypodd_ct["LON"], catalog_hypodd_ct["LAT"], s=500/len(catalog_hypodd_ct), alpha=1.0, linewidth=0)
ax[0, 1].set_title(f"HypoDD (CT): {len(catalog_hypodd_ct)}")
ax[0, 1].set_xlim(xlim)
ax[0, 1].set_ylim(ylim)
# ax[1, 0].scatter(catalog_hypodd_cc["LON"], catalog_hypodd_cc["LAT"], s=500/len(catalog_hypodd_cc), alpha=1.0, linewidth=0)
# ax[1, 0].set_title(f"HypoDD (CC): {len(catalog_hypodd_cc)}")
# ax[1, 0].set_xlim(xlim)
# ax[1, 0].set_ylim(ylim)
ax[1, 1].scatter(growclust_catalog["lonR"], growclust_catalog["latR"], s=800/len(growclust_catalog), alpha=1.0, linewidth=0)
ax[1, 1].set_title(f"GrowClust: {len(growclust_catalog)}")
ax[1, 1].set_xlim(xlim)
ax[1, 1].set_ylim(ylim)
plt.savefig("catalogs.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(4, 2, squeeze=False, figsize=(15,15))
cmin = 0
cmax = 10
ax[0, 0].scatter(gamma_catalog["longitude"], gamma_catalog["depth_km"], c=gamma_catalog["depth_km"], s=8000/len(gamma_catalog), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
ax[0, 0].set_title(f"GaMMA: {len(gamma_catalog)}")
ax[0, 0].invert_yaxis()
xlim = ax[0, 0].get_xlim()
ylim = ax[0, 0].get_ylim()
ax[0, 1].scatter(catalog_hypodd_ct["LON"], catalog_hypodd_ct["DEPTH"], c=catalog_hypodd_ct["DEPTH"], s=8000/len(catalog_hypodd_ct), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
ax[0, 1].set_title(f"HypoDD (CT): {len(catalog_hypodd_ct)}")
ax[0, 1].set_xlim(xlim)
ax[0, 1].set_ylim(ylim)
# ax[1, 0].scatter(catalog_hypodd_cc["LON"], catalog_hypodd_cc["DEPTH"], c=catalog_hypodd_cc["DEPTH"], s=8000/len(catalog_hypodd_cc), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
# ax[1, 0].set_title(f"HypoDD (CC): {len(catalog_hypodd_cc)}")
# ax[1, 0].set_xlim(xlim)
# ax[1, 0].set_ylim(ylim)
ax[1, 1].scatter(growclust_catalog["lonR"], growclust_catalog["depR"], c=growclust_catalog["depR"], s=8000/len(growclust_catalog), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
ax[1, 1].set_title(f"GrowClust: {len(growclust_catalog)}")
ax[1, 1].set_xlim(xlim)
ax[1, 1].set_ylim(ylim)

ax[2, 0].scatter(gamma_catalog["latitude"], gamma_catalog["depth_km"], c=gamma_catalog["depth_km"], s=8000/len(gamma_catalog), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
ax[2, 0].set_title(f"GaMMA: {len(gamma_catalog)}")
ax[2, 0].invert_yaxis()
xlim = ax[2, 0].get_xlim()
ylim = ax[2, 0].get_ylim()
ax[2, 1].scatter(catalog_hypodd_ct["LAT"], catalog_hypodd_ct["DEPTH"], c=catalog_hypodd_ct["DEPTH"], s=8000/len(catalog_hypodd_ct), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
ax[2, 1].set_title(f"HypoDD (CT): {len(catalog_hypodd_ct)}")
ax[2, 1].set_xlim(xlim)
ax[2, 1].set_ylim(ylim)
# ax[3, 0].scatter(catalog_hypodd_cc["LAT"], catalog_hypodd_cc["DEPTH"], c=catalog_hypodd_cc["DEPTH"], s=8000/len(catalog_hypodd_cc), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
# ax[3, 0].set_title(f"HypoDD (CC): {len(catalog_hypodd_cc)}")
# ax[3, 0].set_xlim(xlim)
# ax[3, 0].set_ylim(ylim)
ax[3, 1].scatter(growclust_catalog["latR"], growclust_catalog["depR"], c=growclust_catalog["depR"], s=8000/len(growclust_catalog), alpha=1.0, linewidth=0, vmin=cmin, vmax=cmax, cmap="viridis_r")
ax[3, 1].set_title(f"GrowClust: {len(growclust_catalog)}")
ax[3, 1].set_xlim(xlim)
ax[3, 1].set_ylim(ylim)

plt.savefig("catalogs_depth.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(15,5))
ax[0, 0].plot(gamma_catalog["time"], gamma_catalog["magnitude"], "o", markersize=2, alpha=0.5, label="GaMMA")
ax[0, 0].plot(catalog_hypodd_ct["time"], catalog_hypodd_ct["MAG"], "o", markersize=2, alpha=0.5, label="HypoDD (CT)")
# ax[0, 0].plot(catalog_hypodd_cc["time"], catalog_hypodd_cc["MAG"], "o", markersize=2, alpha=0.5, label="HypoDD (CC)")
# ax[0, 0].plot(growclust_catalog["time"], growclust_catalog["mag"], "o", markersize=2, alpha=0.5, label="GrowClust")
ax[0, 0].legend()
plt.savefig("catalogs_magnitude_time.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10,5))
xlim = [int(np.floor(gamma_catalog["magnitude"].min())), int(np.ceil(gamma_catalog["magnitude"].max()))]
bins = np.arange(xlim[0], xlim[1]+1, 0.2)
ax[0, 0].hist(gamma_catalog["magnitude"], bins = bins, alpha=0.5,label="GaMMA")
xlim = ax[0, 0].get_xlim() 
ax[0, 0].hist(catalog_hypodd_ct["MAG"], bins = bins, alpha=0.5, label="HypoDD (CT)")
# ax[0, 0].hist(catalog_hypodd_cc["MAG"], bins = bins, alpha=0.5, label="HypoDD (CC)")
# ax[0, 0].hist(growclust_catalog["mag"], bins = bins, alpha=0.5, label="GrowClust")
ax[0, 0].set_yscale("log")
ax[0, 0].legend()
plt.savefig("catalogs_magnitude_hist.png", dpi=300)
plt.show()
# %%

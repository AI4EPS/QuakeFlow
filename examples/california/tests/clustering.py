# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# %%
v = 6.0

# scale_x = 4
# eps = 8
# min_samples = 8

scale_x = 1
eps = 10
min_samples = 3

# %%
picks = pd.read_csv("phasenet_picks_20230919_1h.csv", parse_dates=["phase_time"])
# picks = pd.read_csv("phasenet_picks_20230820_1h.csv", parse_dates=["phase_time"])

# %%
stations = pd.read_json("stations.json", orient="index")
stations["station_id"] = stations.index

# %%
picks = picks.merge(stations, on="station_id")
picks["t_s"] = (picks["phase_time"] - picks["phase_time"].min()).dt.total_seconds()

# %%
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(picks[["t_s", "x_km", "y_km"]].values / np.array([1, scale_x * v, v]))

# %%
picks["cluster"] = dbscan.labels_
# %%
mapping_color = lambda x: f"C{x}" if x >= 0 else "black"
plt.figure(figsize=(20, 5))
plt.scatter(picks["t_s"], picks["x_km"], c=picks["cluster"].apply(mapping_color), s=0.3)
plt.title(f"Number of picks: {len(picks)}")
plt.show()

# %%
picks_selected = picks.copy()
dbscan = DBSCAN(eps=1, min_samples=1)
dbscan.fit(picks_selected[["t_s", "x_km", "y_km"]].values / np.array([1, scale_x * v, v]))
picks_selected["cluster"] = dbscan.labels_
picks_selected = (
    picks_selected.groupby("cluster").agg({"t_s": "median", "x_km": "median", "y_km": "median"}).reset_index()
)

# %%
print(f"{len(picks) = }, {len(picks_selected) = }")

# %%
# mapping_color = lambda x: f"C{x}" if x >= 0 else "black"
# plt.figure(figsize=(20, 5))
# plt.scatter(picks_selected["t_s"], picks_selected["x_km"], c=picks_selected["cluster"].apply(mapping_color), s=0.3)
# plt.title(f"Number of picks: {len(picks_selected)}")
# plt.show()

# %%
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(picks_selected[["t_s", "x_km", "y_km"]].values / np.array([1, scale_x * v, v]))

# %%
picks_selected["cluster"] = dbscan.labels_
# %%
mapping_color = lambda x: f"C{x}" if x >= 0 else "black"
plt.figure(figsize=(20, 5))
plt.scatter(picks_selected["t_s"], picks_selected["x_km"], c=picks_selected["cluster"].apply(mapping_color), s=0.3)
plt.title(f"Number of picks: {len(picks_selected)}")
plt.show()

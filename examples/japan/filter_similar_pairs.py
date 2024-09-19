# %%
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# %%
root_path = "local"
region = "hinet"
data_path = f"{root_path}/{region}/cctorch"
result_path = f"{root_path}/{region}/qtm"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Load the datasets
events = pd.read_csv(f"{data_path}/cctorch_events.csv")
picks = pd.read_csv(f"{data_path}/cctorch_picks.csv")
pairs = pd.read_csv(f"{data_path}/ccpairs/CC_002.csv")
print(f"Events: {events.shape}, Picks: {picks.shape}, Pairs: {pairs.shape}")

# basic filtering
events = events[(events["num_picks"] > 12) & (events["adloc_score"] > 0.9)]
picks = picks[picks["idx_eve"].isin(events["idx_eve"])]
pairs = pairs[pairs["idx_eve1"].isin(events["idx_eve"]) & pairs["idx_eve2"].isin(events["idx_eve"])]
print(f"Events: {events.shape}, Picks: {picks.shape}, Pairs: {pairs.shape}")

# %%
# Step 1: Calculate mean CC values, filter for CC > 0.9, and create distance matrix
mean_cc = pairs.groupby(["idx_eve1", "idx_eve2"])["cc"].median().reset_index()
neigh_cc = mean_cc[mean_cc["cc"] > 0.9].copy()
neigh_cc["distance"] = 1 - neigh_cc["cc"]

# Ensure distance matrix includes all events (even those without high CC values)
all_events = np.union1d(neigh_cc["idx_eve1"], neigh_cc["idx_eve2"])
distance_matrix = pd.DataFrame(np.ones((len(all_events), len(all_events))), index=all_events, columns=all_events)

# Populate the distance matrix with valid distances from neigh_cc
for _, row in neigh_cc.iterrows():
    distance_matrix.loc[row["idx_eve1"], row["idx_eve2"]] = row["distance"]

# Symmetrize the matrix
distance_matrix = np.minimum(distance_matrix, distance_matrix.T)

# Set diagonal to 0 (distance of event to itself)
np.fill_diagonal(distance_matrix.values, 0)

# %%
# Step 2: Apply DBSCAN
dbscan = DBSCAN(metric="precomputed", eps=0.1, min_samples=2)
clusters = dbscan.fit_predict(distance_matrix)

# %%
# Step 3: Map events to clusters and find neighbors
cluster_dict = dict(zip(distance_matrix.index, clusters))
neighbors = defaultdict(list)

for idx, cluster_id in cluster_dict.items():
    if cluster_id == -1:  # Ignore noise
        continue
    # Count the number of neighbors (events with CC > 0.9)
    subset = neigh_cc[(neigh_cc["idx_eve1"] == idx) | (neigh_cc["idx_eve2"] == idx)]
    num_neighbors = subset["cc"].count()
    neighbors[cluster_id].append((idx, num_neighbors))

# For each cluster, select the event with the largest number of neighbors
selected_events = {cluster: max(event_list, key=lambda x: x[1])[0] for cluster, event_list in neighbors.items()}

# %%
# Step 4: Map the filtered `events` and `picks` based on the `selected_events`
# We will first create a mapping of the key events to their respective clusters
event_to_key_event = {}
for cluster, key_event in selected_events.items():
    for idx, _ in neighbors[cluster]:
        event_to_key_event[idx] = key_event

# %%
# Step 5: Filter Events by `idx_eve`, keeping the one with the largest `num_picks`
# Map `idx_eve` to the key event (to map neighbors to key events)
# events["mapped_idx_eve"] = events["idx_eve"].map(event_to_key_event)
events["mapped_idx_eve"] = events["idx_eve"].map(lambda x: event_to_key_event.get(x, x))

# %%
# Now filter events by mapped `idx_eve` (key events), keeping the one with the largest `num_picks`
filtered_events = events.loc[events.groupby("mapped_idx_eve")["num_picks"].idxmax()]

# Step 6: Filter Picks by `(idx_eve, idx_sta, phase_type)`, keeping the one with the largest `phase_score`
# Map `idx_eve` in picks to the key event (to map neighbors to key events)
# picks["mapped_idx_eve"] = picks["idx_eve"].map(event_to_key_event)
picks["mapped_idx_eve"] = picks["idx_eve"].map(lambda x: event_to_key_event.get(x, x))

# Now filter picks by mapped `idx_eve`, `idx_sta`, `phase_type`, keeping the one with the largest `phase_score`
filtered_picks = picks.loc[picks.groupby(["mapped_idx_eve", "idx_sta", "phase_type"])["phase_score"].idxmax()]


print(f"Filtered Events: {filtered_events.shape}, Filtered Picks: {filtered_picks.shape}")

# Save the results to files
filtered_events.to_csv(f"{result_path}/qtm_events.csv", index=False)
filtered_picks.to_csv(f"{result_path}/qtm_picks.csv", index=False)


# %%
plt.figure(figsize=(10, 10))
plt.scatter(events["longitude"], events["latitude"], s=1, c="blue", label="All Events")
plt.scatter(
    filtered_events["longitude"],
    filtered_events["latitude"],
    s=1,
    c="red",
    marker="x",
    label="Filtered Events",
)
plt.legend()
plt.savefig(f"{result_path}/filtered_events.png")
# %%
plt.figure(figsize=(10, 10))
plt.hist(events["adloc_score"], bins=100)
# %%

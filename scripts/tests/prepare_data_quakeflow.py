# %%
import os
import pandas as pd
import numpy
import matplotlib.pyplot as plt

# %%
events = pd.read_csv("../local/Ridgecrest/adloc/ransac_events_sst.csv")
events["event_id"] = events["event_index"].astype(str)

plt.figure(figsize=(10, 10))
plt.scatter(events["longitude"], events["latitude"], s=0.1, linewidths=0.0)
plt.show()


events = events[
    (events["latitude"] >= 35.57)
    & (events["latitude"] <= 35.62)
    & (events["longitude"] >= -117.47)
    & (events["longitude"] <= -117.36)
]

plt.figure(figsize=(10, 10))
plt.scatter(events["longitude"], events["latitude"], s=0.5)
plt.title(f"Number of events: {len(events)}")

# events = events[
#     (events["latitude"] >= 35.585)
#     & (events["latitude"] <= 35.592)
#     & (events["longitude"] >= -117.42)
#     & (events["longitude"] <= -117.41)
# ]

plt.scatter(events["longitude"], events["latitude"], s=0.5)
# plt.title(f"Number of events: {len(events)}")
plt.show()

# %%
events.to_csv("events.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")

# %%

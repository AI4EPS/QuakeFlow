# %%
import os
import pandas as pd
import numpy
import matplotlib.pyplot as plt

# %%
catalog_url = "https://www.sciencebase.gov/catalog/file/get/5dd715f3e4b0695797650d18?f=__disk__db%2F88%2Fa1%2Fdb88a1f6754843800f25bd63712ed438dfa7699f"
os.system(f"curl -o catalog.txt {catalog_url}")

# %%
events = pd.read_csv(
    "catalog.txt",
    sep="\s+",
    comment="#",
    header=None,
    names=[
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
        "event_index",
    ],
)
events["time"] = pd.to_datetime(events[["year", "month", "day", "hour", "minute", "second"]])
# events["event_index"] = numpy.arange(len(events))
events.drop(columns=["year", "month", "day", "hour", "minute", "second"], inplace=True, errors="ignore")

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

events = events[
    (events["latitude"] >= 35.585)
    & (events["latitude"] <= 35.592)
    & (events["longitude"] >= -117.42)
    & (events["longitude"] <= -117.41)
]

plt.scatter(events["longitude"], events["latitude"], s=0.5)
# plt.title(f"Number of events: {len(events)}")
plt.show()

# %%
events.to_csv("events.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")

# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

np.random.seed(42)

# %%
result_path = "hypodd"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# %%
stations = pd.read_csv("cctorch/cctorch_stations.csv")

station_lines = {}
for i, row in stations.iterrows():
    station_id = row["station_id"]
    network_code, station_code, comp_code, channel_code = station_id.split(".")
    # tmp_code = f"{station_code}{channel_code}"
    tmp_code = f"{station_code}"
    station_lines[tmp_code] = f"{tmp_code:<8s} {row['latitude']:.3f} {row['longitude']:.3f}\n"


with open(f"{result_path}/stations.dat", "w") as f:
    for line in sorted(station_lines.values()):
        f.write(line)

# %%
events = pd.read_csv("cctorch/cctorch_events.csv")
events["time"] = pd.to_datetime(events["event_time"], format="mixed")

event_lines = []

mean_latitude = events["latitude"].mean()
mean_longitude = events["longitude"].mean()
for i, row in events.iterrows():
    event_index = row["event_index"]
    origin = row["time"]
    magnitude = row["magnitude"]
    x_err = 0.0
    z_err = 0.0
    time_err = 0.0
    dx, dy, dz = 0.0, 0.0, 0.0
    dx = np.random.uniform(-0.01, 0.01)
    dy = np.random.uniform(-0.01, 0.01)
    # dz = np.random.uniform(0, 10)
    dz = 0
    event_lines.append(
        f"{origin.year:4d}{origin.month:02d}{origin.day:02d}  "
        f"{origin.hour:2d}{origin.minute:02d}{origin.second:02d}{round(origin.microsecond / 1e4):02d}  "
        # f"{row['latitude']:8.4f}  {row['longitude']:9.4f}   {row['depth_km']:8.4f}  "
        f"{row['latitude'] + dy:8.4f}  {row['longitude']+ dx:9.4f}   {row['depth_km']+dz:8.4f}  "
        f"{magnitude:5.2f}  {x_err:5.2f}  {z_err:5.2f}  {time_err:5.2f}  {event_index:9d}\n"
    )

with open(f"{result_path}/events.dat", "w") as f:
    f.writelines(event_lines)

# %%
os.system("bash run_hypodd_cc.sh")

# %%
events_true = pd.read_csv(f"events.csv")
# events_true = pd.read_csv("cctorch/cctorch_events.csv")
events_true.set_index("event_index", inplace=True)
# %%
columns = [
    "event_index",
    "latitude",
    "longitude",
    "depth_km",
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
hypodd_file = f"hypodd/hypodd_cc.loc"
events_init = pd.read_csv(
    hypodd_file,
    delim_whitespace=True,
    header=None,
    names=columns[:-6],
    na_values="**",
)
events_init = events_init.dropna()
events_init["HR"] = events_init["HR"].astype(int).clip(0, 23)
events_init["time"] = events_init.apply(
    lambda x: f'{x["YR"]:04.0f}-{x["MO"]:02.0f}-{x["DY"]:02.0f}T{x["HR"]:02.0f}:{x["MI"]:02.0f}:{min(x["SC"], 59.999):05.3f}',
    axis=1,
)
events_init["time"] = pd.to_datetime(events_init["time"])
events_init.set_index("event_index", inplace=True)

hypodd_file = f"hypodd/hypodd_cc.reloc"
events_hypodd = pd.read_csv(
    hypodd_file,
    delim_whitespace=True,
    header=None,
    names=columns,
    na_values="*",
)
events_hypodd = events_hypodd.dropna()
events_hypodd["time"] = events_hypodd.apply(
    lambda x: f'{x["YR"]:04.0f}-{x["MO"]:02.0f}-{x["DY"]:02.0f}T{x["HR"]:02.0f}:{x["MI"]:02.0f}:{min(x["SC"], 59.999):05.3f}',
    axis=1,
)
events_hypodd["time"] = pd.to_datetime(events_hypodd["time"], format="mixed")
events_hypodd.set_index("event_index", inplace=True)

# %%
plt.figure()
plt.scatter(events_true["longitude"], events_true["latitude"], s=3, alpha=1.0, label="True")
# xlim = plt.xlim()
# ylim = plt.ylim()
plt.scatter(events_hypodd["longitude"], events_hypodd["latitude"], s=3, alpha=1.0, label="Hypodd")
plt.scatter(events_init["longitude"], events_init["latitude"], s=3, alpha=1.0, label="Initial")
# plt.xlim(xlim)
# plt.ylim(ylim)
plt.legend()

for event_index, row in events_init.iterrows():
    if event_index in events_hypodd.index:
        plt.plot(
            [row["longitude"], events_hypodd.loc[event_index, "longitude"]],
            [row["latitude"], events_hypodd.loc[event_index, "latitude"]],
            color="black",
            linewidth=0.2,
            alpha=0.5,
        )

xlim = plt.xlim()
ylim = plt.ylim()
plt.show()

plt.figure()
plt.scatter(events_true["longitude"], events_true["latitude"], s=1, alpha=0.5, label="True")
# xlim = plt.xlim()
# ylim = plt.ylim()
plt.scatter(events_hypodd["longitude"], events_hypodd["latitude"], s=1, alpha=0.5, label="Hypodd")
# plt.xlim(xlim)
# plt.ylim(ylim)
plt.legend()

# draw line between true and hypodd
for event_index, row in events_true.iterrows():
    if event_index in events_hypodd.index:
        plt.plot(
            [row["longitude"], events_hypodd.loc[event_index, "longitude"]],
            [row["latitude"], events_hypodd.loc[event_index, "latitude"]],
            color="black",
            linewidth=0.2,
            alpha=0.5,
        )

plt.xlim(xlim)
plt.ylim(ylim)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(events_true["longitude"], events_true["latitude"], s=5, alpha=0.5, label="True")
ax[0].set_title(f"True: {len(events_true)} events")
ax[1].scatter(events_hypodd["longitude"], events_hypodd["latitude"], s=5, alpha=0.5, label="Hypodd")
ax[1].set_title(f"Hypodd: {len(events_hypodd)} events")
# %%

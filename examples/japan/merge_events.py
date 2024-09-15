# %%
import multiprocessing as mp
import os
from glob import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
if __name__ == "__main__":
    # %%
    data_path = "local/hinet/gamma/"
    result_path = "local/hinet/gamma/"
    # %%
    years = sorted(os.listdir(data_path))

    # %%
    event_files = []
    for year in years:
        event_files.extend(glob(f"{data_path}/{year}/gamma_events_*.csv"))

    event_index0 = 0
    events_list = []
    picks_list = []
    for event_file in tqdm(event_files):

        if os.stat(event_file).st_size == 0:
            continue

        events = pd.read_csv(event_file)
        events["event_index"] = events["event_index"] + event_index0

        picks = pd.read_csv(event_file.replace("_events_", "_picks_"))
        picks["event_index"] = picks["event_index"] + event_index0

        event_index0 = events["event_index"].max() + 1

        events_list.append(events)
        picks_list.append(picks)

    events = pd.concat(events_list, ignore_index=True)
    picks = pd.concat(picks_list, ignore_index=True)
    events.to_csv(f"{result_path}/gamma_events.csv", index=False)
    picks.to_csv(f"{result_path}/gamma_picks.csv", index=False)

    # %%
    events = pd.read_csv(f"{result_path}/gamma_events.csv")
    picks = pd.read_csv(f"{result_path}/gamma_picks.csv")
    events["time"] = pd.to_datetime(events["time"])

    # %%
    fig, ax = plt.subplots()
    ax.hist(events["magnitude"], bins=100)
    ax.set_yscale("log")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Number of events: {len(events)}")

    # %%

    fig, ax = plt.subplots()
    s = 10000 / len(events)
    im = ax.scatter(
        events["longitude"], events["latitude"], s=s, c=events["time"], cmap="viridis_r", linewidth=0, alpha=0.5
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Number of events: {len(events)}")
    fig.colorbar(im, ax=ax, label="Depth (km)")
    # %%
    fig, ax = plt.subplots(figsize=(10, 5))
    s = 10000 / len(events)
    zlim, zmax = 0, 20
    im = ax.scatter(
        events["longitude"], events["depth_km"], s=s, c=events["time"], cmap="viridis_r", linewidth=0, alpha=0.5
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Depth (km)")
    ax.set_ylim(zlim, zmax)
    ax.invert_yaxis()
    ax.set_title(f"Number of events: {len(events)}")
    fig.colorbar(im, ax=ax, label="Time")

    fig, ax = plt.subplots(figsize=(10, 5))
    s = 10000 / len(events)
    im = ax.scatter(
        events["latitude"], events["depth_km"], s=s, c=events["time"], cmap="viridis_r", linewidth=0, alpha=0.5
    )
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Depth (km)")
    ax.set_ylim(zlim, zmax)
    ax.invert_yaxis()
    ax.set_title(f"Number of events: {len(events)}")
    fig.colorbar(im, ax=ax, label="Time")

    # %%
    # Set up the map with Cartopy's PlateCarree projection
    fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={"projection": ccrs.PlateCarree()})
    # Define the size of the scatter plot points
    s = 100000 / len(events)

    # Add scatter plot of events (longitude, latitude) with color based on time
    im = ax.scatter(
        events["longitude"],
        events["latitude"],
        s=s,
        c=events["time"],
        cmap="viridis_r",
        linewidth=0,
        alpha=0.5,
        transform=ccrs.PlateCarree(),
    )

    # Set labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Number of events: {len(events)}")

    # Add colorbar with label 'Time' (adjust if 'Depth' is meant)
    fig.colorbar(im, ax=ax, label="Time")

    # Add topography and other features
    ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS, linestyle=":")
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)
    # ax.stock_img()

    # Optionally, adjust the extent of the map to zoom in on the region of interest
    ax.set_extent(
        [
            events["longitude"].min(),
            events["longitude"].max(),
            events["latitude"].min(),
            events["latitude"].max(),
        ],
        crs=ccrs.PlateCarree(),
    )

    plt.show()

    # %%
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(events["time"], events["magnitude"], s=1.0, cmap="viridis_r", linewidth=0, alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Number of events: {len(events)}")
    # ax.grid(True)

    # %%
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(events["time"], bins=200)
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Number of events: {len(events)}")

# %%

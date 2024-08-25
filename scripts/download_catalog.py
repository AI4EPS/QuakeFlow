# %%
import json
import os
import re
from datetime import timezone
from typing import Dict

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fsspec
import matplotlib
import matplotlib.pyplot as plt
import obspy
import obspy.clients.fdsn
import pandas as pd
import pyproj
from args import parse_args

matplotlib.use("Agg")


def download_catalog(
    root_path: str, region: str, config: Dict, protocol: str = "file", bucket: str = "", token: Dict = None
):

    # %%
    fs = fsspec.filesystem(protocol, token=token)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    result_path = f"{region}/obspy"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    print(json.dumps(config, indent=4))

    proj = pyproj.Proj(
        f"+proj=sterea +lon_0={(config['minlongitude'] + config['maxlongitude'])/2} +lat_0={(config['minlatitude'] + config['maxlatitude'])/2} +units=km"
    )

    # %%
    if ("provider" in config) and (config["provider"] is not None):
        print("Downloading standard catalog...")

        catalog = obspy.Catalog()
        for provider in config["provider"]:
            if os.path.exists(f"{root_path}/{result_path}/catalog_{provider.lower()}.xml"):
                print(f"Loading existing {root_path}/{result_path}/catalog_{provider.lower()}.xml")
                catalog += obspy.read_events(f"{root_path}/{result_path}/catalog_{provider.lower()}.xml")
                continue

            client = obspy.clients.fdsn.Client(provider, timeout=1200)
            events = None
            try:
                events = client.get_events(
                    starttime=config["starttime"],
                    endtime=config["endtime"],
                    minlongitude=config["minlongitude"],
                    maxlongitude=config["maxlongitude"],
                    minlatitude=config["minlatitude"],
                    maxlatitude=config["maxlatitude"],
                    minmagnitude=None if "minmagnitude" not in config else config["minmagnitude"],
                )
            except Exception as e:
                print(f"{provider} failed: {e}")
            if events is not None:
                print(f"Dowloaded {len(events)} events from {provider.lower()}")
                events.write(f"{root_path}/{result_path}/catalog_{provider.lower()}.xml", format="QUAKEML")
                if protocol != "file":
                    fs.put(
                        f"{root_path}/{result_path}/catalog_{provider.lower()}.xml",
                        f"{bucket}/{result_path}/catalog_{provider.lower()}.xml",
                    )
                catalog += events

        if len(catalog) > 0:
            catalog.write(f"{root_path}/{result_path}/catalog.xml", format="QUAKEML")
            if protocol != "file":
                fs.put(f"{root_path}/{result_path}/catalog.xml", f"{bucket}/{result_path}/catalog.xml")

    # %%
    if not os.path.exists(f"{root_path}/{result_path}/catalog.xml"):  # no events
        return 0

    catalog = obspy.read_events(f"{root_path}/{result_path}/catalog.xml")

    def parase_catalog(catalog):
        events = {}
        for event in catalog.events:
            origin = event.preferred_origin()
            magnitude = event.preferred_magnitude()

            try:
                event_id = re.search(r"eventid=(\d+)$", event.resource_id.id).group(1)
            except:
                event_id = event.resource_id.id.split("/")[-1]
            try:
                agency_id = origin.creation_info.agency_id.lower()
            except:
                agency_id = ""

            events[event_id] = {
                "event_id": agency_id + event_id,
                "time": origin.time.datetime.replace(tzinfo=timezone.utc).isoformat(timespec="milliseconds"),
                "magnitude": magnitude.mag if magnitude is not None else None,
                "latitude": origin.latitude,
                "longitude": origin.longitude,
                "depth_km": origin.depth / 1000,
                "agency": agency_id,
            }
        return events

    events = parase_catalog(catalog)
    if len(events) > 0:
        events = pd.DataFrame.from_dict(events, orient="index")
        events[["x_km", "y_km"]] = events.apply(lambda row: pd.Series(proj(row["longitude"], row["latitude"])), axis=1)
        events["z_km"] = events["depth_km"]
        events[["latitude", "longitude"]] = events[["latitude", "longitude"]].round(5)
        events["depth_km"] = events["depth_km"].round(3)
        events[["x_km", "y_km", "z_km"]] = events[["x_km", "y_km", "z_km"]].round(3)
        events.sort_values("time", inplace=True)
        events.to_csv(f"{root_path}/{result_path}/catalog.csv", index=False)
        if protocol != "file":
            fs.put(f"{root_path}/{result_path}/catalog.csv", f"{bucket}/{result_path}/catalog.csv")

    # %%
    def visulization(config, events=None, stations=None, fig_name="catalog.png"):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([config["minlongitude"], config["maxlongitude"], config["minlatitude"], config["maxlatitude"]])
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color="gray", alpha=0.5, linestyle="--"
        )
        if events is not None:
            if len(events) > 0:
                ax.scatter(
                    events.longitude,
                    events.latitude,
                    transform=ccrs.PlateCarree(),
                    s=1e4 / len(events),
                    c="r",
                    marker=".",
                    label="events",
                )
                ax.set_title(f"{config['provider']}: {len(events)} events")
        if stations is not None:
            ax.scatter(
                stations.longitude,
                stations.latitude,
                transform=ccrs.PlateCarree(),
                s=100,
                c="b",
                marker="^",
                label="stations",
            )
        ax.legend(scatterpoints=1, markerscale=0.5, fontsize=10)
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")

    stations = None
    if os.path.exists(f"{root_path}/{result_path}/stations.csv"):
        stations = pd.read_csv(f"{root_path}/{result_path}/stations.csv")
    visulization(config, events, stations, f"{root_path}/{result_path}/catalog.png")
    if protocol != "file":
        fs.put(f"{root_path}/{result_path}/catalog.png", f"{bucket}/{result_path}/catalog.png")

    # %% copy to results/network
    if not os.path.exists(f"{root_path}/{region}/results/network"):
        os.makedirs(f"{root_path}/{region}/results/network")
    for file in ["catalog.csv", "catalog.png"]:
        os.system(f"cp {root_path}/{result_path}/{file} {root_path}/{region}/results/network/{file}")
        if protocol != "file":
            fs.put(f"{root_path}/{result_path}/{file}", f"{bucket}/{region}/results/network/{file}")


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    download_catalog(root_path, region=region, config=config, protocol="file", bucket="", token=None)

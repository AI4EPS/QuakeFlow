# %%
import json
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import obspy
import obspy.clients.fdsn
import obspy.geodetics.base
import obspy.taup
from obspy.core import UTCDateTime
from obspy.clients.fdsn import header
from collections import namedtuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pyproj

# %%
region = ""
config = {}
config_region = {}
exec(open("config.py").read())

# %%
region = "BayArea"
config.update(config_region[region])

# %%
root_path = Path(f"{region}")
if not root_path.exists():
    root_path.mkdir()
result_path = root_path / "obspy"
if not result_path.exists():
    result_path.mkdir()

with open(root_path / "config.json", "w") as fp:
    json.dump(config, fp, indent=4)

# %%
with open(root_path / "config.json", "r") as fp:
    config = json.load(fp)
print(json.dumps(config, indent=4))

proj = pyproj.Proj(
    f"+proj=sterea +lon_0={(config['minlongitude'] + config['maxlongitude'])/2} +lat_0={(config['minlatitude'] + config['maxlatitude'])/2} +units-km"
)

# %%
##################### Download catalog #####################
if ("provider" in config) and (config["provider"] is not None):
    print("Downloading standard catalog...")

    catalog = obspy.Catalog()
    for provider in config["provider"]:
        client = obspy.clients.fdsn.Client(provider)
        try:
            events = client.get_events(
                starttime=config["starttime"],
                endtime=config["endtime"],
                minlongitude=config["minlongitude"],
                maxlongitude=config["maxlongitude"],
                minlatitude=config["minlatitude"],
                maxlatitude=config["maxlatitude"],
            )
        except header.FDSNNoDataException as e:
            print(e)
            events = obspy.Catalog()
        print(f"Dowloaded {len(events)} events from {provider}")
        events.write(f"{result_path}/catalog_{provider.lower()}.xml", format="QUAKEML")
        catalog += events

    catalog.write(f"{result_path}/catalog.xml", format="QUAKEML")


# %%
catalog = obspy.read_events(f"{result_path}/catalog.xml")


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
        }
    return events


events = parase_catalog(catalog)
events = pd.DataFrame.from_dict(events, orient="index")

events[["x_km", "y_km"]] = events.apply(lambda row: pd.Series(proj(row["longitude"], row["latitude"])), axis=1)
events["z_km"] = events["depth_km"]
events[["latitude", "longitude"]] = events[["latitude", "longitude"]].round(5)
events["depth_km"] = events["depth_km"].round(3)
events[["x_km", "y_km", "z_km"]] = events[["x_km", "y_km", "z_km"]].round(3)
events.to_csv(result_path / "events.csv", index=False)

# %%
##################### Download stations #####################
if ("provider" in config) and (config["provider"] is not None):
    print("Downloading stations...")
    inventory = obspy.Inventory()
    for provider in config["provider"]:
        client = obspy.clients.fdsn.Client(provider)
        stations = client.get_stations(
            network=config["network"],
            station=config["station"],
            starttime=config["starttime"],
            endtime=config["endtime"],
            minlatitude=config["minlatitude"],
            maxlatitude=config["maxlatitude"],
            minlongitude=config["minlongitude"],
            maxlongitude=config["maxlongitude"],
            channel=config["channel"],
            level=config["level"],
        )

        print(
            f"Dowloaded {len([chn for net in stations for sta in net for chn in sta])} {config['level']}s from {provider}"
        )
        stations.write(f"{result_path}/inventory_{provider.lower()}.xml", format="STATIONXML")
        inventory += stations

    inventory.write(f"{result_path}/inventory.xml", format="STATIONXML")

# %%
if os.path.exists(f"{result_path}/inventory.xml"):
    inventory = obspy.read_inventory(f"{result_path}/inventory.xml")


def parse_response(inventory, mseed_ids=[]):
    stations = {}
    num = 0
    for net in inventory:
        for sta in net:
            components = defaultdict(list)
            unique = {}

            for chn in sta:
                key = f"{chn.location_code}{chn.code[:-1]}"
                components[key].append(chn.code[-1])

                if key not in unique:
                    unique[key] = {
                        "latitude": chn.latitude,
                        "longitude": chn.longitude,
                        "elevation_m": chn.elevation,
                        "location": chn.location_code,
                        "instrument": chn.code[:-1],
                    }

            for key in components:
                station_id = f"{net.code}.{sta.code}.{unique[key]['location']}.{unique[key]['instrument']}"
                if (len(mseed_ids) > 0) and (station_id not in mseed_ids):
                    continue
                num += 1
                x_km, y_km = proj(unique[key]["longitude"], unique[key]["latitude"])
                z_km = -unique[key]["elevation_m"] / 1e3
                stations[station_id] = {
                    "network": net.code,
                    "station": sta.code,
                    "location": unique[key]["location"],
                    "instrument": unique[key]["instrument"],
                    "component": sorted(components[key]),
                    "latitude": unique[key]["latitude"],
                    "longitude": unique[key]["longitude"],
                    "elevation_m": unique[key]["elevation_m"],
                    "depth_km": -unique[key]["elevation_m"] / 1e3,
                    "x_km": round(x_km, 3),
                    "y_km": round(y_km, 3),
                    "z_km": round(z_km, 3),
                }

    print(f"Parse {num} stations")

    return stations


# mseed_ids = None
stations = parse_response(inventory)
with open(result_path / "stations.json", "w") as f:
    json.dump(stations, f, indent=4)
stations = pd.DataFrame.from_dict(stations, orient="index")
# stations[["x_km", "y_km"]] = stations.apply(lambda row: pd.Series(proj(row["longitude"], row["latitude"])), axis=1)
# stations["z_km"] = stations["depth_km"]
stations[["latitude", "longitude"]] = stations[["latitude", "longitude"]].round(4)
stations["depth_km"] = stations["depth_km"].round(2)
stations[["x_km", "y_km", "z_km"]] = stations[["x_km", "y_km", "z_km"]].round(2)
stations.to_csv(result_path / "stations.csv", index_label="station_id")


# %%
# plot stations and events in a map using cartopy
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([config["minlongitude"], config["maxlongitude"], config["minlatitude"], config["maxlatitude"]])
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color="gray", alpha=0.5, linestyle="--")
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
ax.scatter(
    stations.longitude, stations.latitude, transform=ccrs.PlateCarree(), s=100, c="b", marker="^", label="stations"
)
ax.legend(scatterpoints=1, markerscale=0.5, fontsize=10)
plt.savefig(result_path / "stations.png", dpi=300, bbox_inches="tight")


# %%
def calc_arrival_time(event):
    taup_model = obspy.taup.TauPyModel(model="iasp91")
    Location = namedtuple("location", ["longitude", "latitude", "depth_km"])
    min_dist = None
    for _, station in stations.items():
        dist = np.sqrt(
            ((event["longitude"] - station["longitude"]) * np.cos(np.deg2rad(event["latitude"]))) ** 2
            + (event["latitude"] - station["latitude"]) ** 2
        )
        if min_dist is None or min_dist > dist:
            min_dist = dist
            closest = Location(
                longitude=station["longitude"], latitude=station["latitude"], depth_km=station["depth_km"]
            )
    min_dist_km = (
        obspy.geodetics.base.gps2dist_azimuth(event.latitude, event["longitude"], closest.latitude, closest.longitude)[
            0
        ]
        / 1000
    )
    min_dist_deg = obspy.geodetics.base.kilometer2degrees(min_dist_km)
    min_tt = taup_model.get_travel_times(
        distance_in_degree=min_dist_deg,
        source_depth_in_km=max(0, event.depth_km),
        receiver_depth_in_km=max(0, closest.depth_km),
    )[0].time

    return event.time + pd.to_timedelta(min_tt, unit="s")


# %%
def download_event(event, stations, client, waveform_path, time_before=30, time_after=90, lock=None, rank=0):
    if not waveform_path.exists():
        waveform_path.mkdir(parents=True)

    max_retry = 10

    arrival_time = calc_arrival_time(event)
    starttime = arrival_time - pd.to_timedelta(time_before, unit="s")
    endtime = arrival_time + pd.to_timedelta(time_after, unit="s")
    mseed_name = waveform_path / f"{event.event_id}.mseed"
    if mseed_name.exists():
        print(f"{mseed_name} already exists. Skip.")
        return

    data = []
    if rank == 0:
        print(f"Downloading {event.event_id} ...")
    for key, station in stations.items():
        if rank == 0:
            print(f"{key}", end=", ", flush=True)
        for comp in station["component"]:
            retry = 0
            while retry < max_retry:
                try:
                    stream = client.get_waveforms(
                        network=station["network"],
                        station=station["station"],
                        location=station["location"],
                        channel=station["instrument"] + comp,
                        starttime=obspy.UTCDateTime(starttime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")),
                        endtime=obspy.UTCDateTime(endtime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")),
                    )
                    stream = stream.merge()
                    if len(stream) > 1:
                        print(f"Warning: {key} has {len(stream)} traces: {stream.get_id()}")
                    data.append(stream[0])
                    break

                except Exception as err:
                    err = str(err).rstrip("\n")
                    message = "No data available for request."
                    if err[: len(message)] == message:
                        # print(f"No data available for {key}")
                        break
                    else:
                        print(f"Error occurred:\n{err}\nRetrying...")
                    retry += 1
                    time.sleep(5)
                    continue

        if retry == max_retry:
            print(f"Failed to download {key} after {max_retry} retries.")

    if rank == 0:
        print("")
    if len(data) > 0:
        data = obspy.Stream(traces=data)
        data.write(str(mseed_name), format="MSEED")
    else:
        os.system(f"touch {str(mseed_name).replace('.mseed', '.failed')}")


# %%
waveform_path = result_path / "waveforms"
if not waveform_path.exists():
    waveform_path.mkdir()

events = pd.read_csv(result_path / "events.csv", parse_dates=["time"])
for provider in config["provider"]:
    with open(result_path / f"stations.json") as f:
        stations = json.load(f)

    client = obspy.clients.fdsn.Client(provider)

    # for _, event in events.iterrows():
    #     download_event(event, stations, client, waveform_path)

    threads = []
    MAX_THREADS = 3
    lock = threading.Lock()
    for ii, event in events.iterrows():
        t = threading.Thread(target=download_event, args=(event, stations, client, waveform_path, 30, 90, lock, ii))
        t.start()
        threads.append(t)
        if ii % MAX_THREADS == MAX_THREADS - 1:
            for t in threads:
                t.join()
            threads = []
    for t in threads:
        t.join()

# %%

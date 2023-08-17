# %%
import json
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import obspy
import obspy.clients.fdsn
import pandas as pd
import pyproj
from obspy.clients.fdsn import header

# %%
region = ""
config = {}
config_region = {}
exec(open("config.py").read())

# %%
region = "demo"
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
        print(f"Dowloaded {len(events)} events from {provider.lower()}")
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
    print("Downloading station response...")
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
            f"Dowloaded {len([chn for net in stations for sta in net for chn in sta])} stations from {provider.lower()}"
        )
        stations.write(f"{result_path}/inventory_{provider.lower()}.xml", format="STATIONXML")
        inventory += stations

    inventory.write(f"{result_path}/inventory.xml", format="STATIONXML")

# %%
if os.path.exists(f"{result_path}/inventory.xml"):
    inventory = obspy.read_inventory(f"{result_path}/inventory.xml")


def parse_inventory_json(inventory, mseed_ids=[]):
    comp = ["3", "2", "1", "E", "N", "Z"]
    order = {key: i for i, key in enumerate(comp)}
    stations = {}
    num = 0
    for net in inventory:
        for sta in net:
            sta_3c = {}
            components = defaultdict(list)

            for chn in sta:
                key = f"{chn.location_code}.{chn.code[:-1]}"
                components[key].append([chn.code[-1], chn.response.instrument_sensitivity.value])

                if key not in sta_3c:
                    sta_3c[key] = {
                        "latitude": chn.latitude,
                        "longitude": chn.longitude,
                        "elevation_m": chn.elevation,
                        "location": chn.location_code,
                        "instrument": chn.code[:-1],
                    }

            for key in list(sta_3c.keys()):
                station_id = f"{net.code}.{sta.code}.{sta_3c[key]['location']}.{sta_3c[key]['instrument']}"
                if (len(mseed_ids) > 0) and (station_id not in mseed_ids):
                    continue
                num += 1
                x_km, y_km = proj(sta_3c[key]["longitude"], sta_3c[key]["latitude"])
                z_km = -sta_3c[key]["elevation_m"] / 1e3
                stations[station_id] = {
                    "network": net.code,
                    "station": sta.code,
                    "location": sta_3c[key]["location"],
                    "instrument": sta_3c[key]["instrument"],
                    "component": "".join([x[0] for x in sorted(components[key], key=lambda x: order[x[0]])]),
                    "sensitivity": [x[1] for x in sorted(components[key], key=lambda x: order[x[0]])],
                    "latitude": sta_3c[key]["latitude"],
                    "longitude": sta_3c[key]["longitude"],
                    "elevation_m": sta_3c[key]["elevation_m"],
                    "depth_km": -sta_3c[key]["elevation_m"] / 1e3,
                    "x_km": round(x_km, 3),
                    "y_km": round(y_km, 3),
                    "z_km": round(z_km, 3),
                }

    print(f"Parse {num} stations")

    return stations


def parse_inventory_csv(inventory, mseed_ids=[]):
    channel_list = []
    for network in inventory:
        for station in network:
            for channel in station:
                if channel.sensor is None:
                    sensor_description = ""
                else:
                    sensor_description = channel.sensor.description
                channel_list.append(
                    {
                        "network": network.code,
                        "station": station.code,
                        "location": channel.location_code,
                        "instrument": channel.code[:-1],
                        "component": channel.code[-1],
                        "channel": channel.code,
                        "longitude": channel.longitude,
                        "latitude": channel.latitude,
                        "elevation_m": channel.elevation,
                        "depth_km": -channel.elevation / 1e3,
                        # "depth_km": channel.depth,
                        "begin_time": channel.start_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                        if channel.start_date is not None
                        else None,
                        "end_time": channel.end_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                        if channel.end_date is not None
                        else None,
                        "azimuth": channel.azimuth,
                        "dip": channel.dip,
                        "sensitivity": channel.response.instrument_sensitivity.value,
                        "site": station.site.name,
                        "sensor": sensor_description,
                    }
                )
    channel_list = pd.DataFrame(channel_list)

    print(f"Parse {len(channel_list)} stations")

    return channel_list


stations = parse_inventory_csv(inventory)
# with open(result_path / "stations.json", "w") as f:
#     json.dump(stations, f, indent=4)
# stations = pd.DataFrame.from_dict(stations, orient="index")
stations[["x_km", "y_km"]] = stations.apply(lambda row: pd.Series(proj(row["longitude"], row["latitude"])), axis=1)
stations["z_km"] = stations["depth_km"]
stations[["latitude", "longitude"]] = stations[["latitude", "longitude"]].round(4)
stations["depth_km"] = stations["depth_km"].round(2)
stations[["x_km", "y_km", "z_km"]] = stations[["x_km", "y_km", "z_km"]].round(2)
stations = stations.sort_values(by=["network", "station", "location", "channel"])
stations = stations.groupby(["network", "station", "location", "channel"]).first().reset_index()
stations.to_csv(result_path / "stations.csv", index=False)

stations = parse_inventory_json(inventory)
with open(result_path / "stations.json", "w") as f:
    json.dump(stations, f, indent=4)


# %%
def visulization(config, events, stations, fname):
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
    plt.savefig(fname, dpi=300, bbox_inches="tight")


fname = f"{result_path}/stations.png"
stations=pd.DataFrame.from_dict(stations, orient="index")
visulization(config, events, stations, fname)


# %%
def download(starttime, inventory, waveform_path, deltatime="1D", lock=None):
    starttime = obspy.UTCDateTime(starttime)
    if deltatime == "1H":
        endtime = starttime + 3600
        waveform_path = waveform_path / f"{starttime.strftime('%Y-%j')}/{starttime.strftime('%H')}"
    elif deltatime == "1D":
        endtime = starttime + 3600 * 24
        waveform_path = waveform_path / f"{starttime.strftime('%Y-%j')}"
    else:
        raise ValueError("Invalid interval")

    if not waveform_path.exists():
        waveform_path.mkdir(parents=True)

    max_retry = 10
    ## FIXIT: not working correctly
    # inventory = inventory.select(starttime=starttime, endtime=endtime)
    for network in inventory:
        for station in network:
            for channel in station:
                mseed_name = (
                    waveform_path / f"{network.code}.{station.code}.{channel.location_code}.{channel.code}.mseed"
                )
                if mseed_name.exists():
                    print(f"File {mseed_name} already exists. Skip.")
                    continue
                # mseed_txt = waveform_path / f"{network.code}.{station.code}.{channel.location_code}.{channel.code}.txt"
                # if mseed_txt.exists():
                #     print(f"File {mseed_txt} no data available. Skip.")
                #     continue

                retry = 0
                while retry < max_retry:
                    try:
                        stream = client.get_waveforms(
                            network=network.code,
                            station=station.code,
                            location=channel.location_code,
                            channel=channel.code,
                            starttime=starttime,
                            endtime=endtime,
                        )
                        stream.write(mseed_name, format="MSEED")
                        print(f"Downloaded {mseed_name}")
                        break

                    except Exception as err:
                        err = str(err).rstrip("\n")
                        message = "No data available for request."
                        if err[: len(message)] == message:
                            print(f"No data available for {mseed_name.name}")
                            # os.system(f"touch {mseed_txt}")
                            break
                        else:
                            print(f"Error occurred:{err}. Retrying...")
                        retry += 1
                        time.sleep(5)
                        continue

                if retry == max_retry:
                    print(f"Failed to download {mseed_name}")


# %%
waveform_path = result_path / "waveforms"
if not waveform_path.exists():
    waveform_path.mkdir()

for provider in config["provider"]:
    inventory = obspy.read_inventory(f"{result_path}/inventory_{provider.lower()}.xml")

    client = obspy.clients.fdsn.Client(provider)

    deltatime = "1H"  # "1H"
    if deltatime == "1H":
        starttime_period = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%dT%H")
    elif deltatime == "1D":
        starttime_period = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%d")
    starttimes = pd.date_range(starttime_period, config["endtime"], freq=deltatime, tz="UTC", inclusive="left")

    threads = []
    MAX_THREADS = 3
    lock = threading.Lock()
    for ii, starttime in enumerate(starttimes):
        t = threading.Thread(target=download, args=(starttime, inventory, waveform_path, deltatime, lock))
        t.start()
        threads.append(t)
        if ii % MAX_THREADS == MAX_THREADS - 1:
            for t in threads:
                t.join()
            threads = []
    for t in threads:
        t.join()


# %%
mseeds = (result_path / "waveforms").rglob("*.mseed")
mseed_ids = []
for mseed in mseeds:
    mseed_ids.append(mseed.name.split(".mseed")[0][:-1])

# %%
if os.path.exists(f"{result_path}/inventory.xml"):
    inventory = obspy.read_inventory(f"{result_path}/inventory.xml")


# mseed_ids = None
stations = parse_inventory_json(inventory, mseed_ids)
with open(result_path / "stations.json", "w") as f:
    json.dump(stations, f, indent=4)
stations = pd.DataFrame.from_dict(stations, orient="index")
stations.to_csv(result_path / "stations_valid.csv", index_label="station_id")


# %%
visulization(config, events, stations, fname=f"{result_path}/stations_valid.png")
# %%

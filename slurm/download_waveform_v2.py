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
from obspy.clients.fdsn import header
import pandas as pd

# %%
region = ""
config = {}
config_region = {}
exec(open('config.py').read())

# %%
region = "Kilauea"
config.update(config_region[region])

# %%
root_path = Path(f"/workspaces/disk/{region}")
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

# %%
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
        events.write(f"{result_path}/catalog_{provider}.xml", format="QUAKEML")
        catalog += events

    catalog.write(f"{result_path}/catalog.xml", format="QUAKEML")

# %%
if ("provider" in config) and (config["provider"] is not None):

    print("Downloading station response...")
    inventory = obspy.Inventory()
    for provider in config["provider"]:
        client = obspy.clients.fdsn.Client(provider)
        if ("station" in config) and (config["station"] is not None):
            stations = client.get_stations(
                network=config["network"], 
                station=config["station"],
                starttime=config["starttime"], 
                endtime=config["endtime"],
                channel=config["channel"],
                level="response")
        else:
            stations = client.get_stations(
                starttime=config["starttime"], 
                endtime=config["endtime"],
                minlatitude=config["minlatitude"],
                maxlatitude=config["maxlatitude"],
                minlongitude=config["minlongitude"],
                maxlongitude=config["maxlongitude"],
                channel=config["channel"],
                level="response")
            
        print(f"Dowloaded {len([chn for net in stations for sta in net for chn in sta])} stations from {provider}")
        stations.write(f"{result_path}/inventory_{provider}.xml", format="STATIONXML")
        inventory += stations

    inventory.write(f"{result_path}/inventory.xml", format="STATIONXML")

# %%
def download(starttime, inventory, waveform_path, deltatime="1D", lock=None):

    starttime = obspy.UTCDateTime(starttime)
    if deltatime == "1H":
        endtime = starttime + 3600
        waveform_path = waveform_path / f"{starttime.strftime('%Y-%j')}/{starttime.strftime('%H')}"
    elif deltatime == "1D":
        endtime = starttime + 3600*24
        waveform_path = waveform_path / f"{starttime.strftime('%Y-%j')}"
    else:
        raise ValueError("Invalid interval")
    
    if not waveform_path.exists():
        waveform_path.mkdir(parents=True)

    max_retry = 10
    inventory = inventory.select(starttime=starttime, endtime=endtime)
    for network in inventory:
        for station in network:
            for channel in station:

                mseed_name = waveform_path / f"{network.code}.{station.code}.{channel.location_code}.{channel.code}.mseed"
                if mseed_name.exists():
                    print(f"File {mseed_name} already exists. Skip.")
                    continue
                mseed_txt = waveform_path / f"{network.code}.{station.code}.{channel.location_code}.{channel.code}.txt"
                if mseed_txt.exists():
                    print(f"File {mseed_txt} no data available. Skip.")
                    continue
                
                retry = 0
                while retry < max_retry:
                    try:
                        stream = client.get_waveforms(
                            network = network.code,
                            station = station.code,
                            location = channel.location_code,
                            channel = channel.code,
                            starttime = starttime,
                            endtime = endtime,
                        )
                        stream.write(mseed_name, format="MSEED")
                        print(f"Downloaded {mseed_name}")
                        break

                    except Exception as err:
                        err = str(err).rstrip("\n")
                        message = "No data available for request."
                        if err[: len(message)] == message:
                            print(f"No data available for {mseed_name}")
                            os.system(f"touch {mseed_txt}")
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

    inventory = obspy.read_inventory(f"{result_path}/inventory_{provider}.xml")

    client = obspy.clients.fdsn.Client(provider)

    deltatime = "1D" # "1H"
    if deltatime == "1H":
        starttime_hour = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%dT%H")
    elif deltatime == "1D":    
        starttime_day = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%d")
    starttimes = pd.date_range(starttime_day, config["endtime"], freq=deltatime, tz="UTC", inclusive='left')

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
catalog = obspy.read_events(f"{result_path}/catalog.xml")

def parase_catalog(catalog):
    events = {}
    for event in catalog:
        event_id = re.search(r"eventid=(\d+)$", event.resource_id.id).group(1)
        events[event_id] = {
            "time": event.origins[0].time,
            "magnitude": event.magnitudes[0].mag,
            "latitude": event.origins[0].latitude,
            "longitude": event.origins[0].longitude,
            "depth_km": event.origins[0].depth / 1000,
        }
    return events

events = parase_catalog(catalog)
events = pd.DataFrame.from_dict(events, orient="index")
events.to_csv(result_path / "events.csv", index_label="event_id")

# %%
mseeds = (result_path / "waveforms").rglob("*.mseed")
mseed_ids = []
for mseed in mseeds:
    mseed_ids.append(mseed.name.split(".mseed")[0][:-1])

# %%
if os.path.exists(f"{result_path}/inventory.xml"):
    inventory = obspy.read_inventory(f"{result_path}/inventory.xml")

def parse_response(inventory, mseed_ids=None):
    stations = {}
    num = 0
    for net in inventory:
        for sta in net:
            components = defaultdict(list)
            channel = {}

            for chn in sta:
                key = f"{chn.location_code}{chn.code[:-1]}"
                components[key].append(chn.code[-1])

                if key not in channel:
                    channel[key] = {
                        "latitude": chn.latitude,
                        "longitude": chn.longitude,
                        "elevation_m": chn.elevation,
                        "location": chn.location_code,
                        "device": chn.code[:-1],
                    }

            for key in components:
                station_id = f"{net.code}.{sta.code}.{channel[key]['location']}.{channel[key]['device']}"
                if (mseed_ids is not None) and (station_id not in mseed_ids):
                    continue
                num += 1
                stations[station_id] = {
                    "network": net.code,
                    "station": sta.code,
                    "location": channel[key]["location"],
                    "component": sorted(components[key]),
                    "latitude": channel[key]["latitude"],
                    "longitude": channel[key]["longitude"],
                    "elevation_m": channel[key]["elevation_m"],
                    "depth_km": - channel[key]["elevation_m"] / 1e3,
                }
                
    print(f"Parse {num} stations")

    return stations

# mseed_ids = None
stations = parse_response(inventory, mseed_ids)
with open(result_path / "stations.json", "w") as f:
    json.dump(stations, f, indent=4)
stations = pd.DataFrame.from_dict(stations, orient="index")
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
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
if len(events) > 0:
    ax.scatter(events.longitude, events.latitude, transform=ccrs.PlateCarree(), s=1e4/len(events), c="r", marker=".", label="events")
ax.scatter(stations.longitude, stations.latitude, transform=ccrs.PlateCarree(), s=100, c="b", marker="^", label="stations")
ax.legend(scatterpoints=1, markerscale=0.5, fontsize=10)
plt.savefig(result_path / "stations.png", dpi=300, bbox_inches="tight")

# %%

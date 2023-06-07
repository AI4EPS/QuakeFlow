# %%
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import obspy
import obspy.clients.fdsn
from obspy.clients.fdsn import header
import pandas as pd
from obspy.clients.fdsn.mass_downloader import (CircularDomain, GlobalDomain,
                                                MassDownloader,
                                                RectangularDomain,
                                                Restrictions)

# %%
region = ""
config = {}
config_region = {}
exec(open('config.py').read())

# %%
region = "Kilauea_debug"
config.update(config_region[region])

# %%
root_path = Path(f"{region}")
if not root_path.exists():
    root_path.mkdir()
result_path = root_path / "results"
if not result_path.exists():
    result_path.mkdir()

with open(result_path / "config.json", "w") as fp:
    json.dump(config, fp, indent=4)

# %%
with open(result_path / "config.json", "r") as fp:
    config = json.load(fp)
print(config)

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
        events.write(f"{root_path}/catalog_{provider}.xml", format="QUAKEML")
        catalog += events

    catalog.write(f"{root_path}/catalog.xml", format="QUAKEML")

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
        stations.write(f"{root_path}/inventory_{provider}.xml", format="STATIONXML")
        inventory += stations

    inventory.write(f"{root_path}/inventory.xml", format="STATIONXML")


# %%
domain = GlobalDomain()
if ("longitude0" in config) and ("latitude0" in config) and ("maxradius_degree" in config):
    domain = CircularDomain(
        longitude=config["longitude0"], latitude=config["latitude0"], minradius=0, maxradius=config["maxradius_degree"]
    )
if ("minlatitude" in config) and ("maxlatitude" in config) and ("minlongitude" in config) and ("maxlongitude" in config):
    domain = RectangularDomain(minlatitude=config["minlatitude"], maxlatitude=config["maxlatitude"],
                               minlongitude=config["minlongitude"], maxlongitude=config["maxlongitude"])
print(f"{domain = }")

restrictions = Restrictions(
    starttime=obspy.UTCDateTime(config["starttime"]),
    endtime=obspy.UTCDateTime(config["endtime"]),
    chunklength_in_sec=3600,
    network=config["network"],
    station=config["station"],
    minimum_interstation_distance_in_m=0,
    minimum_length=0.1,
    reject_channels_with_gaps=False,
    channel_priorities=config["channel_priorities"],
)
print(f"{restrictions = }")

def get_mseed_storage(network, station, location, channel, starttime, endtime):
    file_name = f"{root_path}/waveforms/{starttime.strftime('%Y-%j')}/{starttime.strftime('%H')}/{network}.{station}.{location}.{channel}.mseed"
    if os.path.exists(file_name):
        return True
    return file_name

print(f"Downloading waveforms...")
mdl = MassDownloader(
    providers=config["provider"],
)
mdl.download(
    domain,
    restrictions,
    mseed_storage=get_mseed_storage,
    stationxml_storage=f"{root_path}/stations",
    download_chunk_size_in_mb=20,
    threads_per_client=3, # default 3
)

# %%
catalog = obspy.read_events(f"{root_path}/catalog.xml")

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
mseeds = (root_path / "waveforms").rglob("*.mseed")
mseed_ids = []
for mseed in mseeds:
    mseed_ids.append(mseed.name.split(".mseed")[0][:-1])

# %%
if os.path.exists(f"{root_path}/response.xml"):
    inventory = obspy.read_inventory(f"{root_path}/response.xml")
if os.path.exists(f"{root_path}/stations"):
    inventory += obspy.read_inventory(f"{root_path}/stations/*xml")

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
                }
                
    print(f"Found {num} stations")
    return stations

# mseed_ids = None
stations = parse_response(inventory, mseed_ids)
with open(result_path / "stations.json", "w") as f:
    json.dump(stations, f, indent=4)
stations = pd.DataFrame.from_dict(stations, orient="index")
stations.to_csv(result_path / "stations.csv", index_label="station_id")
    
# %%
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
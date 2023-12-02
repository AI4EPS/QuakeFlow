# %%
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm
from glob import glob

# %%
root_path = "./"
waveform_path = f"{root_path}/waveform"
catalog_path = f"{root_path}/catalog"
station_path = f"{root_path}/station"
result_path = f"dataset/station"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# %%
def parse_inventory_csv(inventory):
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
                        "local_depth_m": -channel.depth,
                        "depth_km": -(channel.elevation - channel.depth) / 1000,
                        # "depth_km": channel.depth,
                        "begin_time": channel.start_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                        if channel.start_date is not None
                        else None,
                        "end_time": channel.end_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                        if channel.end_date is not None
                        else None,
                        "azimuth": channel.azimuth,
                        "dip": channel.dip,
                        "sensitivity": channel.response.instrument_sensitivity.value
                        if channel.response.instrument_sensitivity
                        else None,
                        "site": station.site.name,
                        "sensor": sensor_description,
                    }
                )
    channel_list = pd.DataFrame(channel_list)

    print(f"Parse {len(channel_list)} channels into csv")

    return channel_list


# for network in station_path.glob("*.info"):
for network in glob(f"{station_path}/*.info"):
    network_name = network.split("/")[-1]
    if network_name in ["broadband.info", "BARD.info", "CISN.info"]:
        continue
    # if (root_path / "station" / f"{network.stem}.csv").exists():
    network_stem = network.split("/")[-1].split(".")[0]
    if os.path.exists(f"{result_path}/{network_stem}.csv"):
        print(f"Skip {network_stem}")
        # continue
    print(f"Parse {network_stem}")
    inv = obspy.Inventory()
    # for xml in (network / f"{network_stem}.FDSN.xml").glob(f"{network_stem}.*.xml"):
    for xml in glob(f"{network}/{network_stem}.FDSN.xml/{network_stem}.*.xml"):
        inv += obspy.read_inventory(xml)
    stations = parse_inventory_csv(inv)
    if len(stations) > 0:
        stations.to_csv(f"{result_path}/{network_stem}.csv", index=False)


# %%

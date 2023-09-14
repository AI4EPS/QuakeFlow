# %%
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fsspec
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm

# %%
protocol = "s3"
bucket = "scedc-pds"
fs = fsspec.filesystem(protocol, anon=True)

# %%
catalog_path = "event_phases"
station_path = "FDSNstationXML"
waveform_path = "continuous_waveforms/"
dataset_path = Path("./dataset")
if not dataset_path.exists():
    dataset_path.mkdir()
if not (dataset_path / "statioin").exists():
    (dataset_path / "station").mkdir()


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


# %%
inv = obspy.Inventory()
for network in fs.glob(f"{bucket}/{station_path}/*"):
    print(f"Parse {network}")
    # inv = obspy.Inventory()
    for xml in tqdm(fs.glob(f"{network}/*.xml")):
        with fs.open(xml) as f:
            inv += obspy.read_inventory(f)

stations = parse_inventory_csv(inv)
# stations.to_csv(dataset_path / "station" / f"{network.split('/')[-1]}.csv", index=False)

for network, sta in stations.groupby(["network"]):
    print(network, sta)
    sta.to_csv(dataset_path / "station" / f"{network}.csv", index=False)
# %%

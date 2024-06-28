# %%
import os
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path

import fsspec
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm

# %%
# root_path = "./"
# waveform_path = f"{root_path}/waveform"
# catalog_path = f"{root_path}/catalog"
# station_path = f"{root_path}/station"
# result_path = f"dataset/station"
# if not os.path.exists(result_path):
#     os.makedirs(result_path)

input_protocol = "gs"
input_bucket = "quakeflow_dataset/NC"
input_fs = fsspec.filesystem(input_protocol, anon=True)

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset/NC"
output_fs = fsspec.filesystem(output_protocol, token=output_token)

# %%
station_path = f"{input_bucket}/raw_data/FDSNstationXML"

# %% copy to FDSNstationXML
for network in input_fs.glob(f"{station_path}/*.info"):
    network_name = network.split("/")[-1]
    if network_name in ["broadband.info", "BARD.info", "CISN.info"]:
        continue
    print(f"Parse {network}")
    network_code = network.split("/")[-1].split(".")[0]
    print(network_code)
    inv = obspy.Inventory()
    for xml in tqdm(input_fs.glob(f"{network}/{network_code}.FDSN.xml/{network_code}.*.xml")):
        with input_fs.open(xml) as src:
            xml_name = xml.split("/")[-1]
            with output_fs.open(f"{output_bucket}/FDSNstationXML/{network_code}/{xml_name}", "wb") as dst:
                dst.write(src.read())


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
                        "local_depth_m": channel.depth,
                        "depth_km": round(-channel.elevation / 1000, 4),
                        # "depth_km": channel.depth,
                        "begin_time": (
                            channel.start_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                            if channel.start_date is not None
                            else None
                        ),
                        "end_time": (
                            channel.end_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                            if channel.end_date is not None
                            else None
                        ),
                        "azimuth": channel.azimuth,
                        "dip": channel.dip,
                        "sensitivity": (
                            channel.response.instrument_sensitivity.value
                            if channel.response.instrument_sensitivity
                            else None
                        ),
                        "site": station.site.name,
                        "sensor": sensor_description,
                    }
                )
    channel_list = pd.DataFrame(channel_list)

    print(f"Parse {len(channel_list)} channels into csv")

    return channel_list


# %%
inv = obspy.Inventory()
for network in input_fs.glob(f"{station_path}/*.info"):
    network_name = network.split("/")[-1]
    if network_name in ["broadband.info", "BARD.info", "CISN.info"]:
        continue
    print(f"Parse {network}")
    network_code = network.split("/")[-1].split(".")[0]
    print(network_code)
    for xml in tqdm(input_fs.glob(f"{network}/{network_code}.FDSN.xml/{network_code}.*.xml")):
        with input_fs.open(xml) as f:
            inv += obspy.read_inventory(f)

# %%
stations = parse_inventory_csv(inv)

# %%
for network, sta in stations.groupby("network"):
    print(network)
    with output_fs.open(f"{output_bucket}/station/{network}.csv", "wb") as f:
        sta.to_csv(f, index=False)

# %%
# # for network in station_path.glob("*.info"):
# for network in glob(f"{station_path}/*.info"):
#     network_name = network.split("/")[-1]
#     if network_name in ["broadband.info", "BARD.info", "CISN.info"]:
#         continue
#     # if (root_path / "station" / f"{network.stem}.csv").exists():
#     network_stem = network.split("/")[-1].split(".")[0]
#     if os.path.exists(f"{result_path}/{network_stem}.csv"):
#         print(f"Skip {network_stem}")
#         # continue
#     print(f"Parse {network_stem}")
#     inv = obspy.Inventory()
#     # for xml in (network / f"{network_stem}.FDSN.xml").glob(f"{network_stem}.*.xml"):
#     for xml in glob(f"{network}/{network_stem}.FDSN.xml/{network_stem}.*.xml"):
#         inv += obspy.read_inventory(xml)
#     stations = parse_inventory_csv(inv)
#     if len(stations) > 0:
#         stations.to_csv(f"{result_path}/{network_stem}.csv", index=False)

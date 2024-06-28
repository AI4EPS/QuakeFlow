# %%
import os
from datetime import timezone

import fsspec
import obspy
import pandas as pd
from tqdm import tqdm

# %%
input_protocol = "s3"
input_bucket = "scedc-pds"
input_fs = fsspec.filesystem(input_protocol, anon=True)

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset/SC"
output_fs = fsspec.filesystem(output_protocol, token=output_token)

# %%
station_path = f"{input_bucket}/FDSNstationXML"


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
for network in input_fs.glob(f"{station_path}/*"):
    print(f"Parse {network}")
    for xml in tqdm(input_fs.glob(f"{network}/*.xml")):
        with input_fs.open(xml) as f:
            inv += obspy.read_inventory(f)

# %%
stations = parse_inventory_csv(inv)

# %%
for network, sta in stations.groupby("network"):
    with output_fs.open(f"{output_bucket}/station/{network}.csv", "wb") as f:
        sta.to_csv(f, index=False)

# %%

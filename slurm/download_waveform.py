# %%
import obspy
from obspy.clients.fdsn.mass_downloader import CircularDomain, RectangularDomain, Restrictions, MassDownloader
from dataclasses import dataclass
import json
import os
from pathlib import Path
from collections import defaultdict

# %%
result_path = Path("results")
if not result_path.exists():
    result_path.mkdir()

# %%
config = {
    "region": "Ridgecrest",
    "center": [
        -117.504,
        35.705
    ],
    "longitude0": -117.504,
    "latitude0": 35.705,
    "xlim_degree": [-118.504, -116.504],
    "ylim_degree": [34.705, 36.705],
    "maxradius_degree": 1.0,
    "horizontal_degree": 1.0,
    "vertical_degree": 1.0,
    # "starttime": "2019-07-04T17:00:00",
    # "endtime": "2019-07-04T19:00:00",
    "starttime": "2019-07-04T00:00:00",
    "endtime": "2019-07-10T00:00:00",
    "channel_list": [
        "HH[321ENZ]",
        "EH[321ENZ]",
        "HN[321ENZ]",
        "BH[321ENZ]",
    ],
    "provider": ["SCEDC", "IRIS"],
    "degree2km": 111.19492474777779,
}

with open(result_path/"config.json", "w") as fp:
    json.dump(config, fp, indent=4)


# %%
with open(result_path / "config.json", "r") as fp:
    config = json.load(fp)
print(config)


# %%
domain = CircularDomain(
    longitude=config["longitude0"], latitude=config["latitude0"], minradius=0, maxradius=config["maxradius_degree"]
)

restrictions = Restrictions(
    starttime=obspy.UTCDateTime(config["starttime"]),
    endtime=obspy.UTCDateTime(config["endtime"]),
    chunklength_in_sec=3600,
    channel_priorities=config["channel_list"],
    location_priorities=["", "--", "00", "10"],
    minimum_interstation_distance_in_m=1000,
)


def get_mseed_storage(network, station, location, channel, starttime, endtime):
    file_name = f"waveforms/{starttime.strftime('%Y-%j')}/{starttime.strftime('%H')}/{network}.{station}.{location}.{channel}.mseed"
    if os.path.exists(file_name):
        return True
    return file_name


# mdl = MassDownloader(providers=["SCEDC",])
mdl = MassDownloader(
    providers=config["provider"],
)
mdl.download(
    domain,
    restrictions,
    mseed_storage=get_mseed_storage,
    stationxml_storage="stations",
)

# %%
responses = obspy.read_inventory("stations/*xml")


def parse_response(response):
    stations = {}
    for net in response:
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
                stations[f"{net.code}.{sta.code}.{channel[key]['location']}.{channel[key]['device']}"] = {
                    "network": net.code,
                    "station": sta.code,
                    "location": channel[key]["location"],
                    "component": sorted(components[key]),
                    "latitude": channel[key]["latitude"],
                    "longitude": channel[key]["longitude"],
                    "elevation_m": channel[key]["elevation_m"],
                }
    return stations


stations = parse_response(responses)
with open(result_path / "stations.json", "w") as f:
    json.dump(stations, f, indent=4)
# %%

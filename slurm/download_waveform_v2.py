# %%
import obspy
from obspy.clients.fdsn.mass_downloader import CircularDomain, RectangularDomain, Restrictions, MassDownloader, GlobalDomain
from dataclasses import dataclass
import json
import os
from pathlib import Path
from collections import defaultdict

config = {
    "channel_priorities": ('HH[ZNE12]', 'BH[ZNE12]', 'MH[ZNE12]', 'EH[ZNE12]', 'LH[ZNE12]', 'HL[ZNE12]', 'BL[ZNE12]', 'ML[ZNE12]', 'EL[ZNE12]', 'LL[ZNE12]', 'SH[ZNE12]'),
    "location_priorities": ('', '00', '10', '01', '20', '02', '30', '03', '40', '04', '50', '05', '60', '06', '70', '07', '80', '08', '90', '09'),
    "degree2km": 111.1949,
}

# %%
# region = "Ridgecrest"
# config_region = {
#     "region": "Ridgecrest",
#     "center": [
#         -117.504,
#         35.705
#     ],
#     "longitude0": -117.504,
#     "latitude0": 35.705,
#     "xlim_degree": [-118.504, -116.504],
#     "ylim_degree": [34.705, 36.705],
#     "maxradius_degree": 1.0,
#     "horizontal_degree": 1.0,
#     "vertical_degree": 1.0,
#     # "starttime": "2019-07-04T17:00:00",
#     # "endtime": "2019-07-04T19:00:00",
#     "starttime": "2019-07-04T00:00:00",
#     "endtime": "2019-07-10T00:00:00",
#     "channel_list": [
#         "HH[321ENZ]",
#         "EH[321ENZ]",
#         "HN[321ENZ]",
#         "BH[321ENZ]",
#     ],
#     "provider": ["SCEDC", "IRIS"],
#     "degree2km": 111.19492474777779,
# }

# region = "Hawaii_Loa"
# config_region = {
#     # "region": "Hawaii_Loa",
#     region: region,
#     "center": [
#         -155.602737, 
#         19.451827,
#     ],
#     "longitude0": -155.602737,
#     "latitude0": 19.451827,
#     "minlatitude": 19.451827 - 0.3/2, 
#     "maxlatitude": 19.451827 + 0.3/2,
#     "minlongitude": -155.602737 - 0.3/2,
#     "maxlongitude": -155.602737 + 0.3/2,
#     "xlim_degree": [-155.602737 - 0.3/2, -155.602737 + 0.3/2],
#     "ylim_degree": [19.451827 - 0.3/2, 19.451827 + 0.3/2],
#     "maxradius_degree": 0.3,
#     "horizontal_degree": 0.3,
#     "vertical_degree": 0.3,
#     # "starttime": "2019-07-04T17:00:00",
#     # "endtime": "2019-07-04T19:00:00",
#     "starttime": "2022-11-30T22:00:00",
#     "endtime": "2022-11-30T23:00:00",
#     # "endtime": "2023-04-27T00:00:00",
#     "channel_list": [
#         "HH[321ENZ]",
#         "BH[321ENZ]",
#         "EH[321ENZ]",
#         "HN[321ENZ]",
#     ],
#     "provider": ["IRIS"],
#     "degree2km": 111.19492474777779,
# }

region = "South_Pole2"
config_region = {
    "region": region,
    "starttime": "2003-01-01T00:00:00",
    # "starttime": "2020-05-01T00:00:00",
    # "endtime": "2020-06-01T00:00:00",
    # "starttime": "2023-04-01T00:00:00",
    "endtime": "2023-05-01T00:00:00",
    # "minlatitude": -90,
    # "maxlatitude": -80,
    # "minlongitude": -180,
    # "maxlongitude": 180,
    "network": "IU",
    "station": "QSPA",
    "provider": ["IRIS"],
    "channel_priorities": [
        "HH[321ENZ]",
        "BH[321ENZ]",
        # "EH[321ENZ]",
        # "HN[321ENZ]",
    ],
    "degree2km": 111.19492474777779,
}

region = "Montezuma"
config_region = {
    "region": region,
    "starttime": "2020-05-01T00:00:00",
    "endtime": "2023-05-01T00:00:00",
    "minlatitude": 37.9937,
    "maxlatitude": 38.1657,
    "minlongitude": -122.0325,
    "maxlongitude": -121.7275,
    "provider": None,
    "network": None,
    "station": None,
    "degree2km": 111.19492474777779,
}

region = "Kilauea"
config_region = {
    "region": region,
    "starttime": "2018-04-29T00:00:00",
    "endtime": "2018-08-12T00:00:00",
    "minlatitude": 19.41 - 0.1,
    "maxlatitude": 19.41 + 0.1,
    "minlongitude": -155.28 - 0.1,
    "maxlongitude": -155.28 + 0.1,
    "provider": ["IRIS"],
    # "provider": None,
    "network": None,
    "station": None,
    "degree2km": 111.19492474777779,
}

config.update(config_region)

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

    print("Downloading station response...")
    inventory = obspy.Inventory()
    for provider in config["provider"]:
        client = obspy.clients.fdsn.Client(provider)
        if "station" in config:
            inventory += client.get_stations(
                network=config["network"], 
                station=config["station"],
                starttime=config["starttime"], 
                endtime=config["endtime"],
                level="response")
        else:
            inventory += client.get_stations(
                starttime=config["starttime"], 
                endtime=config["endtime"],
                minlatitude=config["minlatitude"],
                maxlatitude=config["maxlatitude"],
                minlongitude=config["minlongitude"],
                maxlongitude=config["maxlongitude"],
                level="response")

    inventory.write(f"{root_path}/response.xml", format="STATIONXML")
    inventory.plot(projection="local", resolution="i", label=False, outfile=f"{root_path}/inventory.png")


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
    minimum_interstation_distance_in_m=1000,
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
mseeds = (root_path / "waveforms").rglob("*.mseed")
mseed_ids = []
for mseed in mseeds:
    mseed_ids.append(mseed.name.split(".mseed")[0][:-1])


# %%
responses = obspy.read_inventory(f"{root_path}/stations/*xml")

def parse_response(response, mseed_ids=None):
    stations = {}
    num = 0
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
                station_id = f"{net.code}.{sta.code}.{channel[key]['location']}.{channel[key]['device']}"
                if station_id not in mseed_ids:
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


stations = parse_response(responses, mseed_ids)
with open(result_path / "stations.json", "w") as f:
    json.dump(stations, f, indent=4)
# %%

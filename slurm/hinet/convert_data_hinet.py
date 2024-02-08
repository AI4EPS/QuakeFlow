# %%
import os
from collections import defaultdict
from glob import glob

import obspy
import pandas as pd
from HinetPy import win32

# %%
root_path = "local"
region = "hinet"
# folder_depth = 2 # year-jday/cntfiles
folder_depth = 3  # year-jday/hour/cntfiles
win32_path = f"{root_path}/{region}/win32"
win32_list = glob(f"{win32_path}/**/*.cnt", recursive=True)

mseed_path = f"{root_path}/{region}/waveforms"
if not os.path.exists(mseed_path):
    os.makedirs(mseed_path)

station_path = f"{root_path}/{region}/results/data"
if not os.path.exists(station_path):
    os.makedirs(station_path)


# %%
def parse_ch(ctable):
    stations = []
    with open(ctable, "r") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("#"):
                continue
            fields = line.split()
            station = {
                "network": "",
                "station": fields[3],
                "location": "",
                "instrument": "",
                "component": fields[4],
                "latitude": float(fields[13]),
                "longitude": float(fields[14]),
                "elevation_m": float(fields[15]),
            }
            stations.append(station)
    return stations


# %%
# stations = []
for cnt in win32_list:
    tmp = cnt.split("/")
    ctable = "/".join(tmp[:-1]) + "/" + tmp[-1][:13] + ".ch"
    # stations.extend(parse_ch(ctable))
    outdir = f"{mseed_path}/{'/'.join(tmp[-folder_depth:-1])}"
    win32.extract_sac(data=cnt, ctable=ctable, suffix="sac", outdir=outdir)


# %%
# stations_raw = pd.DataFrame(stations).copy()
# stations = pd.DataFrame(stations)
# stations = stations.drop_duplicates()
# stations = (
#     stations.groupby(["network", "station"])
#     .agg(
#         {
#             "latitude": "first",
#             "longitude": "first",
#             "elevation_m": "first",
#             "location": "first",
#             "instrument": "first",
#             "component": lambda x: ",".join(sorted(set(x))),
#         }
#     )
#     .reset_index()
# )
# stations["station_id"] = (
#     stations["network"] + "." + stations["station"] + "." + stations["location"] + "." + stations["instrument"]
# )
# stations.to_csv(f"{station_path}/stations.csv", index=False)
# stations.set_index("station_id", inplace=True)
# stations.to_json(f"{station_path}/stations.json", orient="index", indent=2)


# %%
sac_list = sorted(glob(f"{mseed_path}/**/*.sac", recursive=True))

sac_3c = defaultdict(list)
for sac in sac_list:
    sac_3c[sac.replace(".sac", "")[:-1]].append(sac)
sac_3c = [sorted(v) for k, v in sac_3c.items()]

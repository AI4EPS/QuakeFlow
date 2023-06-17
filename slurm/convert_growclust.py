# %%
from pathlib import Path
import h5py
import scipy
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
from datetime import datetime
import os
import argparse

# %%
def parse_args():
    parser = argparse.ArgumentParser()
    ## true
    parser.add_argument("--dtcc", action="store_true", help="run convert_dtcc.py")
    args, unknown = parser.parse_known_args()
    return args

args = parse_args()

# %%
region = "Kilauea"
# region = "Kilauea_debug"
root_path = Path(region)
data_path = root_path / "gamma"
result_path = root_path / "growclust"
if not result_path.exists():
    result_path.mkdir()


# %%
# station_file = Path("templates/stations_filtered.csv")
# station_df = pd.read_csv(station_file)
# station_file = Path("results/stations.json")
# station_df = pd.read_json(station_file, orient="index")
station_json = root_path / "obspy" / "stations.json"
station_df = pd.read_json(station_json, orient="index")

lines = []
for i, row in station_df.iterrows():
    # line = f"{row['network']}{row['station']:<4} {row['latitude']:.4f} {row['longitude']:.4f}\n"
    line = f"{row['station']:<4} {row['latitude']:.4f} {row['longitude']:.4f}\n"
    lines.append(line)

with open(result_path / "stlist.txt", "w") as fp:
    fp.writelines(lines)


# %%
catalog_file = data_path / "gamma_catalog.csv"
catalog_df = pd.read_csv(catalog_file)
# catalog_df = catalog_df[catalog_df["gamma_score"] > 10]
# event_index = [f"{x:06d}" for x in catalog_df["event_index"]]

catalog_df[["year", "month", "day", "hour", "minute", "second"]] = (
    catalog_df["time"]
    .apply(lambda x: datetime.fromisoformat(x).strftime("%Y %m %d %H %M %S.%f").split(" "))
    .apply(pd.Series)
    .apply(pd.to_numeric)
)

lines = []
for i, row in catalog_df.iterrows():
    # yr mon day hr min sec lat lon dep mag eh ez rms evid
    line = f"{row['year']:4d} {row['month']:2d} {row['day']:2d} {row['hour']:2d} {row['minute']:2d} {row['second']:7.3f} {row['latitude']:.4f} {row['longitude']:.4f} {row['depth(m)']/1e3:7.3f} {row['magnitude']:.2f} 0.000 0.000 0.000 {row['event_index']:6d}\n"
    lines.append(line)

with open(result_path / "evlist.txt", "w") as fp:
    fp.writelines(lines)

# %%
if not args.dtcc:
    dt_ct = root_path / "hypodd" / "dt.ct"
    lines = []
    with open(dt_ct, "r") as fp:
        for line in tqdm(fp):
            if line.startswith("#"):
                ev1, ev2 = line.split()[1:3]
                lines.append(f"# {ev1} {ev2} 0.000\n")
            else:
                station, t1, t2, score, phase = line.split()
                lines.append(f"{station} {float(t1)-float(t2):.5f} {score} {phase}\n")

    with open(result_path / "dt.ct", "w") as fp:
        fp.writelines(lines)

# %%
if args.dtcc:
    if not os.path.exists("relocation/growclust/dt.cc"):
        os.system("python convert_dtcc.py")
    os.system("cp templates/dt.cc relocation/growclust/")
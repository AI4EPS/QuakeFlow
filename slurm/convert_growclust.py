# %%
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", nargs="?", type=str, default="local", help="root path")
    parser.add_argument("region", nargs="?", type=str, default="demo", help="region")
    parser.add_argument("--dtct", action="store_true")
    args, unknown = parser.parse_known_args()
    return args


args = parse_args()

# %%
root_path = args.root_path
region = args.region
result_path = f"{region}/growclust"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")

# %%
station_json = f"{region}/obspy/stations.json"
station_df = pd.read_json(f"{root_path}/{station_json}", orient="index")

lines = []
for i, row in station_df.iterrows():
    # line = f"{row['network']}{row['station']:<4} {row['latitude']:.4f} {row['longitude']:.4f}\n"
    line = f"{row['station']:<4} {row['latitude']:.4f} {row['longitude']:.4f}\n"
    lines.append(line)

with open(f"{root_path}/{result_path}/stlist.txt", "w") as fp:
    fp.writelines(lines)


# %%
catalog_file = f"{region}/gamma/gamma_events.csv"
# catalog_file = f"{region}/cctorch/events.csv"
catalog_df = pd.read_csv(f"{root_path}/{catalog_file}")
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
    line = f"{row['year']:4d} {row['month']:2d} {row['day']:2d} {row['hour']:2d} {row['minute']:2d} {row['second']:7.3f} {row['latitude']:.4f} {row['longitude']:.4f} {row['depth_km']:7.3f} {row['magnitude']:.2f} 0.000 0.000 0.000 {row['event_index']:6d}\n"
    lines.append(line)

with open(f"{root_path}/{result_path}/evlist.txt", "w") as fp:
    fp.writelines(lines)

# %%
if args.dtct:
    dt_ct = f"{root_path}/{region}/hypodd/dt.ct"
    lines = []
    with open(dt_ct, "r") as fp:
        for line in tqdm(fp):
            if line.startswith("#"):
                ev1, ev2 = line.split()[1:3]
                lines.append(f"# {ev1} {ev2} 0.000\n")
            else:
                station, t1, t2, score, phase = line.split()
                lines.append(f"{station} {float(t1)-float(t2):.5f} {score} {phase}\n")

    with open(f"{root_path}/{result_path}/dt.ct", "w") as fp:
        fp.writelines(lines)

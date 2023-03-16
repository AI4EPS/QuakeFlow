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

# %%
output_path = Path("relocation/growclust/")
if not output_path.exists():
    output_path.mkdir(parents=True)


# %%
station_file = Path("templates/stations_filtered.csv")
# station_df = pd.read_json(station_file, orient="index")
station_df = pd.read_csv(station_file)

lines = []
for i, row in station_df.iterrows():
    # line = f"{row['network']}{row['station']:<4} {row['latitude']:.4f} {row['longitude']:.4f}\n"
    line = f"{row['station']:<4} {row['latitude']:.4f} {row['longitude']:.4f}\n"
    lines.append(line)

with open(output_path / "stlist.txt", "w") as fp:
    fp.writelines(lines)


# %%
catalog_file = Path("results/gamma_catalog.csv")
catalog_df = pd.read_csv(catalog_file)

# %%
catalog_df[["year", "month", "day", "hour", "minute", "second"]] = (
    catalog_df["time"]
    .apply(lambda x: datetime.fromisoformat(x).strftime("%Y %m %d %H %M %S.%f").split(" "))
    .apply(pd.Series)
    .apply(pd.to_numeric)
)

# %%
lines = []
for i, row in catalog_df.iterrows():
    # yr mon day hr min sec lat lon dep mag eh ez rms evid
    line = f"{row['year']:4d} {row['month']:2d} {row['day']:2d} {row['hour']:2d} {row['minute']:2d} {row['second']:7.3f} {row['latitude']:.4f} {row['longitude']:.4f} {row['depth(m)']/1e3:7.3f} {row['magnitude']:.2f} 0.000 0.000 0.000 {row['event_index']:6d}\n"
    lines.append(line)

with open(output_path / "evlist.txt", "w") as fp:
    fp.writelines(lines)
# %%

os.system("python convert_dtcc.py")
os.system("cp templates/dt.cc relocation/growclust/")
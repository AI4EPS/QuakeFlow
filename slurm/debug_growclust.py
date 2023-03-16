# %%
from pathlib import Path
import h5py
import scipy
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
from datetime import datetime

# %%
output_path = Path("relocation/growclust/")
if not output_path.exists():
    output_path.mkdir(parents=True)

# %%
dt_ct = Path("relocation/hypodd/dt.ct")

lines = []
with open(dt_ct, "r") as fp:
    for line in fp:
        if line.startswith("#"):
            ev1, ev2 = line.split()[1:3]
            lines.append(f"# {ev1} {ev2} 0.000\n")
        else:
            station, t1, t2, score, phase = line.split()
            station = station[:-2]
            if station in ["WAS2", "FUR", "RRX"]:
                continue
            lines.append(f"{station} {float(t1)-float(t2):.5f} {score} {phase}\n")

# %%
# with open(result_path/"dt.cc", "w") as fp:
with open(output_path / "xcordata_debug.txt", "w") as fp:
    fp.writelines(lines)

# # %%
# catalog_file = Path("results/gamma_catalog.csv")
# catalog_df = pd.read_csv(catalog_file)

# # %%
# catalog_df[["year", "month", "day", "hour", "minute", "second"]] = (
#     catalog_df["time"]
#     .apply(lambda x: datetime.fromisoformat(x).strftime("%Y %m %d %H %M %S.%f").split(" "))
#     .apply(pd.Series)
#     .apply(pd.to_numeric)
# )

# # %%
# lines = []
# for i, row in catalog_df.iterrows():
#     # yr mon day hr min sec lat lon dep mag eh ez rms evid
#     line = f"{row['year']:4d} {row['month']:2d} {row['day']:2d} {row['hour']:2d} {row['minute']:2d} {row['second']:7.3f} {row['latitude']:.4f} {row['longitude']:.4f} {row['depth(m)']/1e3:7.3f} {row['magnitude']:.2f} 0.000 0.000 0.000 {row['event_index']:6d}\n"
#     lines.append(line)

# with open(output_path / "evlist.txt", "w") as fp:
#     fp.writelines(lines)
# # %%

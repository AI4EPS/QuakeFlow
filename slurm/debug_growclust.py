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
    for line in tqdm(fp):
        if line.startswith("#"):
            ev1, ev2 = line.split()[1:3]
            lines.append(f"# {ev1} {ev2} 0.000\n")
        else:
            station, t1, t2, score, phase = line.split()
            #station = station[:-2]
            # if station in ["WAS2", "FUR", "RRX"]:
            #     continue
            lines.append(f"{station} {float(t1)-float(t2):.5f} {score} {phase}\n")

# %%
with open(output_path / "dt.ct", "w") as fp:
    fp.writelines(lines)


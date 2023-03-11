# %%
from pathlib import Path
import h5py
import scipy
from tqdm import tqdm
import numpy as np
import json

# %%
template_path = Path("templates/")
with open(template_path/"config.json", "r") as fp:  
    config = json.load(fp)

config["min_cc_score"] = 0.8

# %%
traveltime_file = Path("templates/travel_time.dat")
tt_memmap = np.memmap(traveltime_file, dtype=np.float32, mode='r', shape=tuple(config["traveltime_shape"]))

# %%
h5_path = Path("ccpairs/")
h5_list = sorted(list(h5_path.rglob("*.h5")))

result_path = Path("results")

# %%
data = {}
dt = 0.01
dt_cubic = dt/100
x = np.linspace(0, 1, 3) 
xs = np.linspace(0, 1, 3*int(dt/dt_cubic))
num_channel = 3
phase_list = ["P", "S"]

for h5 in h5_list:
    with h5py.File(h5, "r") as fp:
        for id1 in tqdm(fp):
            
            gp1 = fp[id1]
            for id2 in gp1:
                
                cc_score = gp1[id2]["cc_score"][:]
                cc_index = gp1[id2]["cc_index"][:]
                cc_weight = gp1[id2]["cc_weight"][:]
                
                neighbor_score = gp1[id2]["neighbor_score"][:]

                cubic_score = scipy.interpolate.interp1d(x, neighbor_score, axis=-1, kind="quadratic")(xs)
                cubic_index = (np.argmax(cubic_score, axis=-1, keepdims=True) - (len(xs)//2-1))
                dt_cc = cc_index * dt + cubic_index * dt_cubic

                key = (id1, id2)
                nch, nsta, npick = cc_score.shape
                records = []
                for i in range(nch//num_channel):
                    for j in range(nsta):
                        dt_ct = tt_memmap[int(id1)][i,j] - tt_memmap[int(id2)][i,j]
                        best = np.argmax(cc_score[i*num_channel:(i+1)*num_channel, j, 0]) + i*num_channel
                        if cc_score[best, j, 0] > config["min_cc_score"]:
                            records.append([f"{j:05d}", dt_ct + dt_cc[best, j, 0], cc_score[best, j, 0]*cc_weight[best, j], phase_list[i]])

                data[key] = records
            
# %%
with open(result_path/"dt.cc", "w") as fp: 
    for key in data:
        fp.write(f"# {key[0]} {key[1]}\n")
        for record in data[key]:
            fp.write(f"{record[0]} {record[1]:.4f} {record[2]:.4f} {record[3]}\n")
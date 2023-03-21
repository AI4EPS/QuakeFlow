# %%
from pathlib import Path
import h5py
import scipy
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
from datetime import datetime
import multiprocess as mp

# %%
output_path = Path("templates/")
if not output_path.exists():
    output_path.mkdir(parents=True)


template_path = Path("templates/")
with open(template_path / "config.json", "r") as fp:
    config = json.load(fp)

config["min_cc_score"] = 0.6
config["min_cc_weight"] = 0.3

# %%
traveltime_file = Path("templates/travel_time.dat")
tt_memmap = np.memmap(traveltime_file, dtype=np.float32, mode="r", shape=tuple(config["traveltime_shape"]))

# %%
h5_path = Path("templates/ccpairs/")
h5_list = sorted(list(h5_path.rglob("*.h5")))

# %%
station_file = Path("templates/stations_filtered.csv")
# station_df = pd.read_json(station_file, orient="index")
station_df = pd.read_csv(station_file)

lines = []
for i, row in station_df.iterrows():
    # tmp = f"{row['network']}{row['station']}"
    tmp = f"{row['station']}"
    line = f"{tmp:<4} {row['latitude']:.4f} {row['longitude']:.4f}\n"
    lines.append(line)

with open(output_path / "stlist.txt", "w") as fp:
    fp.writelines(lines)

# %%
dt = 0.01
dt_cubic = dt / 100
x = np.linspace(0, 1, 2 + 1)
xs = np.linspace(0, 1, 2 * int(dt / dt_cubic) + 1)
num_channel = 3
phase_list = ["P", "S"]

# data = {}
# for h5 in h5_list:
#     with h5py.File(h5, "r") as fp:

#         for id1 in tqdm(fp):

#             gp1 = fp[id1]
#             for id2 in gp1:

#                 cc_score = gp1[id2]["cc_score"][:]
#                 cc_index = gp1[id2]["cc_index"][:]
#                 cc_weight = gp1[id2]["cc_weight"][:]

#                 neighbor_score = gp1[id2]["neighbor_score"][:]

#                 cubic_score = scipy.interpolate.interp1d(x, neighbor_score, axis=-1, kind="quadratic")(xs)
#                 cubic_index = np.argmax(cubic_score, axis=-1, keepdims=True) - (len(xs) // 2 - 1)
#                 dt_cc = cc_index * dt + cubic_index * dt_cubic

#                 key = (id1, id2)
#                 if id1 > id2:
#                     continue
#                 nch, nsta, npick = cc_score.shape
#                 records = []
#                 for i in range(nch // num_channel):
#                     for j in range(nsta):
#                         dt_ct = tt_memmap[int(id1)][i, j] - tt_memmap[int(id2)][i, j]
#                         best = np.argmax(cc_score[i * num_channel : (i + 1) * num_channel, j, 0]) + i * num_channel
#                         if cc_score[best, j, 0] > config["min_cc_score"]:
#                             # records.append([f"{j:05d}", dt_ct + dt_cc[best, j, 0], cc_score[best, j, 0]*cc_weight[best, j], phase_list[i]])
#                             if cc_weight[best, j] > config["min_cc_weight"]:
#                                 records.append(
#                                     [
#                                         # f"{station_df.iloc[j]['network']}{station_df.iloc[j]['station']:<4}",
#                                         f"{station_df.iloc[j]['station']:<4}",
#                                         # dt_ct,
#                                         dt_ct + dt_cc[best, j, 0],
#                                         # cc_score[best, j, 0] * cc_weight[best, j],
#                                         cc_score[best, j, 0],
#                                         phase_list[i],
#                                     ]
#                                 )

#                 data[key] = records


# %%
def extract_picks(pair, data):

    h5, id1, id2 = pair
    if int(id1) > int(id2):
        return 0

    with h5py.File(h5, "r") as fp:
    
        cc_score = fp[id1][id2]["cc_score"][:]
        cc_index = fp[id1][id2]["cc_index"][:]
        cc_weight = fp[id1][id2]["cc_weight"][:]

        neighbor_score = fp[id1][id2]["neighbor_score"][:]

        cubic_score = scipy.interpolate.interp1d(x, neighbor_score, axis=-1, kind="quadratic")(xs)
        cubic_index = np.argmax(cubic_score, axis=-1, keepdims=True) - len(xs) // 2
        dt_cc = cc_index * dt + cubic_index * dt_cubic

        
        key = (id1, id2)
        nch, nsta, npick = cc_score.shape
        records = []
        for i in range(nch // num_channel):
            for j in range(nsta):
                dt_ct = tt_memmap[int(id1)][i, j] - tt_memmap[int(id2)][i, j]
                best = np.argmax(cc_score[i * num_channel : (i + 1) * num_channel, j, 0]) + i * num_channel
                if cc_score[best, j, 0] > config["min_cc_score"]:
                    # records.append([f"{j:05d}", dt_ct + dt_cc[best, j, 0], cc_score[best, j, 0]*cc_weight[best, j], phase_list[i]])
                    if cc_weight[best, j] > config["min_cc_weight"]:
                        records.append(
                            [
                                # f"{station_df.iloc[j]['network']}{station_df.iloc[j]['station']:<4}",
                                f"{station_df.iloc[j]['station']:<4}",
                                # dt_ct,
                                dt_ct + dt_cc[best, j, 0],
                                # cc_score[best, j, 0] * cc_weight[best, j],
                                cc_score[best, j, 0],
                                phase_list[i],
                            ]
                        )

        data[key] = records

    if len(data) % 1000 == 0:
        print(f"{len(data)} pairs processed.")

    return 0

# %%
with mp.Manager() as manager:

    data = manager.dict()
    pair_list = []
    for h5 in h5_list:
        with h5py.File(h5, "r") as fp:
            for id1 in tqdm(fp):
                gp1 = fp[id1]
                for id2 in gp1:
                    pair_list.append((h5, id1, id2))

    print(f"Initial number of pair: {len(pair_list)//2}")
    ncpu = max(1, mp.cpu_count()-1)
    with mp.Pool(processes=ncpu) as pool:
        pool.starmap(extract_picks, [(pair, data) for pair in pair_list])

    data = dict(data)

# %%
with open(output_path / "dt.cc", "w") as fp:
    for key in tqdm(sorted(data.keys())):
        fp.write(f"# {key[0]} {key[1]} 0.000\n")
        for record in data[key]:
            fp.write(f"{record[0]} {record[1]: .4f} {record[2]:.4f} {record[3]}\n")

# %%

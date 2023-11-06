# %%
import json
import multiprocessing as mp
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm


# %%
def extract_picks(pair, data, config, tt_memmap, station_df):
    # h5, id1, id2 = pair
    # h5, id1, id2_list = pair
    h5, id1 = pair

    x = config["interp"]["x"]
    x_interp = config["interp"]["x_interp"]
    dt = config["interp"]["dt"]
    dt_interp = config["interp"]["dt_interp"]
    min_cc_score = config["min_cc_score"]
    min_cc_weight = config["min_cc_weight"]
    num_channel = config["num_channel"]
    phase_list = config["phase_list"]

    with h5py.File(h5, "r") as fp:
        gp = fp[id1]

        for id2 in gp:
            if int(id1) > int(id2):
                continue

            ds = gp[id2]
            cc_score = ds["cc_score"][:]
            cc_index = ds["cc_index"][:]
            cc_weight = ds["cc_weight"][:]
            neighbor_score = ds["neighbor_score"][:]

            if (np.max(cc_score) < min_cc_score) or (np.max(cc_weight) < min_cc_weight):
                continue

            cubic_score = scipy.interpolate.interp1d(x, neighbor_score, axis=-1, kind="quadratic")(x_interp)
            cubic_index = np.argmax(cubic_score, axis=-1, keepdims=True) - len(x_interp) // 2
            dt_cc = cc_index * dt + cubic_index * dt_interp

            key = (id1, id2)
            nch, nsta, npick = cc_score.shape
            records = []
            for i in range(nch // num_channel):
                for j in range(nsta):
                    dt_ct = tt_memmap[int(id1)][i, j] - tt_memmap[int(id2)][i, j]
                    best = np.argmax(cc_score[i * num_channel : (i + 1) * num_channel, j, 0]) + i * num_channel
                    if cc_score[best, j, 0] > min_cc_score:
                        # records.append([f"{j:05d}", dt_ct + dt_cc[best, j, 0], cc_score[best, j, 0]*cc_weight[best, j], phase_list[i]])
                        if cc_weight[best, j] > min_cc_weight:
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

            if len(records) > 0:
                data[key] = records

    return 0


if __name__ == "__main__":
    # %%
    root_path = "local"
    region = "demo"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    # %%
    cctorch_path = f"{region}/cctorch"

    # %%
    with open(f"{root_path}/{cctorch_path}/config.json", "r") as fp:
        config = json.load(fp)
    config["min_cc_score"] = 0.6
    config["min_cc_weight"] = 0.2

    tt_memmap = np.memmap(
        f"{root_path}/{cctorch_path}/traveltime.dat",
        dtype=np.float32,
        mode="r",
        shape=tuple(config["traveltime_shape"]),
    )

    # %%
    station_df = pd.read_csv(f"{root_path}/{cctorch_path}/stations_filtered.csv")

    lines = []
    for i, row in station_df.iterrows():
        # tmp = f"{row['network']}{row['station']}"
        tmp = f"{row['station']}"
        line = f"{tmp:<4} {row['latitude']:.4f} {row['longitude']:.4f}\n"
        lines.append(line)

    with open(f"{root_path}/{cctorch_path}/stlist.txt", "w") as fp:
        fp.writelines(lines)

    h5_list = sorted(list(glob(f"{root_path}/{cctorch_path}/ccpairs/*.h5")))

    # %%
    dt = 0.01
    dt_interp = dt / 100
    x = np.linspace(0, 1, 2 + 1)
    x_interp = np.linspace(0, 1, 2 * int(dt / dt_interp) + 1)
    num_channel = 3
    phase_list = ["P", "S"]

    config["interp"] = {"x": x, "x_interp": x_interp, "dt": dt, "dt_interp": dt_interp}
    config["num_channel"] = num_channel
    config["phase_list"] = phase_list

    # %%
    with mp.Manager() as manager:
        data = manager.dict()
        pair_list = []
        num_pair = 0
        for h5 in h5_list:
            with h5py.File(h5, "r") as fp:
                for id1 in tqdm(fp, desc=f"Loading {h5.split('/')[-1]}"):
                    gp1 = fp[id1]
                    # for id2 in gp1:
                    #     pair_list.append((h5, id1, id2))
                    # pair_list.append([h5, id1, list(gp1.keys())])
                    pair_list.append([h5, id1])
                    num_pair += len(gp1.keys())

        print(f"Total pairs: {num_pair}")
        ncpu = max(1, min(8, mp.cpu_count() - 1))
        pbar = tqdm(total=len(pair_list), desc="Extracting pairs")

        # for pair in pair_list:
        #     extract_picks(pair, data, config)
        #     pbar.update()

        with mp.get_context("spawn").Pool(processes=ncpu) as pool:
            # with mp.Pool(processes=ncpu) as pool:
            for pair in pair_list:
                pool.apply_async(
                    extract_picks, args=(pair, data, config, tt_memmap, station_df), callback=lambda x: pbar.update()
                )
            pool.close()
            pool.join()
        pbar.close()

        data = dict(data)
        print(f"Valid pairs: {len(data)}")

    # %%
    with open(f"{root_path}/{cctorch_path}/dt.cc", "w") as fp:
        for key in tqdm(sorted(data.keys()), desc="Writing dt.cc"):
            fp.write(f"# {key[0]} {key[1]} 0.000\n")
            for record in data[key]:
                fp.write(f"{record[0]} {record[1]: .4f} {record[2]:.4f} {record[3]}\n")

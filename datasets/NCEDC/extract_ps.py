# %%
import multiprocessing as mp
import os

import h5py
from tqdm import tqdm

# %%
h5_dir = "waveform_h5"
h5_files = os.listdir(h5_dir)

# %%
result_path = "waveform_ps_h5"
if not os.path.exists(result_path):
    os.makedirs(result_path)


# %%
def run(h5_file):
    h5_input = os.path.join(h5_dir, h5_file)
    h5_output = os.path.join(result_path, h5_file)
    pos = 2022 - int(h5_file.split("/")[-1].split(".")[0])
    with h5py.File(h5_input, "r") as fin:
        with h5py.File(h5_output, "w") as fout:
            for event in tqdm(fin.keys(), desc=h5_file, total=len(fin.keys()), position=pos, leave=True):
                # copy event and attributes
                gp = fout.create_group(event)
                for key in fin[event].attrs.keys():
                    gp.attrs[key] = fin[event].attrs[key]
                num_station = 0
                for station in fin[event].keys():
                    if "S" in fin[event][station].attrs["phase_type"]:
                        ds = gp.create_dataset(station, data=fin[event][station])
                        for key in fin[event][station].attrs.keys():
                            ds.attrs[key] = fin[event][station].attrs[key]
                        num_station += 1
                    else:
                        continue
                gp.attrs["nx"] = num_station


# %%
if __name__ == "__main__":
    # run(0, h5_files[0])

    ncpu = len(h5_files)
    print(f"Using {ncpu} CPUs")
    with mp.Pool(ncpu) as p:
        p.map(run, h5_files)

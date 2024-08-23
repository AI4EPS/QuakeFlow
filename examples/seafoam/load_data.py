# %%
import multiprocessing as mp

import fsspec
import h5py
from tqdm import tqdm


# %%
def read_hdf5(file, key_path):
    fs = fsspec.filesystem("gcs", token=key_path)
    with fs.open(file, "rb") as f:
        with h5py.File(f, "r") as hf:
            print(file.split("/")[-1], hf["Acquisition/Raw[0]/RawData"])


# %%
if __name__ == "__main__":

    # %%
    token_file = ""

    fs = fsspec.filesystem("gcs", token=token_file)

    folders = fs.ls("berkeley-mbari-das/")

    hdf5_files = []
    for folder in folders:
        if folder.split("/")[-1] in ["ContextData", "MBARI_cable_geom_dx10m.csv"]:
            continue
        years = fs.ls(folder)
        for year in tqdm(years, desc=folder):
            jdays = fs.ls(year)
            for jday in jdays:
                files = fs.ls(jday)
                for file in files:
                    if file.endswith(".h5"):
                        hdf5_files.append(file)

    # %%
    # for file in hdf5_files:
    #     read_hdf5(file, key_path)

    ctx = mp.get_context("spawn")
    pbar = tqdm(total=len(hdf5_files))
    ncpu = 8
    with ctx.Pool(ncpu) as pool:
        jobs = []
        for file in hdf5_files:
            job = pool.apply_async(read_hdf5, (file, key_path), callback=lambda _: pbar.update())
        pool.close()
        pool.join()

        for job in jobs:
            result = job.get()
            if result:
                print(result)

    pbar.close()

# %%

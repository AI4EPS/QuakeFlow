# %%
import multiprocessing as mp
import os
from glob import glob

import pandas as pd
from tqdm import tqdm


def merge_csv(csv_files, pick_file):
    keep_header = True
    for csv_file in csv_files:
        if os.stat(csv_file).st_size == 0:
            continue
        if keep_header:
            cmd = f"cat {csv_file} > {pick_file}"
            keep_header = False
        else:
            cmd = f"tail -n +2 {csv_file} >> {pick_file}"
        os.system(cmd)


# %%
if __name__ == "__main__":
    # %%
    csv_path = "local/hinet/phasenet/csvs"
    pick_path = "local/hinet/phasenet/picks"
    if not os.path.exists(pick_path):
        os.makedirs(pick_path)

    # %%
    jdays = sorted(os.listdir(csv_path))

    # %%
    ncpu = min(32, mp.cpu_count())
    ctx = mp.get_context("spawn")
    pbar = tqdm(total=len(jdays))

    # %%
    jobs = []
    with ctx.Pool(ncpu) as pool:

        # %%
        for jday in jdays:
            csv_files = []
            for hour in sorted(os.listdir(f"{csv_path}/{jday}")):
                tmp = glob(f"{csv_path}/{jday}/{hour}/*.csv")
                csv_files.extend(tmp)

            year, jday = jday.split("-")
            if not os.path.exists(f"local/hinet/phasenet/picks/{year}"):
                os.makedirs(f"local/hinet/phasenet/picks/{year}")
            pick_file = f"local/hinet/phasenet/picks/{year}/{jday}.csv"

            # merge_csv(csv_files, pick_file)
            job = pool.apply_async(merge_csv, (csv_files, pick_file), callback=lambda _: pbar.update(1))
            jobs.append(job)

        pool.close()
        pool.join()

        for job in jobs:
            out = job.get()
            if out is not None:
                print(out)

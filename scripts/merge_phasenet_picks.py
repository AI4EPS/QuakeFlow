# %%
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread

import fsspec
import numpy as np
import pandas as pd
import pyproj
from obspy import read_inventory
from obspy.clients.fdsn import Client
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from args import parse_args
from glob import glob


def scan_csv(year, root_path, fs=None, bucket=None, protocol="file"):
    # %%
    csv_list = []
    if protocol != "file":
        jdays = fs.ls(f"{bucket}/{region}/{folder}/{year}")
    else:
        jdays = os.listdir(f"{root_path}/{region}/phasenet/picks/{year}/")

    for jday in jdays:
        if protocol != "file":
            csvs = fs.glob(f"{jday}/??/*.csv")
        else:
            csvs = glob(f"{root_path}/{region}/phasenet/picks/{year}/{jday}/??/*.csv")

        csv_list.extend([[year, jday, csv] for csv in csvs])

    csvs = pd.DataFrame(csv_list, columns=["year", "jday", "csv"])
    csv_file = f"{root_path}/{region}/phasenet/csv_list_{year}.csv"
    csvs.to_csv(csv_file, index=False)

    return csv_file


# %%
def read_csv(rows, region, year, jday, root_path, fs=None, bucket=None):

    picks = []
    for i, row in rows.iterrows():
        # if fs.info(row["csv"])["size"] == 0:
        #     continue
        # with fs.open(row["csv"], "r") as f:
        #     picks_ = pd.read_csv(f, dtype=str)
        if os.path.getsize(row["csv"]) == 0:
            continue
        with open(row["csv"], "r") as f:
            picks_ = pd.read_csv(f, dtype=str)
        picks.append(picks_)

    if len(picks) > 0:
        picks = pd.concat(picks, ignore_index=True)
        if not os.path.exists(f"{root_path}/{region}/phasenet/{year}"):
            os.makedirs(f"{root_path}/{region}/phasenet/{year}", exist_ok=True)
        picks.to_csv(f"{root_path}/{region}/phasenet/{year}/{year}.{jday}.csv", index=False)
        # fs.put(
        #     f"{root_path}/{region}/phasenet/{year}/{jday}/{year}.{jday}.csv",
        #     f"{bucket}/{region}/phasenet_merged/{year}/{year}.{jday}.csv",
        # )
    else:
        with open(f"{root_path}/{region}/phasenet/{year}/{year}.{jday}.csv", "w") as f:
            f.write("")


# %%
if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region

    data_path = f"{region}/phasenet/picks"
    result_path = f"{region}/phasenet"

    # %%
    # protocol = "gs"
    # token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    # with open(token_json, "r") as fp:
    #     token = json.load(fp)
    # fs = fsspec.filesystem(protocol, token=token)

    # %%
    years = os.listdir(f"{root_path}/{region}/phasenet/picks")

    for year in years:

        csv_list = scan_csv(year, root_path)

        # %%
        csv_list = pd.read_csv(csv_list, dtype=str)

        # for jday, csvs in csv_list.groupby("jday"):
        #     read_csv(csvs, region, year, jday, root_path)
        #     raise

        # ncpu = os.cpu_count()
        ncpu = 64
        print(f"Number of processors: {ncpu}")
        csv_by_jday = csv_list.groupby("jday")
        pbar = tqdm(total=len(csv_by_jday), desc=f"Loading csv files (year {year})")

        # with mp.Pool(ncpu) as pool:
        ctx = mp.get_context("spawn")
        with ctx.Pool(ncpu) as pool:
            jobs = []
            for jday, csvs in csv_by_jday:
                job = pool.apply_async(
                    read_csv, (csvs, region, year, jday, root_path), callback=lambda _: pbar.update()
                )
                jobs.append(job)
            pool.close()
            pool.join()
            for job in jobs:
                output = job.get()
                if output:
                    print(output)

        pbar.close()

    # %%
    csvs = glob(f"{root_path}/{region}/phasenet/????/????.???.csv")
    picks = []
    for csv in tqdm(csvs, desc="Merge csv files"):
        picks.append(pd.read_csv(csv, dtype=str))
    picks = pd.concat(picks, ignore_index=True)
    picks.to_csv(f"{root_path}/{region}/phasenet/phasenet_picks.csv", index=False)

# %%

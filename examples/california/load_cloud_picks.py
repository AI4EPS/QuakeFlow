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


def scan_csv(regions, year, root_path, fs, bucket, folder="phasenet"):
    # %%
    csv_list = []
    for region in regions:
        if region == "SC":
            if fs.exists(f"{bucket}/{region}/{folder}/{year}"):
                jdays = fs.ls(f"{bucket}/{region}/{folder}/{year}")
                for jday in tqdm(jdays, desc=f"Scan csv files: {bucket}/{region}/{folder}/{year}"):
                    csvs = fs.glob(f"{jday}/*.csv")
                    # csv_list.append([region, year, jday, ",".join(csvs)])
                    jday = jday.split("/")[-1].split("_")[-1]
                    csv_list.extend([[region, year, jday, csv] for csv in csvs])
        elif region == "NC":
            networks = fs.ls(f"{bucket}/{region}/{folder}")
            for network in networks:
                if fs.exists(f"{network}/{year}"):
                    jdays = fs.ls(f"{network}/{year}")
                    for jday in tqdm(jdays, desc=f"Scan csv files: {network}/{year}"):
                        csvs = fs.glob(f"{jday}/*.csv")
                        jday = jday.split("/")[-1].split(".")[-1]
                        csv_list.extend([[region, year, jday, csv] for csv in csvs])

    csv_list = pd.DataFrame(csv_list, columns=["region", "year", "jday", "csv"])
    csv_list.to_csv(f"{root_path}/Cal/{folder}/csv_list/{year}.csv", index=False)
    fs.put(f"{root_path}/Cal/{folder}/csv_list/{year}.csv", f"{bucket}/Cal/{folder}/csv_list/{year}.csv")

    for region in regions:
        csv_list[csv_list["region"] == region].to_csv(f"{root_path}/{region}/{folder}/csv_list/{year}.csv", index=False)
        fs.put(f"{root_path}/{region}/{folder}/csv_list/{year}.csv", f"{bucket}/{region}/{folder}/csv_list/{year}.csv")


# %%
def read_csv(rows, year, jday, root_path, fs, bucket, folder="phasenet", regions=["NC", "SC"]):
    picks_total = []
    for region in regions:
        picks = []
        rows_ = rows[rows["region"] == region]
        # for i, row in tqdm(
        # rows_.iterrows(), total=len(rows_), desc=f"Reading csv files: {region}/{folder}/{year}.{jday}"
        # ):
        for i, row in rows_.iterrows():
            if fs.info(row["csv"])["size"] == 0:
                continue
            with fs.open(row["csv"], "r") as f:
                picks_ = pd.read_csv(f, dtype=str)
            picks.append(picks_)

        if len(picks) == 0:
            continue
        picks = pd.concat(picks, ignore_index=True)
        if not os.path.exists(f"{root_path}/{region}/{folder}/{year}"):
            os.makedirs(f"{root_path}/{region}/{folder}/{year}", exist_ok=True)
        picks.to_csv(f"{root_path}/{region}/{folder}/{year}/{year}.{jday}.csv", index=False)
        fs.put(
            f"{root_path}/{region}/{folder}/{year}/{year}.{jday}.csv",
            f"{bucket}/{region}/{folder}_merged/{year}/{year}.{jday}.csv",
        )
        picks["source"] = region
        picks_total.append(picks)

    if len(picks_total) > 0:
        picks_total = pd.concat(picks_total, ignore_index=True)
        if not os.path.exists(f"{root_path}/Cal/{folder}/{year}"):
            os.makedirs(f"{root_path}/Cal/{folder}/{year}", exist_ok=True)
        picks_total.to_csv(f"{root_path}/Cal/{folder}/{year}/{year}.{jday}.csv", index=False)
        fs.put(
            f"{root_path}/Cal/{folder}/{year}/{year}.{jday}.csv",
            f"{bucket}/Cal/{folder}_merged/{year}/{year}.{jday}.csv",
        )


# %%
if __name__ == "__main__":

    # %%
    root_path = "local"
    regions = ["NC", "SC"]
    # year = "2023"
    # years = range(2017, 2021)
    # years = range(2011, 2017)
    # years = range(2008, 2011)
    # years = range(2005, 2008)
    years = range(1999, 2005)
    bucket = "quakeflow_catalog"
    folder = "phasenet"

    # %%
    dirs = [root_path]
    for region in regions + ["Cal"]:
        dirs.extend(
            [
                f"{root_path}/{region}/phasenet",
                f"{root_path}/{region}/phasenet/csv_list",
            ]
        )

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # %%
    protocol = "gs"
    token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)
    fs = fsspec.filesystem(protocol, token=token)

    # %%
    for year in years:
        # if not os.path.exists(f"{root_path}/Cal/{folder}/csv_list/{year}.csv"):
        scan_csv(regions, year, root_path, fs, bucket, folder)

        # %%
        csv_list = pd.read_csv(f"{root_path}/Cal/{folder}/csv_list/{year}.csv", dtype=str)

        # for jday, csvs in csv_list.groupby("jday"):
        #     read_csv(csvs, year, jday, root_path, fs, bucket, folder)
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
                    read_csv, (csvs, year, jday, root_path, fs, bucket, folder), callback=lambda _: pbar.update()
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

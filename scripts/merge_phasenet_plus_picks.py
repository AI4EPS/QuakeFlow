# %%
import json
import multiprocessing as mp
import os
from glob import glob

import fsspec
import numpy as np
import pandas as pd
from args import parse_args
from tqdm import tqdm


def scan_csv(year, root_path, region, model, data="picks", fs=None, bucket=None, protocol="file"):
    # %%
    csv_list = []
    if protocol != "file":
        jdays = fs.ls(f"{bucket}/{region}/{model}/{data}_{model}/{year}")
    else:
        jdays = os.listdir(f"{root_path}/{region}/{model}/{data}_{model}/{year}/")

    for jday in jdays:
        if protocol != "file":
            csvs = fs.glob(f"{jday}/??/*.csv")
        else:
            # csvs = glob(f"{root_path}/{region}/{model}/{data}_{model}/{year}/{jday}/??/*.csv")
            csvs = glob(f"{root_path}/{region}/{model}/{data}_{model}/{year}/{jday}/*.csv")

        csv_list.extend([[year, jday, csv] for csv in csvs])

    csvs = pd.DataFrame(csv_list, columns=["year", "jday", "csv"])
    csv_file = f"{root_path}/{region}/{model}/{data}_list_{year}.csv"
    csvs.to_csv(csv_file, index=False)

    return csv_file


# %%
def read_csv(rows, region, model, data, year, jday, root_path, fs=None, bucket=None):

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
        if not os.path.exists(f"{root_path}/{region}/{model}/{year}"):
            os.makedirs(f"{root_path}/{region}/{model}/{year}", exist_ok=True)
        picks.to_csv(f"{root_path}/{region}/{model}/{year}/{year}.{jday}.{data}.csv", index=False)
        # fs.put(
        #     f"{root_path}/{region}/phasenet/{year}/{jday}/{year}.{jday}.csv",
        #     f"{bucket}/{region}/phasenet_merged/{year}/{year}.{jday}.csv",
        # )
    else:
        with open(f"{root_path}/{region}/{model}/{year}/{year}.{jday}.{data}.csv", "w") as f:
            f.write("")


# %%
if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region
    # model = args.model
    model = "phasenet_plus"

    result_path = f"{region}/{model}"

    # %%
    # protocol = "gs"
    # token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    # with open(token_json, "r") as fp:
    #     token = json.load(fp)
    # fs = fsspec.filesystem(protocol, token=token)

    # %%
    # years = os.listdir(f"{root_path}/{region}/{model}/picks_{model}")
    years = glob(f"{root_path}/{region}/{model}/picks_{model}/????/")
    years = [year.rstrip("/").split("/")[-1] for year in years]
    print(f"Years: {years}")

    for year in years:

        for data in ["picks", "events"]:

            csv_list = scan_csv(year, root_path, region, model, data)

            # %%
            csv_list = pd.read_csv(csv_list, dtype=str)

            # for jday, csvs in csv_list.groupby("jday"):
            #     read_csv(csvs, region, model, data, year, jday, root_path)
            #     raise

            ncpu = min(64, mp.cpu_count())
            print(f"Number of processors: {ncpu}")
            csv_by_jday = csv_list.groupby("jday")
            pbar = tqdm(total=len(csv_by_jday), desc=f"Loading {data} csv files (year {year})")

            ctx = mp.get_context("spawn")
            with ctx.Pool(ncpu) as pool:
                jobs = []
                for jday, csvs in csv_by_jday:
                    job = pool.apply_async(
                        read_csv, (csvs, region, model, data, year, jday, root_path), callback=lambda _: pbar.update()
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
    for data in ["picks", "events"]:
        csvs = glob(f"{root_path}/{region}/{model}/????/????.???.{data}.csv")
        picks = []
        for csv in tqdm(csvs, desc=f"Merge {data} csv files"):
            picks.append(pd.read_csv(csv, dtype=str))
        picks = pd.concat(picks, ignore_index=True)
        print(f"Number of {data}: {len(picks):,}")
        if data == "picks":
            print(f"Number of P picks: {len(picks[picks['phase_type'] == 'P']):,}")
            print(f"Number of S picks: {len(picks[picks['phase_type'] == 'S']):,}")
        picks.to_csv(f"{root_path}/{region}/{model}/{model}_{data}.csv", index=False)

    # %%

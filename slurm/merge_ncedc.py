# %%
import json
import multiprocessing as mp
import os
from collections import defaultdict

import fsspec
import pandas as pd
from tqdm import tqdm


# %%
def read_csv(csv, df_list, success_list, empty_list, fail_list, protocol, token):
    fs = fsspec.filesystem(protocol, token=token)
    try:
        with fs.open(csv, token=token) as fp:
            df = pd.read_csv(fp)
        if len(df) > 0:
            df_list.append(df)
            success_list.append(csv)
        else:
            empty_list.append(csv)
    except Exception as e:
        fail_list.append(csv)
        return f"{csv}: {e}"

# %%
if __name__ == "__main__":

    # %%
    protocol = "gs"
    token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)

    # %%
    bucket = "quakeflow_catalog"
    folder = "NC/phasenet"

    fs = fsspec.filesystem(protocol, token=token)

    # %%
    result_path = "local/ncedc/results"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # %% Download stations
    station_list = fs.glob(f"quakeflow_dataset/NC/station/*.csv")
    df = []
    for station in station_list:
        df.append(pd.read_csv(fs.open(station, token=token), dtype={"network": str, "station": str, "location": str, "channel": str}))
    df = pd.concat(df, ignore_index=True)
    df["location"] = df["location"].fillna("")
    df["station_id"] = df["network"] + "." + df["station"] + "." + df["location"] + "." + df["channel"].str[:-1]
    df.to_csv(f"{result_path}/network/stations_raw.csv", index=False)
    df = df[["station_id", "latitude", "longitude", "elevation_m", "depth_km"]].groupby("station_id").first().reset_index()
    
    if not os.path.exists(f"{result_path}/network"):
        os.makedirs(f"{result_path}/network")
    df.to_csv(f"{result_path}/network/stations.csv", index=False)
    df.set_index("station_id", inplace=True)
    df.to_json(f"{result_path}/network/stations.json", orient="index", indent=2)
    
    # %% Download picks
    if not os.path.exists(f"{result_path}/phase_picking"):
        os.makedirs(f"{result_path}/phase_picking")

    year = 2023
    if not os.path.exists(f"{result_path}/phase_picking/{year}"):
        os.makedirs(f"{result_path}/phase_picking/{year}")
    if not os.path.exists(f"{result_path}/phase_picking/{year}/logs"):
        os.makedirs(f"{result_path}/phase_picking/{year}/logs")

    if os.path.exists(f"{result_path}/phase_picking/{year}/csv_list.txt"):
        with open(f"{result_path}/phase_picking/{year}/csv_list.txt", "r") as fp:
            csv_list = fp.read().splitlines()
    elif fs.exists(f"{bucket}/{folder}/csv_list/{year}.txt"):
        fs.get(f"{bucket}/{folder}/csv_list/{year}.txt", f"{result_path}/phase_picking/{year}/csv_list.txt")
    else:
        csv_list = fs.glob(f"{bucket}/{folder}/??/{year}/{year}.???/*.csv")  ## choose year and jday
        csv_list = [f"{protocol}://{csv}" for csv in csv_list]
        print(f"Save {len(csv_list)} CSV files")
        with open(f"{result_path}/phase_picking/{year}/csv_list.txt", "w") as fp:
            fp.write("\n".join(csv_list))

    # with fs.open(f"{bucket}/{folder}/csv_list/{year}.txt", "r") as fp:
    #     csv_list = fp.read().splitlines()

    # %% Example to read one pick file
    # for csv in csv_list:
    #     df = pd.read_csv(fs.open(csv, token=token))
    #     print(df)
    #     raise
            
    csv_list_by_day = defaultdict(list)
    for csv in tqdm(csv_list, desc="Group_by_day"):
        tmp = csv.split("/")
        year, jday = tmp[-3], tmp[-2].split(".")[-1]
        csv_list_by_day[f"{year}-{jday}"].append(csv)
    
    ncpu = 64*2
    with mp.get_context("spawn").Pool(processes=ncpu) as pool:
        for key in csv_list_by_day:
            year, jday = key.split("-")

            csv_list = csv_list_by_day[key]
            with mp.Manager() as manager:
                success_list = manager.list()
                fail_list = manager.list()
                empty_list = manager.list()
                df_list = manager.list()
                proc_list = []            
                with tqdm(total=len(csv_list), desc=f"{year}.{jday}") as pbar:
                    for csv in csv_list:
                        proc = pool.apply_async(read_csv, args=(csv, df_list, success_list, empty_list, fail_list, protocol, token), callback=lambda *a: pbar.update())
                        proc_list.append(proc)
                    for proc in proc_list:
                        out = proc.get()
                        if out is not None:
                            print(out)

                df_list = list(df_list)
                df = pd.concat(df_list, ignore_index=True)
                # df.to_csv(f"{result_path}/phase_picks_{year}.csv", index=False)
                df.to_csv(f"{result_path}/phase_picking/{year}/phase_picks_{jday}.csv", index=False)

                print(f"{year}.{jday} success: {len(success_list)}")
                print(f"{year}.{jday} empty: {len(empty_list)}")
                print(f"{year}.{jday} fail: {len(fail_list)}")
                success_list = list(success_list)
                fail_list = list(fail_list)
                with open(f"{result_path}/phase_picking/{year}/logs/success_{jday}.txt", "w") as fp:
                    fp.write("\n".join(success_list))
                with open(f"{result_path}/phase_picking/{year}/logs/fail_{jday}.txt", "w") as fp:
                    fp.write("\n".join(fail_list))
                with open(f"{result_path}/phase_picking/{year}/logs/empty_{jday}.txt", "w") as fp:
                    fp.write("\n".join(empty_list))


# %%

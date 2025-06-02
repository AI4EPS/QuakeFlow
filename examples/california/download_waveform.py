# %%
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timedelta
from glob import glob
from typing import Dict, List

import fsspec
import numpy as np
import obspy
import obspy.clients.fdsn
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# %%
# def download(client, stations, root_path, waveform_dir, lock=None, cloud=None):
def download(client, year, jday, stations, root_path, waveform_dir, lock=None, cloud=None):
    if isinstance(year, str):
        year = int(year)
    if isinstance(jday, str):
        jday = int(jday)

    if cloud is not None:
        protocol = cloud["protocol"]
        token = cloud["token"]
        bucket = cloud["bucket"]
    else:
        protocol = "file"
        token = None

    fs = fsspec.filesystem(protocol=protocol, token=token)
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        os.makedirs(f"{root_path}/{waveform_dir}")

    max_retry = 10

    # for key, station_group in stations.groupby(["begin_time"]):
    batch_size = 100
    begin_time = datetime.strptime(f"{year:04d}-{jday:03d}", "%Y-%j")
    end_time = begin_time + timedelta(days=1)
    begin_time = obspy.UTCDateTime(begin_time)
    end_time = obspy.UTCDateTime(end_time)
    for i in range(0, len(stations), batch_size):
        bulk = []
        for _, station in stations.iloc[i : i + batch_size].iterrows():
            bulk.append(
                (
                    station.network,
                    station.station,
                    station.location,
                    station.channel,
                    begin_time,
                    end_time,
                )
            )

        retry = 0
        while retry < max_retry:
            try:
                stream = client.get_waveforms_bulk(bulk)
                try:
                    stream.merge(fill_value="latest")
                except Exception as e:
                    print(f"Error merging traces: {waveform_dir}")
                    print([(tr.id, tr.stats.sampling_rate) for tr in stream])
                stream.sort()

                def write_trace(tr):
                    tr.data = tr.data.astype(np.float32)
                    network = tr.stats.network
                    station = tr.stats.station
                    location = tr.stats.location
                    channel = tr.stats.channel
                    fname = f"{network}.{station}.{location}.{channel}.mseed"
                    mseed_dir = f"{waveform_dir}/{network}/{year:04d}/{jday:03d}"
                    if not os.path.exists(mseed_dir):
                        os.makedirs(mseed_dir)
                    tr.write(f"{root_path}/{mseed_dir}/{fname}", format="MSEED")
                    if protocol != "file":
                        print(f"Uploading {fname} to {bucket}/{mseed_dir}/{fname}")
                        fs.put(f"{root_path}/{mseed_dir}/{fname}", f"{bucket}/{mseed_dir}/{fname}")
                        os.remove(f"{root_path}/{mseed_dir}/{fname}")

                with ThreadPoolExecutor(max_workers=16) as executor:
                    executor.map(write_trace, stream)

                break

            except Exception as err:
                err = str(err).rstrip("\n")
                message1 = "No data available for request"
                message2 = "The current client does not have a dataselect service"
                if err[: len(message1)] == message1:
                    # print(f"{message1} from {client.base_url}")
                    break
                elif err[: len(message2)] == message2:
                    print(f"{message2} from {client.base_url}")
                    break
                else:
                    print(f"Error occurred from {client.base_url}: {err}. Retrying...")
                retry += 1
                time.sleep(30)
                continue

        if retry == max_retry:
            print(f"Failed to download from {client.base_url}")

    return


def download_waveform(
    root_path: str,
    region: str,
    node_rank: int = 0,
    num_nodes: int = 1,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    cloud_config = {
        "protocol": protocol,
        "token": token,
        "bucket": bucket,
    }
    # %%
    waveform_dir = f"{region}/waveforms"
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        os.makedirs(f"{root_path}/{waveform_dir}")

    provider = region
    client = obspy.clients.fdsn.Client(provider)

    stations = pd.read_csv(f"stations_{node_rank:03d}.csv", dtype=str)
    stations = stations.fillna("")
    # stations = stations.sort_values(by=["begin_time"], ascending=False)
    # stations["begin_time"] = pd.to_datetime(stations["begin_time"])
    # stations["year"] = stations["begin_time"].dt.strftime("%Y")
    # stations["jday"] = stations["begin_time"].dt.strftime("%j")
    print(f"Total stations: {len(stations)}")

    networks = []
    for root, dirs, files in fs.walk(f"{bucket}/{waveform_dir}", maxdepth=1):
        networks.extend([f"{root}/{d}" for d in dirs])

    years = []
    for network in networks:
        for root, dirs, files in fs.walk(network, maxdepth=1):
            years.extend([f"{root}/{d}" for d in dirs])

    def process_network(network):
        network_jdays = []
        for root, dirs, files in fs.walk(network, maxdepth=1):
            network_jdays.extend([f"{root}/{d}" for d in dirs])
        return network_jdays

    with ThreadPoolExecutor(max_workers=10) as executor:
        all_jdays = list(tqdm(executor.map(process_network, years), total=len(years), desc="Processing years"))
    jdays = [jday for sublist in all_jdays for jday in sublist]

    def process_jday(jday):
        jday_mseeds = []
        for root, dirs, files in fs.walk(jday, maxdepth=1):
            jday_mseeds.extend([f"{root}/{d}" for d in files])
        return jday_mseeds

    with ThreadPoolExecutor(max_workers=20) as executor:
        all_mseeds = list(tqdm(executor.map(process_jday, jdays), total=len(jdays), desc="Processing jdays"))
    mseeds = [mseed for sublist in all_mseeds for mseed in sublist]

    processed = set(mseeds)
    print(f"Total processed: {len(processed)}")

    # # Function to process a single file rename
    # def process_rename(f_old):
    #     f = f_old.split("/")
    #     if len(f) == 6:
    #         tmp = f[4].split(".")
    #         f_new = f[:4] + [tmp[0], tmp[1]] + f[5:]
    #         f_new = "/".join(f_new)
    #         fs.copy(f_old, f_new)
    #         fs.rm(f_old)
    #         # print(f"move {f_old} to {f_new}")
    #         return f_old, f_new
    #     else:
    #         print(f"skip {f_old}")
    #         return f_old, None

    # with ThreadPoolExecutor(max_workers=40) as executor:
    #     results = list(tqdm(executor.map(process_rename, processed), total=len(processed), desc="Renaming mseeds"))

    stations["mseed"] = stations.apply(
        lambda x: f"{bucket}/{waveform_dir}/{x.network}/{x.year}/{x.jday}/{x.network}.{x.station}.{x.location}.{x.channel}.mseed",
        axis=1,
    )
    stations = stations[~stations["mseed"].isin(processed)]
    print(f"Stations to download: {len(stations)}")

    # MAX_THREADS = 3
    # BATCH_SIZE = 200

    # for i in range(0, len(stations), BATCH_SIZE):
    #     download(
    #         client,
    #         stations.iloc[i : i + BATCH_SIZE],
    #         root_path,
    #         waveform_dir,
    #         None,
    #         cloud_config,
    #     )

    # iterate stations by group by year and jday
    # stations = stations.groupby(["year", "jday"])
    # pbar = tqdm(total=len(stations), desc="Downloading waveforms")
    # for (year, jday), stations_ in stations:
    #     pbar.set_description(f"Downloading {year}/{jday}")
    #     pbar.update(1)
    #     download(
    #         client,
    #         year,
    #         jday,
    #         stations_,
    #         root_path,
    #         waveform_dir,
    #         None,
    #         cloud_config,
    #     )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for (year, jday), stations_ in stations.groupby(["year", "jday"]):
            future = executor.submit(
                download,
                client,
                year,
                jday,
                stations_,
                root_path,
                waveform_dir,
                None,
                cloud_config,
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading waveforms"):
            try:
                out = future.result()
                if out is not None:
                    print(out)
            except Exception as e:
                print(f"Error downloading waveforms: {e}")

    # with mp.Manager() as manager:
    #     lock = manager.Lock()
    #     with mp.Pool(MAX_THREADS) as pool:
    #         jobs = []
    #         for i in range(0, len(stations), BATCH_SIZE):
    #             job = pool.apply_async(
    #                 download,
    #                 (
    #                     client,
    #                     stations.iloc[i : i + BATCH_SIZE],
    #                     root_path,
    #                     waveform_dir,
    #                     lock,
    #                     cloud_config,
    #                 ),
    #             )
    #             jobs.append(job)
    #             time.sleep(1)
    #         pool.close()
    #         pool.join()
    #         for job in jobs:
    #             out = job.get()
    #             if out is not None:
    #                 print(out)


def parse_args():
    parser = argparse.ArgumentParser(description="Download waveforms")
    parser.add_argument("--num_nodes", type=int, default=4, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument("--root_path", type=str, default="./", help="Root path")
    parser.add_argument("--region", type=str, default="IRIS", help="Region to process")
    parser.add_argument("--bucket", type=str, default="quakeflow_catalog", help="Bucket")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    root_path = args.root_path
    region = args.region
    bucket = args.bucket
    node_rank = args.node_rank
    num_nodes = args.num_nodes

    protocol = "gs"
    token_json = f"application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)

    # ### Download stations
    # stations = pd.read_csv("iris_stations.csv")
    fs = fsspec.filesystem("gs", token=token)
    with fs.open("quakeflow_catalog/IRIS/stations/mendocino_bo.csv", "r") as fp:
        stations = pd.read_csv(fp)
    stations["missed_date"] = stations["missed_date"].fillna("")
    stations["location"] = stations["location"].fillna("")
    stations["missed_date"] = stations["missed_date"].apply(lambda x: x.split("|"))
    stations = stations[["network", "station", "location", "instrument", "component", "missed_date"]]
    stations = stations.explode("missed_date")
    stations = stations[stations["missed_date"].str.strip() != ""]
    stations["component"] = stations["component"].apply(lambda x: list(x))
    stations = stations.explode("component")
    stations["year"] = stations["missed_date"].str.split("-").str[0]
    stations["jday"] = stations["missed_date"].str.split("-").str[1]
    # # stations["missed_date"] = pd.to_datetime(stations["missed_date"], format="%Y-%j")
    # # stations = stations.rename(columns={"missed_date": "begin_time"})
    # # stations["end_time"] = stations["begin_time"] + pd.Timedelta(days=1)
    # # stations["begin_time"] = stations["begin_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    # # stations["end_time"] = stations["end_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    stations["channel"] = stations["instrument"] + stations["component"]
    stations.drop(columns=["instrument", "component"], inplace=True)
    # stations.sort_values(by=["begin_time"], ascending=False, inplace=True)
    # stations = stations[stations["network"] == "7D"]
    stations.reset_index(drop=True, inplace=True)
    stations = stations[~stations["network"].isin(["NC", "BK", "CI"])]

    stations.sort_values(by=["year", "jday"], ascending=False, inplace=True)
    idx = np.array_split(np.arange(len(stations)), num_nodes)[node_rank]
    stations = stations.iloc[idx]
    # stations = stations.iloc[node_rank::num_nodes]
    stations.to_csv(f"stations_{node_rank:03d}.csv", index=False)

    download_waveform(
        root_path=root_path,
        region=region,
        node_rank=node_rank,
        num_nodes=num_nodes,
        protocol=protocol,
        bucket=bucket,
        token=token,
    )
# %%

# %%
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from glob import glob
from typing import Dict, List

import fsspec
import numpy as np
import obspy
import obspy.clients.fdsn
import pandas as pd
from args import parse_args


def map_remote_path(provider, bucket, starttime, network, station, location, instrument, component):

    starttime = pd.Timestamp(starttime.datetime).round("H").to_pydatetime()  # in case like 2021-01-01T23:59:xxx
    if provider.lower() == "scedc":
        year = starttime.strftime("%Y")
        dayofyear = starttime.strftime("%j")
        if location == "":
            location = "__"
        path = f"s3://{bucket}/{year}/{year}_{dayofyear}/{network}{station:_<5}{instrument}{component}{location:_<2}_{year}{dayofyear}.ms"
    elif provider.lower() == "ncedc":
        year = starttime.strftime("%Y")
        dayofyear = starttime.strftime("%j")
        path = f"s3://{bucket}/{network}/{year}/{year}.{dayofyear}/{station}.{network}.{instrument}{component}.{location}.D.{year}.{dayofyear}"
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return path


# %%
def download(client, starttime, stations, root_path, waveform_dir, deltatime="1H", skip_list=[], lock=None, cloud=None):
    if cloud is not None:
        protocol = cloud["protocol"]
        token = cloud["token"]
        bucket = cloud["bucket"]
    else:
        protocol = "file"
        token = None
    fs = fsspec.filesystem(protocol=protocol, token=token)

    starttime = obspy.UTCDateTime(starttime)
    if deltatime == "1H":
        deltatime_s = 3600
        endtime = starttime + 3600
        mseed_dir = f"{waveform_dir}/{starttime.strftime('%Y/%j')}/{starttime.strftime('%H')}"
    elif deltatime == "1D":
        deltatime_s = 3600 * 24
        endtime = starttime + 3600 * 24
        mseed_dir = f"{waveform_dir}/{starttime.strftime('%Y/%j')}"
    else:
        raise ValueError("Invalid interval")

    if not os.path.exists(f"{root_path}/{mseed_dir}"):
        os.makedirs(f"{root_path}/{mseed_dir}")

    max_retry = 10

    bulk = []
    station_list = list(stations.values())
    np.random.shuffle(station_list)
    for station in station_list:
        # for _, station in stations.items():
        for comp in station["component"]:
            mseed_name = (
                f"{station['network']}.{station['station']}.{station['location']}.{station['instrument']}{comp}.mseed"
            )
            if os.path.exists(f"{root_path}/{mseed_dir}/{mseed_name}"):
                if protocol != "file":
                    if not fs.exists(f"{bucket}/{mseed_dir}/{mseed_name}"):
                        fs.put(f"{root_path}/{mseed_dir}/{mseed_name}", f"{bucket}/{mseed_dir}/{mseed_name}")
                print(f"{root_path}/{mseed_dir}/{mseed_name} already exists. Skip.")
                continue

            if protocol != "file":
                if fs.exists(f"{bucket}/{mseed_dir}/{mseed_name}"):
                    fs.get(f"{bucket}/{mseed_dir}/{mseed_name}", f"{root_path}/{mseed_dir}/{mseed_name}")
                    print(f"{bucket}/{mseed_dir}/{mseed_name} already exists. Skip.")
                    continue

            if cloud is not None:
                mseed_path = map_remote_path(
                    cloud["provider"],
                    cloud["bucket"],
                    starttime,
                    station["network"],
                    station["station"],
                    station["location"],
                    station["instrument"],
                    comp,
                )
                if mseed_path in skip_list:
                    continue
                try:
                    with fsspec.open(f"{mseed_path}", "rb", s3={"anon": True}) as f:
                        stream = obspy.read(f)
                        stream.merge(fill_value="latest")
                        trace = stream[0]
                        for i in range(int(np.ceil((trace.stats.endtime - starttime) / deltatime_s))):
                            tmp_time = starttime + i * deltatime_s
                            if deltatime == "1H":
                                tmp_dir = f"{waveform_dir}/{tmp_time.strftime('%Y/%j')}/{tmp_time.strftime('%H')}"
                            elif deltatime == "1D":
                                tmp_dir = f"{waveform_dir}/{tmp_time.strftime('%Y/%j')}"
                            tr = trace.slice(
                                starttime=starttime + i * deltatime_s,
                                endtime=starttime + (i + 1) * deltatime_s,
                            )
                            if tr.stats.npts < 1000:  # 10s at 100Hz
                                continue
                            os.makedirs(f"{root_path}/{tmp_dir}", exist_ok=True)
                            tr.write(f"{root_path}/{tmp_dir}/{tr.id}.mseed", format="MSEED")
                            if protocol != "file":
                                fs.put(f"{root_path}/{tmp_dir}/{tr.id}.mseed", f"{bucket}/{tmp_dir}/{tr.id}.mseed")
                    print(f"Downloaded from {cloud['provider']}:{mseed_path}")
                except Exception as e:
                    with lock:
                        skip_list.append(mseed_path)
                    print(f"Failed to download {e}")

            else:
                bulk.append(
                    (
                        station["network"],
                        station["station"],
                        station["location"],
                        f"{station['instrument']}{comp}",
                        starttime,
                        endtime,
                    )
                )

    if len(bulk) == 0:
        if cloud is not None:
            print(f"Already downloaded from {cloud['provider']}: {starttime.isoformat()} - {endtime.isoformat()}")
        else:
            print(f"Already downloaded from {client.base_url}: {starttime.isoformat()} - {endtime.isoformat()}")
        return

    print(f"Downloading from {client.base_url}: {starttime.isoformat()} - {endtime.isoformat()} {len(bulk)} traces")

    retry = 0
    while retry < max_retry:
        try:
            stream = client.get_waveforms_bulk(bulk)
            stream.merge(fill_value="latest")
            for tr in stream:
                tr.data = tr.data.astype(np.float32)
                tr.write(f"{root_path}/{mseed_dir}/{tr.id}.mseed", format="MSEED")
            if protocol != "file":
                fs.put(f"{root_path}/{mseed_dir}/{tr.id}.mseed", f"{bucket}/{mseed_dir}/{tr.id}.mseed")
            break

        except Exception as err:
            err = str(err).rstrip("\n")
            message1 = "No data available for request"
            message2 = "The current client does not have a dataselect service"
            if err[: len(message1)] == message1:
                print(f"{message1} from {client.base_url}: {starttime.isoformat()} - {endtime.isoformat()}")
                break
            elif err[: len(message2)] == message2:
                print(f"{message2} from {client.base_url}: {starttime.isoformat()} - {endtime.isoformat()}")
                break
            else:
                print(f"Error occurred from {client.base_url}:{err}. Retrying...")
            retry += 1
            time.sleep(30)
            continue

    if retry == max_retry:
        print(f"Failed to download from {client.base_url} {mseed_name}")

    return


def download_waveform(
    root_path: str,
    region: str,
    config: Dict,
    rank: int = 0,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)
    # %%
    network_dir = f"{region}/results/network"
    num_nodes = config["num_nodes"] if "num_nodes" in config else 1
    print(f"{num_nodes = }, {rank = }")
    waveform_dir = f"{region}/waveforms"
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        os.makedirs(f"{root_path}/{waveform_dir}")

    for provider in config["provider"]:
        # inventory = obspy.read_inventory(f"{root_path}/{data_dir}/inventory_{provider.lower()}.xml")
        if protocol == "file":
            with open(f"{root_path}/{network_dir}/stations.json") as f:
                stations = json.load(f)
        else:
            with fs.open(f"{bucket}/{network_dir}/stations.json") as f:
                stations = json.load(f)
        stations = {key: station for key, station in stations.items() if station["provider"] == provider}

        client = obspy.clients.fdsn.Client(provider)

        DELTATIME = "1H"  # 1H or 1D
        if DELTATIME == "1H":
            start = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%dT%H")
        elif DELTATIME == "1D":
            start = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%d")
        starttimes = pd.date_range(start, config["endtime"], freq=DELTATIME, tz="UTC", inclusive="left").to_list()
        starttimes = np.array_split(starttimes, num_nodes)[rank]
        print(f"rank {rank}: {len(starttimes) = }, {starttimes[0]}, {starttimes[-1]}")

        if provider.lower() in ["scedc", "ncedc"]:
            cloud_config = {
                "provider": provider.lower(),
                "bucket": f"{provider.lower()}-pds/continuous_waveforms",
                "protocol": protocol,
                "token": token,
            }
            starttimes = [
                starttimes[j] for i in range(24) for j in range(i, len(starttimes), 24)
            ]  ## NC, SC save data by day while we download by hour
        else:
            cloud_config = None

        MAX_THREADS = 3
        if cloud_config is not None:
            MAX_THREADS = 16

        # skip_list = []
        # lock = threading.Lock()
        # with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        #     jobs = []
        #     for starttime in starttimes:
        #         job = executor.submit(
        #             download,
        #             client,
        #             starttime,
        #             stations,
        #             root_path,
        #             waveform_dir,
        #             DELTATIME,
        #             skip_list,
        #             lock,
        #             cloud_config,
        #         )
        #         jobs.append(job)
        #         time.sleep(1)
        #     for job in jobs:
        #         print(job.result())

        with mp.Manager() as manager:
            lock = manager.Lock()
            skip_list = manager.list()
            with mp.Pool(MAX_THREADS) as pool:
                jobs = []
                for starttime in starttimes:
                    job = pool.apply_async(
                        download,
                        (
                            client,
                            starttime,
                            stations,
                            root_path,
                            waveform_dir,
                            DELTATIME,
                            skip_list,
                            lock,
                            cloud_config,
                        ),
                    )
                    jobs.append(job)
                    time.sleep(1)
                pool.close()
                pool.join()
                for job in jobs:
                    out = job.get()
                    if out is not None:
                        print(out)

    tmp_list = sorted(glob(f"{root_path}/{waveform_dir}/????/???/??/*.mseed", recursive=True))
    mseed_list = []
    for mseed in tmp_list:
        tmp = mseed.split("/")
        # year, jday = tmp[-3].split("-")
        # hour = tmp[-2]
        year, jday, hour = tmp[-4], tmp[-3], tmp[-2]
        if starttimes[0].strftime("%Y-%jT%H") <= f"{year}-{jday}T{hour}" <= starttimes[-1].strftime("%Y-%jT%H"):
            mseed_list.append(mseed)

    print(f"rank {rank}: {len(mseed_list) = }, {mseed_list[0]}, {mseed_list[-1]}")

    # %% copy to results/network
    if not os.path.exists(f"{root_path}/{region}/results/network"):
        os.makedirs(f"{root_path}/{region}/results/network")
    with open(f"{root_path}/{region}/results/network/mseed_list_{rank:03d}_{num_nodes:03d}.csv", "w") as fp:
        fp.write("\n".join(mseed_list))
    if protocol != "file":
        fs.put(
            f"{root_path}/{region}/results/network/mseed_list_{rank:03d}_{num_nodes:03d}.csv",
            f"{bucket}/{region}/results/network/mseed_list_{rank:03d}_{num_nodes:03d}.csv",
        )

    return f"{region}/results/network/mseed_list_{rank:03d}_{num_nodes:03d}.csv"


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    download_waveform(root_path=root_path, region=region, config=config)
# %%

from typing import Dict, List

from kfp import dsl


@dsl.component(base_image="zhuwq0/quakeflow:latest")
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
    import json
    import os
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    from glob import glob

    import fsspec
    import numpy as np
    import obspy
    import obspy.clients.fdsn
    import pandas as pd

    def map_remote_path(provider, bucket, starttime, network, station, location, instrument, component):
        if provider.lower() == "scedc":
            year = starttime.strftime("%Y")
            dayofyear = starttime.strftime("%j")
            if location == "":
                location = "__"
            path = f"s3://{bucket}/{year}/{year}_{dayofyear}/{network}{station:_<5}{instrument}{component}_{location}{year}{dayofyear}.ms"
        if provider.lower() == "ncedc":
            year = starttime.strftime("%Y")
            dayofyear = starttime.strftime("%j")
            path = f"s3://{bucket}/{network}/{year}/{year}.{dayofyear}/{station}.{network}.{instrument}{component}.{location}.D.{year}.{dayofyear}"
        else:
            raise ValueError(f"Unknown provider: {provider}")
        return path

    # %%
    def download(
        client,
        starttime,
        stations,
        root_path,
        waveform_dir,
        deltatime="1H",
        skip_list=[],
        lock=None,
        cloud=None,
    ):
        starttime = obspy.UTCDateTime(starttime)
        if deltatime == "1H":
            deltatime_s = 3600
            endtime = starttime + 3600
            mseed_dir = f"{waveform_dir}/{starttime.strftime('%Y-%j')}/{starttime.strftime('%H')}"
        elif deltatime == "1D":
            deltatime_s = 3600 * 24
            endtime = starttime + 3600 * 24
            mseed_dir = f"{waveform_dir}/{starttime.strftime('%Y-%j')}"
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
                mseed_name = f"{station['network']}.{station['station']}.{station['location']}.{station['instrument']}{comp}.mseed"

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
                                    tmp_dir = f"{waveform_dir}/{tmp_time.strftime('%Y-%j')}/{tmp_time.strftime('%H')}"
                                elif deltatime == "1D":
                                    tmp_dir = f"{waveform_dir}/{tmp_time.strftime('%Y-%j')}"
                                tr = trace.slice(
                                    starttime=starttime + i * deltatime_s,
                                    endtime=starttime + (i + 1) * deltatime_s,
                                )
                                if tr.stats.npts < 1000:
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
            print(f"Already downloaded from {client.base_url}: {starttime.isoformat()} - {endtime.isoformat()}")
            return 0

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

        return 0

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)
    # %%
    data_dir = f"{region}/results/data"
    num_nodes = config["kubeflow"]["num_nodes"] if "num_nodes" in config["kubeflow"] else 1
    print(f"{num_nodes = }, {rank = }")
    waveform_dir = f"{region}/waveforms"
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        os.makedirs(f"{root_path}/{waveform_dir}")

    for provider in config["provider"]:
        # inventory = obspy.read_inventory(f"{root_path}/{data_dir}/inventory_{provider.lower()}.xml")
        if protocol == "file":
            with open(f"{root_path}/{data_dir}/stations.json") as f:
                stations = json.load(f)
        else:
            with fs.open(f"{bucket}/{data_dir}/stations.json") as f:
                stations = json.load(f)
        stations = {key: station for key, station in stations.items() if station["provider"] == provider}

        client = obspy.clients.fdsn.Client(provider)

        DELTATIME = "1H"  # 1H or 1D
        if DELTATIME == "1H":
            start = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%dT%H")
        elif DELTATIME == "1D":
            start = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%d")
        starttimes = pd.date_range(start, config["endtime"], freq=DELTATIME, tz="UTC", inclusive="left")
        starttimes = np.array_split(starttimes, num_nodes)[rank]
        print(f"rank {rank}: {len(starttimes) = }, {starttimes[0]}, {starttimes[-1]}")

        if provider.lower() in ["scedc", "ncedc"]:
            cloud_config = {"provider": provider.lower(), "bucket": f"{provider.lower()}-pds/continuous_waveforms"}
        else:
            cloud_config = None

        skip_list = []
        MAX_THREADS = 3
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            for ii, starttime in enumerate(starttimes):
                executor.submit(
                    download,
                    client,
                    starttime,
                    stations,
                    root_path,
                    waveform_dir,
                    DELTATIME,
                    skip_list,
                    lock,
                    cloud_config,
                )
                time.sleep(1)

    tmp_list = sorted(glob(f"{root_path}/{waveform_dir}/????-???/??/*.mseed", recursive=True))
    mseed_list = []
    for mseed in tmp_list:
        tmp = mseed.split("/")
        year, jday = tmp[-3].split("-")
        hour = tmp[-2]
        if starttimes[0].strftime("%Y-%jT%H") <= f"{year}-{jday}T{hour}" <= starttimes[-1].strftime("%Y-%jT%H"):
            mseed_list.append(mseed)

    print(f"rank {rank}: {len(mseed_list) = }, {mseed_list[0]}, {mseed_list[-1]}")

    # %% copy to results/data
    if not os.path.exists(f"{root_path}/{region}/results/data"):
        os.makedirs(f"{root_path}/{region}/results/data")
    with open(f"{root_path}/{region}/results/data/mseed_list_{rank:03d}.csv", "w") as fp:
        fp.write("\n".join(mseed_list))
    if protocol != "file":
        fs.put(
            f"{root_path}/{region}/results/data/mseed_list_{rank:03d}.csv",
            f"{bucket}/{region}/results/data/mseed_list_{rank:03d}.csv",
        )

    return f"{region}/results/data/mseed_list_{rank:03d}.csv"


if __name__ == "__main__":
    import json
    import os
    import sys

    root_path = "local"
    region = "demo"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    download_waveform.execute(root_path=root_path, region=region, config=config)

    # # %%
    # bucket = "quakeflow_share"
    # protocol = "gs"
    # token = None
    # token_file = "/Users/weiqiang/.config/gcloud/application_default_credentials.json"
    # if os.path.exists(token_file):
    #     with open(token_file, "r") as fp:
    #         token = json.load(fp)

    # download_waveform.python_func(
    #     root_path, region=region, config=config, protocol=protocol, bucket=bucket, token=token
    # )

# %%

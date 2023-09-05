from typing import Dict

from kfp import dsl


@dsl.component(base_image="zhuwq0/quakeflow:latest")
def download_waveform(
    root_path: str,
    region: str,
    config: Dict,
    index: int = 0,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
):
    # %%
    import json
    import os
    import threading
    import time
    from datetime import datetime

    import fsspec
    import numpy as np
    import obspy
    import obspy.clients.fdsn
    import pandas as pd

    # %%
    def download(starttime, stations, root_path, waveform_dir, deltatime="1D", rank=0, lock=None):
        starttime = obspy.UTCDateTime(starttime)
        if deltatime == "1H":
            endtime = starttime + 3600
            waveform_dir = f"{waveform_dir}/{starttime.strftime('%Y-%j')}/{starttime.strftime('%H')}"
        elif deltatime == "1D":
            endtime = starttime + 3600 * 24
            waveform_dir = f"{waveform_dir}/{starttime.strftime('%Y-%j')}"
        else:
            raise ValueError("Invalid interval")

        if not os.path.exists(f"{root_path}/{waveform_dir}"):
            os.makedirs(f"{root_path}/{waveform_dir}")

        max_retry = 10
        ## FIXIT: not working correctly when endtime is None
        # inventory = inventory.select(starttime=starttime, endtime=endtime)
        # for network in inventory:
        #     for station in network:
        #         for channel in station:
        for key, station in stations.items():
            mseed_name = (
                f"{station['network']}.{station['station']}.{station['location']}.{station['instrument']}.mseed"
            )

            if os.path.exists(f"{root_path}/{waveform_dir}/{mseed_name}"):
                print(f"{root_path}/{waveform_dir}/{mseed_name} already exists. Skip.")
                if protocol != "file":
                    if not fs.exists(f"{bucket}/{waveform_dir}/{mseed_name}"):
                        fs.put(f"{root_path}/{waveform_dir}/{mseed_name}", f"{bucket}/{waveform_dir}/{mseed_name}")
                continue

            if protocol != "file":
                if fs.exists(f"{bucket}/{waveform_dir}/{mseed_name}"):
                    print(f"{bucket}/{waveform_dir}/{mseed_name} already exists. Skip.")
                    fs.get(f"{bucket}/{waveform_dir}/{mseed_name}", f"{root_path}/{waveform_dir}/{mseed_name}")
                    continue

            print(f"Downloading {starttime.strftime('%Y-%m-%dT%H')}-{endtime.strftime('%Y-%m-%dT%H')} {mseed_name}")
            # mseed_txt = waveform_path / f"{network.code}.{station.code}.{channel.location_code}.{channel.code}.txt"
            # if mseed_txt.exists():
            #     print(f"File {mseed_txt} no data available. Skip.")
            #     continue

            retry = 0
            while retry < max_retry:
                try:
                    stream = client.get_waveforms(
                        network=station["network"],
                        station=station["station"],
                        location=station["location"],
                        channel=",".join(set([f"{station['instrument']}{comp}" for comp in station["component"]])),
                        starttime=starttime,
                        endtime=endtime,
                    )
                    stream.merge(fill_value="latest")
                    stream.write(f"{root_path}/{waveform_dir}/{mseed_name}", format="MSEED")
                    if protocol != "file":
                        fs.put(f"{root_path}/{waveform_dir}/{mseed_name}", f"{bucket}/{waveform_dir}/{mseed_name}")
                    break

                except Exception as err:
                    err = str(err).rstrip("\n")
                    message = "No data available for request"
                    if err[: len(message)] == message:
                        print(f"{message}: {key}")
                        # os.system(f"touch {mseed_txt}")
                        break
                    else:
                        print(f"Error occurred:{err}. Retrying...")
                    retry += 1
                    time.sleep(30)
                    continue

            if retry == max_retry:
                print(f"Failed to download {mseed_name}")

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)
    # %%
    data_dir = f"{region}/obspy"
    if "num_nodes" in config:
        num_nodes = config["num_nodes"]
    else:
        num_nodes = 1
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

        DELTATIME = "1H"  # "1D"
        if DELTATIME == "1H":
            start = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%dT%H")
        elif DELTATIME == "1D":
            start = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%d")
        starttimes = pd.date_range(start, config["endtime"], freq=DELTATIME, tz="UTC", inclusive="left")
        # starttimes = starttimes[index::num_nodes]
        starttimes = np.array_split(starttimes, num_nodes)[index]

        threads = []
        MAX_THREADS = 3
        lock = threading.Lock()
        for ii, starttime in enumerate(starttimes):
            t = threading.Thread(
                target=download, args=(starttime, stations, root_path, waveform_dir, DELTATIME, lock, len(threads))
            )
            t.start()
            threads.append(t)
            time.sleep(1)
            if (len(threads) - 1) % MAX_THREADS == MAX_THREADS - 1:
                for t in threads:
                    t.join()
                threads = []
        for t in threads:
            t.join()


if __name__ == "__main__":
    import json
    import os

    root_path = "local"
    region = "demo"
    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    download_waveform.python_func(root_path, region=region, config=config)

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

from typing import Dict

from kfp import dsl


@dsl.component(base_image="zhuwq0/quakeflow:latest")
def download_waveform_event(
    root_path: str,
    region: str,
    config: Dict,
    index: int = 0,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
):
    import json
    import os
    import random
    import threading
    import time
    from collections import namedtuple

    import fsspec
    import numpy as np
    import obspy
    import obspy.clients.fdsn
    import obspy.geodetics.base
    import obspy.taup
    import pandas as pd

    # %%
    def calc_arrival_time(event, stations):
        taup_model = obspy.taup.TauPyModel(model="iasp91")
        Location = namedtuple("location", ["longitude", "latitude", "depth_km"])
        min_dist = None
        for _, station in stations.items():
            dist = np.sqrt(
                ((event["longitude"] - station["longitude"]) * np.cos(np.deg2rad(event["latitude"]))) ** 2
                + (event["latitude"] - station["latitude"]) ** 2
            )
            if min_dist is None or min_dist > dist:
                min_dist = dist
                closest = Location(
                    longitude=station["longitude"], latitude=station["latitude"], depth_km=station["depth_km"]
                )
        min_dist_km = (
            obspy.geodetics.base.gps2dist_azimuth(
                event.latitude, event["longitude"], closest.latitude, closest.longitude
            )[0]
            / 1000
        )
        min_dist_deg = obspy.geodetics.base.kilometer2degrees(min_dist_km)
        min_tt = taup_model.get_travel_times(
            distance_in_degree=min_dist_deg,
            source_depth_in_km=max(0, event.depth_km),
            receiver_depth_in_km=max(0, closest.depth_km),
        )[0].time

        return event.time + pd.to_timedelta(min_tt, unit="s")

    # %%
    def download_event(
        event, stations, client, root_path, waveform_dir, time_before=30, time_after=90, lock=None, rank=0
    ):
        if not os.path.exists(f"{root_path}/{waveform_dir}"):
            os.makedirs(f"{root_path}/{waveform_dir}")

        max_retry = 10

        arrival_time = calc_arrival_time(event, stations)
        starttime = arrival_time - pd.to_timedelta(time_before, unit="s")
        endtime = arrival_time + pd.to_timedelta(time_after, unit="s")

        bulk = []
        for _, station in stations.items():
            location = "*" if len(station["location"]) == 0 else station["location"]
            channel = ",".join(set([f"{station['instrument']}{comp}" for comp in station["component"]]))
            bulk.append(
                (
                    station["network"],
                    station["station"],
                    location,
                    channel,
                    starttime.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    endtime.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                )
            )

        print(f"Downloading {event.event_id} ...")
        mseed_name = f"{event.event_id}.mseed"

        if os.path.exists(f"{root_path}/{waveform_dir}/{mseed_name}"):
            print(f"{root_path}/{waveform_dir}/{mseed_name} already exists. Skip.")
            if protocol != "file":
                if not fs.exists(f"{bucket}/{waveform_dir}/{mseed_name}"):
                    fs.put(f"{root_path}/{waveform_dir}/{mseed_name}", f"{bucket}/{waveform_dir}/{mseed_name}")
            return

        if protocol != "file":
            if fs.exists(f"{bucket}/{waveform_dir}/{mseed_name}"):
                print(f"{bucket}/{waveform_dir}/{mseed_name} already exists. Skip.")
                fs.get(f"{bucket}/{waveform_dir}/{mseed_name}", f"{root_path}/{waveform_dir}/{mseed_name}")
                return

        retry = 0
        while retry < max_retry:
            try:
                stream = client.get_waveforms_bulk(bulk)

                stream.merge(fill_value="latest")
                stream.write(f"{root_path}/{waveform_dir}/{mseed_name}", format="MSEED")
                if protocol != "file":
                    fs.put(f"{root_path}/{waveform_dir}/{mseed_name}", f"{bucket}/{waveform_dir}/{mseed_name}")
                break

            except Exception as err:
                err = str(err).rstrip("\n")
                message = "No data available for request"
                if err[: len(message)] == message:
                    print(f"{message}: {mseed_name}")
                    break
                else:
                    print(f"Error occurred:\n{err}\nRetrying...")
                retry += 1
                time.sleep(20 ** ((random.random() + 1)))
                continue

        if retry == max_retry:
            print(f"Failed to download {mseed_name} after {max_retry} retries.")
            os.system(f"touch {str(mseed_name).replace('.mseed', '.failed')}")

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)
    # %%
    data_dir = f"{region}/obspy"
    # with open(f"{root_path}/{region}/config.json") as f:
    #     config = json.load(f)
    if "num_nodes" in config:
        num_nodes = config["num_nodes"]
    else:
        num_nodes = 1
    waveform_dir = f"{region}/waveforms"
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        os.makedirs(f"{root_path}/{waveform_dir}")

    if protocol == "file":
        events = pd.read_csv(f"{root_path}/{data_dir}/catalog.csv", parse_dates=["time"])
    else:
        events = pd.read_csv(f"{protocol}://{bucket}/{data_dir}/catalog.csv", parse_dates=["time"])
    events = events.iloc[index::num_nodes, :]
    for provider in config["provider"]:
        if protocol == "file":
            with open(f"{root_path}/{data_dir}/stations.json") as f:
                stations = json.load(f)
        else:
            with fs.open(f"{bucket}/{data_dir}/stations.json") as f:
                stations = json.load(f)
        stations = {key: station for key, station in stations.items() if station["provider"] == provider}

        client = obspy.clients.fdsn.Client(provider, timeout=1200)

        threads = []
        MAX_THREADS = 1
        TIME_BEFORE = 30
        TIME_AFTER = 90
        lock = threading.Lock()
        events = events.sort_values(by="time", ignore_index=True, ascending=False)
        for _, event in events.iterrows():
            t = threading.Thread(
                target=download_event,
                args=(event, stations, client, root_path, waveform_dir, TIME_BEFORE, TIME_AFTER, lock, len(threads)),
            )
            t.start()
            threads.append(t)
            time.sleep(1)
            if (len(threads) - 1) % MAX_THREADS == (MAX_THREADS - 1):
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

    download_waveform_event.python_func(root_path, region=region, config=config)

    # # %%
    # bucket = "quakeflow_share"
    # protocol = "gs"
    # token = None
    # token_file = "/Users/weiqiang/.config/gcloud/application_default_credentials.json"
    # if os.path.exists(token_file):
    #     with open(token_file, "r") as fp:
    #         token = json.load(fp)

    # download_waveform_event.python_func(
    #     root_path, region=region, config=config, protocol=protocol, bucket=bucket, token=token
    # )

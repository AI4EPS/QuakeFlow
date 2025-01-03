import json
import os
import sys
from datetime import datetime
from glob import glob
from typing import Dict

import fsspec
import numpy as np
import obspy
import obspy.clients.fdsn
import pandas as pd
from args import parse_args
from obspy.clients.fdsn.mass_downloader import (
    CircularDomain,
    GlobalDomain,
    MassDownloader,
    RectangularDomain,
    Restrictions,
)


def download_waveform(
    root_path: str,
    region: str,
    config: Dict,
    node_rank: int = 0,
    num_nodes: int = 1,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
):

    # %%
    fs = fsspec.filesystem(protocol, token=token)

    # %%
    waveform_dir = f"{region}/waveforms"
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        os.makedirs(f"{root_path}/{waveform_dir}")

    DELTATIME = "1D"  # "1D"
    if DELTATIME == "1H":
        start = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%dT%H")
        DELTATIME_SEC = 3600
    elif DELTATIME == "1D":
        start = datetime.fromisoformat(config["starttime"]).strftime("%Y-%m-%d")
        DELTATIME_SEC = 3600 * 24
    starttimes = pd.date_range(start, config["endtime"], freq=DELTATIME, tz="UTC", inclusive="left")
    # starttimes = starttimes[rank::num_nodes]
    starttimes = np.array_split(starttimes, num_nodes)[node_rank]
    if len(starttimes) == 0:
        return

    # %%
    domain = GlobalDomain()
    if ("longitude0" in config) and ("latitude0" in config) and ("maxradius_degree" in config):
        domain = CircularDomain(
            longitude=config["longitude0"],
            latitude=config["latitude0"],
            minradius=0,
            maxradius=config["maxradius_degree"],
        )
    if (
        ("minlatitude" in config)
        and ("maxlatitude" in config)
        and ("minlongitude" in config)
        and ("maxlongitude" in config)
    ):
        domain = RectangularDomain(
            minlatitude=config["minlatitude"],
            maxlatitude=config["maxlatitude"],
            minlongitude=config["minlongitude"],
            maxlongitude=config["maxlongitude"],
        )
    print(f"{domain = }")

    restrictions = Restrictions(
        # starttime=obspy.UTCDateTime(config["starttime"]),
        # endtime=obspy.UTCDateTime(config["endtime"]),
        # chunklength_in_sec=3600,
        starttime=obspy.UTCDateTime(starttimes[0]),
        endtime=obspy.UTCDateTime(starttimes[-1]) + DELTATIME_SEC,
        chunklength_in_sec=DELTATIME_SEC,
        network=config["network"] if "network" in config else None,
        station=config["station"] if "station" in config else None,
        minimum_interstation_distance_in_m=0,
        minimum_length=0.1,
        reject_channels_with_gaps=False,
        # channel_priorities=config["channel_priorities"],
    )
    print(f"{restrictions = }")

    def get_mseed_storage(network, station, location, channel, starttime, endtime):
        if DELTATIME == "1H":
            mseed_name = f"{starttime.strftime('%Y-%j')}/{starttime.strftime('%H')}/{network}.{station}.{location}.{channel}.mseed"
        elif DELTATIME == "1D":
            mseed_name = f"{starttime.strftime('%Y/%j')}/{network}.{station}.{location}.{channel}.mseed"
        mseed_name = f"{starttime.strftime('%Y/%j')}/{network}.{station}.{location}.{channel}.mseed"
        if os.path.exists(f"{root_path}/{waveform_dir}/{mseed_name}"):
            print(f"{root_path}/{waveform_dir}/{mseed_name} already exists. Skip.")
            return True
        if protocol != "file":
            if fs.exists(f"{bucket}/{waveform_dir}/{mseed_name}"):
                print(f"{bucket}/{waveform_dir}/{mseed_name} already exists. Skip.")
                fs.get(f"{bucket}/{waveform_dir}/{mseed_name}", f"{root_path}/{waveform_dir}/{mseed_name}")
                return True
        return f"{root_path}/{waveform_dir}/{mseed_name}"

    print(f"Downloading waveforms...")
    mdl = MassDownloader(
        providers=config["provider"],
    )
    mdl.download(
        domain,
        restrictions,
        mseed_storage=get_mseed_storage,
        stationxml_storage=f"{root_path}/{waveform_dir}/stations",
        download_chunk_size_in_mb=20,
        threads_per_client=3,  # default 3
    )

    if protocol != "file":
        fs.put(f"{root_path}/{waveform_dir}/", f"{bucket}/{waveform_dir}/", recursive=True)
        # fs.put(f"{root_path}/{waveform_dir}/stations/", f"{bucket}/{waveform_dir}/stations/", recursive=True)

    return


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region
    protocol = args.protocol
    bucket = args.bucket
    token = args.token
    node_rank = args.node_rank
    num_nodes = args.num_nodes

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    download_waveform(root_path, region=region, config=config, node_rank=node_rank, num_nodes=num_nodes)

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

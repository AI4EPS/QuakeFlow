# %%
import argparse
import json
import os
import fsspec
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import threading
from functools import partial


np.random.seed(42)


# %%
def calc_snr(data, index0, noise_window=300, signal_window=300, gap_window=50):
    """Calculate signal-to-noise ratio for each channel.

    Args:
        data: Array of shape (3, nt) with waveform data
        index0: Sample index of the first arrival
        noise_window: Number of samples for noise window
        signal_window: Number of samples for signal window
        gap_window: Gap between noise and signal windows

    Returns:
        List of SNR values for each channel
    """
    snr = []
    for i in range(data.shape[0]):
        j = index0
        noise_start = max(0, j - noise_window)
        noise_end = max(0, j - gap_window)
        signal_start = min(data.shape[1], j + gap_window)
        signal_end = min(data.shape[1], j + signal_window)

        if noise_end <= noise_start or signal_end <= signal_start:
            snr.append(0)
            continue

        noise = np.std(data[i, noise_start:noise_end])
        signal = np.std(data[i, signal_start:signal_end])

        if noise > 0 and signal > 0:
            snr.append(signal / noise)
        else:
            snr.append(0)

    return snr


def flip_polarity(phase_polarity, channel_dip):
    """Flip polarity based on channel dip angle.

    Args:
        phase_polarity: List of polarity values ('U', 'D', '+', '-', etc.)
        channel_dip: List of dip angles for each pick

    Returns:
        List of corrected polarity values
    """
    pol_out = []
    for pol, dip in zip(phase_polarity, channel_dip):
        if pol == "U" or pol == "+":
            if dip == -90:
                pol_out.append("U")
            elif dip == 90:
                pol_out.append("D")
            else:
                pol_out.append("N")
        elif pol == "D" or pol == "-":
            if dip == -90:
                pol_out.append("D")
            elif dip == 90:
                pol_out.append("U")
            else:
                pol_out.append("N")
        else:
            pol_out.append("N")
    return pol_out


# %%
def extract_template_numpy(
    picks_group,
    events,
    picks,
    config,
    fp,
    lock,
):

    region = config["region"]

    for picks in picks_group:

        waveforms_dict = {}

        # for _, pick in picks.iterrows():
        for key, picks_ in picks.groupby(["event_id", "network", "station", "location", "instrument"]):
            event_id, network, station, location, instrument = key
            station_id = f"{network}.{station}.{location}.{instrument}"

            begin_time = picks_.iloc[0]["begin_time"]
            end_time = picks_.iloc[0]["end_time"]

            p_pick = picks_[picks_["phase_type"] == "P"]
            s_pick = picks_[picks_["phase_type"] == "S"]

            pick = p_pick.iloc[0] if len(p_pick) > 0 else s_pick.iloc[0]

            if len(picks_) > 2:
                print(f"More than two picks: {picks_}")
                continue

            event = events.loc[event_id]

            ENZ = pick["ENZ"].split(",")

            if pick["ENZ"] not in waveforms_dict:
                stream_3c = obspy.Stream()
                for c in ENZ:
                    try:
                        with fsspec.open(c, "rb", anon=True) as f:
                            stream = obspy.read(f)
                            stream.merge(fill_value="latest")
                            if len(stream) > 1:
                                print(f"More than one trace: {stream}")
                            trace = stream[0]
                            if trace.stats.sampling_rate != config["sampling_rate"]:
                                if trace.stats.sampling_rate % config["sampling_rate"] == 0:
                                    trace.decimate(int(trace.stats.sampling_rate / config["sampling_rate"]))
                                else:
                                    trace.resample(config["sampling_rate"])
                            stream_3c.append(trace)
                    except Exception as err:
                        print(f"Error reading: {err}")
                        continue

                try:
                    fs = fsspec.filesystem(
                        "gs", token=os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
                    )
                    if region == "SC":
                        with fs.open(
                            f"quakeflow_catalog/SC/FDSNstationXML/{network}/{network}_{station}.xml", "r"
                        ) as f:
                            inv = obspy.read_inventory(f)
                    elif region == "NC":
                        with fs.open(
                            f"quakeflow_catalog/NC/FDSNstationXML/{network}/{network}.{station}.xml", "r"
                        ) as f:
                            inv = obspy.read_inventory(f)
                    else:
                        raise ValueError(f"Invalid region: {region}")

                except Exception as err:
                    print(f"Error reading: {err}")
                    continue

                stream_3c.remove_sensitivity(inv)
                stream_3c.detrend("constant")
                stream_3c.rotate("->ZNE", inventory=inv)

            else:
                stream_3c = waveforms_dict[pick["ENZ"]]

            tmp = stream_3c.slice(
                obspy.UTCDateTime(begin_time), obspy.UTCDateTime(end_time), keep_empty_traces=False, nearest_sample=True
            )

            template = np.zeros((3, config["nt"]), dtype=np.float32)
            component = []
            for i, ch in enumerate(["E", "N", "Z"]):
                trace = tmp.select(component=f"{ch}")
                if len(trace) == 0:
                    continue
                elif len(trace) > 1:
                    print(f"More than one trace: {trace}")
                    continue
                else:
                    component.append(ch)
                    template[i, : len(trace[0].data)] = trace[0].data[: config["nt"]] * 1e6  # to micro m/s

            # Calculate SNR based on first arrival
            first_arrival_index = pick["phase_index"]
            snr = calc_snr(template, first_arrival_index)

            # Skip if all channels have zero SNR
            if max(snr) == 0:
                continue

            with lock:
                if f"{event_id}" not in fp:
                    gp = fp.create_group(f"{event_id}")
                    for key, value in event.items():
                        if key in ["event_time", "begin_time", "end_time"]:
                            value = value.strftime("%Y-%m-%dT%H:%M:%S.%f")
                        gp.attrs[key] = value

                    gp.attrs["begin_time"] = begin_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
                    gp.attrs["end_time"] = end_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

                ds = fp.create_dataset(f"{event_id}/{station_id}", data=template)
                for key, value in pick.items():
                    if key in ["event_id", "ENZ", "begin_time", "end_time"]:
                        continue
                    if key in ["phase_time"]:
                        value = value.strftime("%Y-%m-%dT%H:%M:%S.%f")
                    ds.attrs[key] = value

                if len(p_pick) > 0:
                    ds.attrs["p_phase_time"] = p_pick["phase_time"].iloc[0].strftime("%Y-%m-%dT%H:%M:%S.%f")
                    ds.attrs["p_phase_index"] = p_pick["phase_index"].values[0]
                    ds.attrs["p_phase_score"] = p_pick["phase_score"].values[0]
                    ds.attrs["p_phase_polarity"] = p_pick["phase_polarity"].values[0]
                    ds.attrs["p_phase_remark"] = p_pick["phase_remark"].values[0]
                    ds.attrs["p_phase_status"] = p_pick["review_status"].values[0]
                    ds.attrs["p_phase_weight"] = p_pick["phase_weight"].values[0]

                if len(s_pick) > 0:
                    ds.attrs["s_phase_time"] = s_pick["phase_time"].iloc[0].strftime("%Y-%m-%dT%H:%M:%S.%f")
                    ds.attrs["s_phase_index"] = s_pick["phase_index"].values[0]
                    ds.attrs["s_phase_score"] = s_pick["phase_score"].values[0]
                    ds.attrs["s_phase_polarity"] = s_pick["phase_polarity"].values[0]
                    ds.attrs["s_phase_remark"] = s_pick["phase_remark"].values[0]
                    ds.attrs["s_phase_status"] = s_pick["review_status"].values[0]
                    ds.attrs["s_phase_weight"] = s_pick["phase_weight"].values[0]

                ds.attrs["unit"] = "micro m/s"
                ds.attrs["component"] = component
                ds.attrs["snr"] = snr

                picks_in_window = picks[(picks["phase_time"] >= begin_time) & (picks["phase_time"] <= end_time)].copy()
                picks_in_window.sort_values(by="phase_time", inplace=True)
                ds.attrs["event_id"] = picks_in_window["event_id"].values
                ds.attrs["phase_type"] = picks_in_window["phase_type"].values
                ds.attrs["phase_time"] = (
                    picks_in_window["phase_time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f")).values
                )
                ds.attrs["phase_index"] = picks_in_window["phase_index"].values
                ds.attrs["phase_score"] = picks_in_window["phase_score"].values
                ds.attrs["phase_polarity"] = picks_in_window["phase_polarity"].values
                ds.attrs["phase_remark"] = picks_in_window["phase_remark"].values
                ds.attrs["phase_weight"] = picks_in_window["phase_weight"].values


# %%
def cut_templates(jdays, root_path, data_path, result_path, region, config, bucket, protocol, token):

    # %%
    fs = fsspec.filesystem(protocol, token=token)

    sampling_rate = 100.0
    time_before = 40.96
    time_after = 40.96 * 2

    time_window = time_before + time_after
    nt = int(round(time_window * sampling_rate))
    assert nt == 4096 * 3

    config["nt"] = nt
    config["sampling_rate"] = sampling_rate
    config["time_before"] = time_before
    config["time_after"] = time_after

    for jday in jdays:
        year, jday = jday.split(".")
        year, jday = int(year), int(jday)

        os.makedirs(f"{root_path}/{result_path}/{year:04d}/{jday:03d}", exist_ok=True)

        if protocol == "file":
            events = pd.read_csv(f"{root_path}/{data_path}/{year:04d}/{jday:03d}/events.csv", dtype=str)
        else:
            with fs.open(f"{bucket}/{data_path}/{year:04d}/{jday:03d}/events.csv", "r") as fp:
                events = pd.read_csv(fp, dtype=str)

        events.fillna("", inplace=True)
        events.rename(columns={"time": "event_time"}, inplace=True)
        events["event_time"] = pd.to_datetime(events["event_time"])

        print(f"{len(events) = }")
        print(events.head())

        # %%
        if protocol == "file":
            picks = pd.read_csv(f"{root_path}/{data_path}/{year:04d}/{jday:03d}/phases.csv", dtype=str)
        else:
            with fs.open(f"{bucket}/{data_path}/{year:04d}/{jday:03d}/phases.csv", "r") as fp:
                picks = pd.read_csv(fp, dtype=str)

        picks.fillna("", inplace=True)
        picks.rename(columns={"component": "pick_component"}, inplace=True)
        picks["phase_time"] = pd.to_datetime(picks["phase_time"])
        picks["begin_time"] = picks.groupby("event_id")["phase_time"].transform("min") - pd.Timedelta(
            seconds=config["time_before"]
        )
        picks["end_time"] = picks.groupby("event_id")["phase_time"].transform("min") + pd.Timedelta(
            seconds=config["time_after"]
        )
        picks["phase_index"] = (
            (picks["phase_time"] - picks["begin_time"]).dt.total_seconds() * config["sampling_rate"]
        ).astype(int)

        print(f"{len(picks) = }")
        print(picks.head())

        tmp = picks.groupby("event_id")["begin_time"].first()
        events = events.merge(tmp, on="event_id", how="left")
        events["event_time_index"] = (
            (events["event_time"] - events["begin_time"]).dt.total_seconds() * config["sampling_rate"]
        ).astype(int)
        events.set_index("event_id", inplace=True)

        ############################# CLOUD #########################################
        # Use separate filesystem for quakeflow_catalog bucket (contains mseed_list and station XML)
        catalog_bucket = "quakeflow_catalog"
        catalog_token_json = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        with open(catalog_token_json, "r") as fp:
            catalog_token = json.load(fp)
        catalog_fs = fsspec.filesystem(protocol="gs", token=catalog_token)
        mseeds_df = []
        for folder in [region]:
            with catalog_fs.open(f"{catalog_bucket}/{folder}/mseed_list/{year}_3c.txt", "r") as f:
                mseeds = f.readlines()
            mseeds = [x.strip("\n") for x in mseeds]
            mseeds = pd.DataFrame(mseeds, columns=["ENZ"])
            if folder == "SC":
                mseeds["fname"] = mseeds["ENZ"].apply(lambda x: x.split("/")[-1])
                mseeds["network"] = mseeds["fname"].apply(lambda x: x[:2])
                mseeds["station"] = mseeds["fname"].apply(lambda x: x[2:7].strip("_"))
                mseeds["instrument"] = mseeds["fname"].apply(lambda x: x[7:9])
                mseeds["location"] = mseeds["fname"].apply(lambda x: x[10:12].strip("_"))
                mseeds["year"] = mseeds["fname"].apply(lambda x: x[13:17])
                mseeds["jday"] = mseeds["fname"].apply(lambda x: x[17:20])
            if folder == "NC":
                mseeds["fname"] = mseeds["ENZ"].apply(lambda x: x.split("/")[-1])
                mseeds["network"] = mseeds["fname"].apply(lambda x: x.split(".")[1])
                mseeds["station"] = mseeds["fname"].apply(lambda x: x.split(".")[0])
                mseeds["instrument"] = mseeds["fname"].apply(lambda x: x.split(".")[2][:-1])
                mseeds["location"] = mseeds["fname"].apply(lambda x: x.split(".")[3])
                mseeds["year"] = mseeds["fname"].apply(lambda x: x.split(".")[5])
                mseeds["jday"] = mseeds["fname"].apply(lambda x: x.split(".")[6])
            mseeds_df.append(mseeds)
        mseeds_df = pd.concat(mseeds_df)

        print(f"{len(mseeds_df) = }")
        print(mseeds_df.head())

        # picks["network"] = picks["station_id"].apply(lambda x: x.split(".")[0])
        # picks["station"] = picks["station_id"].apply(lambda x: x.split(".")[1])
        # picks["location"] = picks["station_id"].apply(lambda x: x.split(".")[2])
        # picks["instrument"] = picks["station_id"].apply(lambda x: x.split(".")[3])
        picks["year"] = picks["phase_time"].dt.strftime("%Y")
        picks["jday"] = picks["phase_time"].dt.strftime("%j")

        mseeds_df = mseeds_df[(mseeds_df["year"].astype(int) == year) & (mseeds_df["jday"].astype(int) == jday)]

        picks = picks.merge(mseeds_df, on=["network", "station", "location", "instrument", "year", "jday"])
        picks.drop(columns=["fname", "year", "jday"], inplace=True)

        if len(picks) == 0:
            print(f"No picks found for {year:04d}/{jday:03d}")
            continue

        picks_group = picks.copy()
        picks_group = picks_group.groupby("ENZ")

        ############################################################
        os.makedirs(f"{result_path}/{year:04d}/{jday:03d}", exist_ok=True)
        fp = h5py.File(f"{result_path}/{year:04d}/{jday:03d}/waveform.h5", "a")

        # nsplit = 1
        # group_chunk = np.array_split(list(picks_group.groups.keys()), nsplit)
        # picks_group_chunk = [[picks_group.get_group(g) for g in group] for group in group_chunk]

        # lock = nullcontext()
        # for group in picks_group_chunk:
        #     extract_template_numpy(group, events, picks, config, fp, lock)
        # raise

        ncpu = min(32, multiprocessing.cpu_count())
        nsplit = len(picks_group)
        print(f"Using {ncpu} cores")

        group_chunk = np.array_split(list(picks_group.groups.keys()), nsplit)
        picks_group_chunk = [[picks_group.get_group(g) for g in group] for group in group_chunk]

        lock = threading.Lock()  # Use threading.Lock for ThreadPoolExecutor

        # with ProcessPoolExecutor(max_workers=ncpu) as executor:
        with ThreadPoolExecutor(max_workers=ncpu) as executor:
            futures = [
                executor.submit(
                    partial(extract_template_numpy, events=events, picks=picks, config=config, fp=fp, lock=lock), group
                )
                for group in picks_group_chunk
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting templates"):
                out = future.result()
                if out is not None:
                    print(out)

        # if protocol == "gs":
        #     fs.put(
        #         f"{root_path}/{result_path}/{year:04d}/{jday:03d}/cctorch_picks.csv",
        #         f"{bucket}/{result_path}/{year:04d}/{jday:03d}/cctorch_picks.csv",
        #     )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="NC")
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument("--bucket", type=str, default="quakeflow_dataset")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--year", type=int, default=2024)

    return parser.parse_args()


# %%
if __name__ == "__main__":

    # %%
    protocol = "gs"
    token_json = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    with open(token_json, "r") as fp:
        token = json.load(fp)

    # protocol = "file"
    # token = None

    # %%
    args = parse_args()
    region = args.region
    root_path = args.root_path
    bucket = args.bucket
    num_nodes = args.num_nodes
    node_rank = args.node_rank
    year = args.year

    num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
    jdays = range(1, num_jday + 1)

    jdays = np.array_split(jdays, num_nodes)[node_rank]

    processed = []

    jdays = [jday for jday in jdays if f"{jday:03d}" not in processed]
    print(f"Remaining days: {len(jdays)}")

    if len(jdays) == 0:
        print("No days to process")
        exit()

    jdays = [f"{year}.{jday:03d}" for jday in jdays]

    print(f"{node_rank}/{num_nodes}: {jdays[0] = }, {jdays[-1] = }")

    config = vars(args)

    # %%
    if protocol == "file":
        data_path = f"{region}EDC/dataset"
        result_path = f"{region}EDC/dataset"
    else:
        data_path = f"{region}/catalog"
        result_path = f"{region}/dataset"

    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    cut_templates(jdays, root_path, data_path, result_path, region, config, bucket, protocol, token)

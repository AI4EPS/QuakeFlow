# %%
import multiprocessing as mp
import os
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path

import fsspec
import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm

# warnings.filterwarnings("error")
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# %%
protocol = "gs"
token = "/home/zhuwq/.config/gcloud/application_default_credentials.json"
bucket = "quakeflow_dataset"

# root_path = "dataset"
root_path = f"{bucket}/NC"
mseed_path = f"{root_path}/waveform_mseed"
catalog_path = f"{root_path}/catalog"
station_path = f"{root_path}/station"
result_path = f"waveform_h5"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# %%
fs = fsspec.filesystem(protocol=protocol, token=token)

# %%
comp = ["3", "2", "1", "U", "V", "E", "N", "Z"]
order = {key: i for i, key in enumerate(comp)}
comp2idx = {
    "3": 0,
    "2": 1,
    "1": 2,
    "U": 0,
    "V": 1,
    "E": 0,
    "N": 1,
    "Z": 2,
}  ## only for cases less than 3 components

sampling_rate = 100

# %%
stations = []
for csv in fs.glob(f"{station_path}/*.csv"):
    with fs.open(csv, "rb") as f:
        stations.append(
            pd.read_csv(
                f,
                parse_dates=["begin_time", "end_time"],
                date_format="%Y-%m-%dT%H:%M:%S",
                dtype={"network": str, "station": str, "location": str, "instrument": str},
            )
        )
stations = pd.concat(stations)
stations["location"] = stations["location"].fillna("")
stations.set_index(["network", "station", "location", "instrument"], inplace=True)
stations = stations.sort_index()


# %%
def calc_snr(data, index0, noise_window=300, signal_window=300, gap_window=50):
    snr = []
    for i in range(data.shape[0]):
        j = index0
        if (len(data[i, j - noise_window : j - gap_window]) == 0) or (
            len(data[i, j + gap_window : j + signal_window]) == 0
        ):
            snr.append(0)
            continue
        noise = np.std(data[i, j - noise_window : j - gap_window])
        signal = np.std(data[i, j + gap_window : j + signal_window])

        if (noise > 0) and (signal > 0):
            snr.append(signal / noise)
        else:
            snr.append(0)

    return snr


# %%
def extract_pick(picks, begin_time, sampling_rate):
    phase_type = []
    phase_index = []
    phase_score = []
    phase_time = []
    phase_polarity = []
    phase_remark = []
    event_id = []
    for idx, pick in picks.sort_values("phase_time").iterrows():
        phase_type.append(pick.phase_type)
        phase_index.append(int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate)))
        phase_score.append(pick.phase_score)
        phase_time.append(pick.phase_time.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        phase_remark.append(pick.remark)
        phase_polarity.append(pick.phase_polarity)
        event_id.append(pick.event_id)

    return phase_type, phase_index, phase_score, phase_time, phase_remark, phase_polarity, event_id


# %%
def convert(i, year):
    # %%
    with h5py.File(f"{result_path}/{year}.h5", "w") as fp:
        jdays = sorted(fs.ls(f"{mseed_path}/{year}"))[::-1]
        jdays = [x.split("/")[-1] for x in jdays]
        for jday in tqdm(jdays, total=len(jdays), desc=f"{year}", position=i, leave=True):
            tmp = datetime.strptime(jday, "%Y.%j")

            with fs.open(f"{catalog_path}/{tmp.year:04d}.{tmp.month:02d}.event.csv", "rb") as f:
                events = pd.read_csv(f, parse_dates=["time"], date_format="%Y-%m-%dT%H:%M:%S.%f")
            events["time"] = pd.to_datetime(events["time"])
            events.set_index("event_id", inplace=True)
            with fs.open(f"{catalog_path}/{tmp.year:04d}.{tmp.month:02d}.phase.csv", "rb") as f:
                phases = pd.read_csv(
                    f,
                    parse_dates=["phase_time"],
                    date_format="%Y-%m-%dT%H:%M:%S.%f",
                    dtype={"location": str},
                )

            phases["phase_time"] = pd.to_datetime(phases["phase_time"])
            phases["phase_polarity"] = phases["phase_polarity"].fillna("N")
            phases["location"] = phases["location"].fillna("")
            phases["station_id"] = phases["network"] + "." + phases["station"] + "." + phases["location"]
            phases.sort_values(["event_id", "phase_time"], inplace=True)
            phases_by_station = phases.copy()
            phases_by_station.set_index(["station_id"], inplace=True)
            phases_by_event = phases.copy()
            phases_by_event.set_index(["event_id"], inplace=True)
            phases.set_index(["event_id", "station_id"], inplace=True)
            phases = phases.sort_index()

            event_ids = sorted(fs.ls(f"{mseed_path}/{year}/{jday}"), reverse=True)
            event_ids = [x.split("/")[-1] for x in event_ids]
            for event_id in event_ids:
                gp = fp.create_group(event_id)
                gp.attrs["event_id"] = event_id
                gp.attrs["event_time"] = events.loc[event_id, "time"].strftime("%Y-%m-%dT%H:%M:%S.%f")
                gp.attrs["latitude"] = events.loc[event_id, "latitude"]
                gp.attrs["longitude"] = events.loc[event_id, "longitude"]
                gp.attrs["depth_km"] = events.loc[event_id, "depth_km"]
                gp.attrs["magnitude"] = events.loc[event_id, "magnitude"]
                gp.attrs["magnitude_type"] = events.loc[event_id, "magnitude_type"]
                gp.attrs["source"] = "NC"

                mseed_list = sorted(list(fs.glob(f"{mseed_path}/{year}/{jday}/{event_id}/*.mseed")))
                st = obspy.Stream()
                for file in mseed_list:
                    with fs.open(file, "rb") as f:
                        st += obspy.read(f)
                arrival_time = phases.loc[event_id, "phase_time"].min()
                begin_time = arrival_time - pd.Timedelta(seconds=30)
                end_time = arrival_time + pd.Timedelta(seconds=90)
                gp.attrs["begin_time"] = begin_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
                gp.attrs["end_time"] = end_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
                gp.attrs["event_time_index"] = int(
                    round((events.loc[event_id, "time"] - begin_time).total_seconds() * 100)
                )
                gp.attrs["sampling_rate"] = sampling_rate
                gp.attrs["nt"] = 12000  # default 120s
                gp.attrs["nx"] = len(mseed_list)
                gp.attrs["delta"] = 1 / sampling_rate

                station_channel_ids = [x.split("/")[-1].replace(".mseed", "") for x in mseed_list]
                for station_channel_id in station_channel_ids:
                    ds = gp.create_dataset(station_channel_id, (3, gp.attrs["nt"]), dtype=np.float32)
                    tr = st.select(id=station_channel_id + "?")
                    for t in tr:
                        if t.stats.sampling_rate != sampling_rate:
                            t.resample(sampling_rate)

                    chn = [tr.stats.channel for tr in tr]
                    chn = sorted(chn, key=lambda x: order[x[-1]])
                    components = []
                    for i, t in enumerate(tr):
                        index0 = int(
                            round(
                                (t.stats.starttime.datetime.replace(tzinfo=timezone.utc) - begin_time).total_seconds()
                                * sampling_rate
                            )
                        )
                        if index0 > 3000:
                            # del ds
                            del fp[f"{event_id}/{station_channel_id}"]
                            break

                        if len(chn) != 3:
                            i = comp2idx[t.stats.channel[-1]]

                        if index0 > 0:
                            it1 = 0
                            it2 = index0
                            ll = min(len(t.data), len(ds[i, it2:]))  # data length
                        elif index0 < 0:
                            it1 = -index0
                            it2 = 0
                            ll = min(len(t.data[it1:]), len(ds[i, :]))
                        else:
                            it1 = 0
                            it2 = 0
                            ll = min(len(t.data), len(ds[i, :]))

                        ds[i, it2 : it2 + ll] = (t.data - np.mean(t.data))[it1 : it1 + ll] * 1e6
                        components.append(t.stats.channel[-1])

                    if index0 > 3000:
                        continue
                    network, station, location, instrument = station_channel_id.split(".")
                    ds.attrs["network"] = network
                    ds.attrs["station"] = station
                    ds.attrs["location"] = location
                    ds.attrs["instrument"] = instrument
                    ds.attrs["component"] = "".join(components)
                    ds.attrs["unit"] = "1e-6m/s" if instrument[-1] != "N" else "1e-6m/s**2"
                    ds.attrs["dt_s"] = 0.01

                    sta = (
                        stations.loc[(network, station, location, instrument)]
                        .sort_values("begin_time", ascending=False)
                        .iloc[0]
                    )
                    ds.attrs["longitude"] = sta["longitude"]
                    ds.attrs["latitude"] = sta["latitude"]
                    ds.attrs["elevation_m"] = sta["elevation_m"]
                    ds.attrs["local_depth_m"] = sta["local_depth_m"]
                    ds.attrs["depth_km"] = sta["depth_km"]

                    station_id = f"{network}.{station}.{location}"

                    picks_ = phases_by_station.loc[[station_id]]
                    picks_ = picks_[(picks_["phase_time"] > begin_time) & (picks_["phase_time"] < end_time)]
                    if len(picks_[picks_["event_id"] == event_id]) == 0:
                        print(f"{jday}.{event_id}.{network}.{station}.{location}: no picks")
                        # del ds
                        del fp[f"{event_id}/{station_channel_id}"]
                        continue

                    pick = picks_[picks_["event_id"] == event_id].iloc[0]  # after sort_value
                    ds.attrs["azimuth"] = pick.azimuth
                    ds.attrs["distance_km"] = pick.distance_km
                    ds.attrs["takeoff_angle"] = pick.takeoff_angle
                    snr = calc_snr(ds[:, :], int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate)))
                    tmp = int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate))
                    if ((tmp - 300) < 0) or ((tmp + 300) > 12000):
                        print(
                            f"{jday}.{event_id}.{network}.{station}.{location}: tmp={tmp}, {pick.phase_time}, {begin_time}"
                        )

                    if max(snr) == 0:
                        # print(f"{jday}.{event_id}.{network}.{station}.{location}: snr={snr}")
                        # del ds
                        del fp[f"{event_id}/{station_channel_id}"]
                        continue
                    ds.attrs["snr"] = snr

                    (
                        phase_type,
                        phase_index,
                        phase_score,
                        phase_time,
                        phase_remark,
                        phase_polarity,
                        phase_event_id,
                    ) = extract_pick(picks_, begin_time, sampling_rate)

                    ds.attrs["phase_type"] = phase_type
                    ds.attrs["phase_index"] = phase_index
                    ds.attrs["phase_score"] = phase_score
                    ds.attrs["phase_time"] = phase_time
                    ds.attrs["phase_remark"] = phase_remark
                    ds.attrs["phase_polarity"] = phase_polarity
                    ds.attrs["event_id"] = phase_event_id

                    if (
                        len(
                            np.array(phase_type)[(np.array(phase_event_id) == event_id) & (np.array(phase_type) == "S")]
                        )
                        > 0
                    ):
                        ds.attrs["phase_status"] = "manual"
                    else:
                        ds.attrs["phase_status"] = "automatic"

            # return


if __name__ == "__main__":
    # %%
    years = sorted(fs.ls(mseed_path), reverse=True)
    years = [x.split("/")[-1] for x in years]

    ncpu = len(years)
    ctx = mp.get_context("spawn")
    with ctx.Pool(ncpu) as pool:
        pool.starmap(convert, [x for x in enumerate(years)])

    # # check hdf5
    # with h5py.File("2000.h5", "r") as fp:
    #     for event_id in fp:
    #         print(event_id)
    #         for k in sorted(fp[event_id].attrs.keys()):
    #             print(k, fp[event_id].attrs[k])
    #         for station_id in fp[event_id]:
    #             print(station_id)
    #             print(fp[event_id][station_id].shape)
    #             for k in sorted(fp[event_id][station_id].attrs.keys()):
    #                 print(k, fp[event_id][station_id].attrs[k])
    #         raise
    # raise

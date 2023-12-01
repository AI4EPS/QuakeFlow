# %%
import multiprocessing as mp
import os
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fsspec
import h5py
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm
from glob import glob

# warnings.filterwarnings("error")
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# %%
root_path = "dataset"
waveform_path = f"gs/waveform_mseed"
catalog_path = f"{root_path}/catalog"
station_path = f"{root_path}/station"
result_path = f"dataset/waveform_h5"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# %%
comp = ["3", "2", "1", "U", "V", "E", "N", "Z"]
order = {key: i for i, key in enumerate(comp)}
comp2idx = {"3": 0, "2": 1, "1": 2, "U": 0, "V": 1, "E": 0, "N": 1, "Z": 2}  ## only for cases less than 3 components

sampling_rate = 100

# %%
stations = []
# for csv in fs.glob(f"{bucket}/station/*.csv"):
for csv in glob(f"{station_path}/*.csv"):
    stations.append(
        pd.read_csv(
            csv,
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
    for picks_ in picks:
        for idx, pick in picks_.iterrows():
            phase_type.append(idx[1])
            phase_index.append(int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate)))
            phase_score.append(pick.phase_score)
            phase_time.append(pick.phase_time.isoformat())
            phase_remark.append(pick.remark)
            phase_polarity.append(pick.phase_polarity)
            event_id.append(pick.event_id)

    return phase_type, phase_index, phase_score, phase_time, phase_remark, phase_polarity, event_id


# %%
def convert(i, year):
    # %%
    with h5py.File(f"{result_path}/{year}.h5", "w") as fp:
        jdays = sorted(glob(f"{waveform_path}/{year}/*"))[::-1]
        jdays = [x.split("/")[-1] for x in jdays]
        for jday in tqdm(jdays, total=len(jdays), desc=f"{year}", position=i):
            tmp = datetime.strptime(jday, "%Y.%j")

            events = pd.read_csv(
                f"{catalog_path}/{tmp.year:04d}.{tmp.month:02d}.event.csv", parse_dates=["event_time"])
            events.set_index("event_id", inplace=True)

            # phases = pd.read_csv(
            #     f"{protocol}://{bucket}/catalog/{tmp.year:04d}.{tmp.month:02d}.phase.csv",
            #     parse_dates=["phase_time"],
            #     date_format="%Y-%m-%dT%H:%M:%S.%f",
            #     dtype={"location": str},
            # )

            phases = pd.read_csv(
                f"{catalog_path}/{tmp.year:04d}.{tmp.month:02d}.phase.csv",
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
            phases_by_station.set_index(["station_id", "phase_type"], inplace=True)
            phases_by_event = phases.copy()
            phases_by_event.set_index(["event_id"], inplace=True)
            phases.set_index(["event_id", "station_id", "phase_type"], inplace=True)
            phases = phases.sort_index()

            event_ids = sorted(glob(f"{waveform_path}/{year}/{jday}/*"))[::-1]
            event_ids = [x.split("/")[-1] for x in event_ids]
            for event_id in event_ids:
                gp = fp.create_group(event_id)
                gp.attrs["event_id"] = event_id
                gp.attrs["event_time"] = events.loc[event_id, "event_time"].isoformat()
                gp.attrs["latitude"] = events.loc[event_id, "latitude"]
                gp.attrs["longitude"] = events.loc[event_id, "longitude"]
                gp.attrs["depth_km"] = events.loc[event_id, "depth_km"]
                gp.attrs["magnitude"] = events.loc[event_id, "magnitude"]
                gp.attrs["magnitude_type"] = events.loc[event_id, "magnitude_type"]
                gp.attrs["source"] = "NC"

                # st = obspy.read(str(event_id / "*.mseed"))
                # st = obspy.Stream()
                # for file in fs.glob(f"{bucket}/waveform_mseed/{year}/{jday}/{event_id}/*.mseed"):
                # for f in glob(f"{waveform_path}/{year}/{jday}/{event_id}/*.mseed"):
                    # with fs.open(file, "rb") as f:
                    # st += obspy.read(f)
                st = obspy.read(f"{waveform_path}/{year}/{jday}/{event_id}/*.mseed")
                # begin_time = min([tr.stats.starttime for tr in st]).datetime
                # begin_time = begin_time.replace(tzinfo=timezone.utc)
                # end_time = max([tr.stats.endtime for tr in st]).datetime
                # end_time = end_time.replace(tzinfo=timezone.utc)
                arrival_time = phases.loc[event_id, "phase_time"].min()
                begin_time = arrival_time - pd.Timedelta(seconds=30)
                end_time = arrival_time + pd.Timedelta(seconds=90)
                gp.attrs["begin_time"] = begin_time.isoformat()
                gp.attrs["end_time"] = end_time.isoformat()
                gp.attrs["event_time_index"] = int(
                    (events.loc[event_id, "event_time"] - begin_time).total_seconds() * 100
                )
                gp.attrs["sampling_rate"] = sampling_rate
                gp.attrs["nt"] = 12000  # default 120s
                # gp.attrs["nx"] = len(list(event_id.glob("*")))
                # gp.attrs["nx"] = len(fs.ls(f"{bucket}/waveform_mseed/{year}/{jday}/{event_id}/"))
                gp.attrs["nx"] = len(glob(f"{waveform_path}/{year}/{jday}/{event_id}/*"))
                gp.attrs["delta"] = 1 / sampling_rate

                # for station_id in event_id.glob("*"):
                # station_ids = fs.glob(f"{bucket}/waveform_mseed/{year}/{jday}/{event_id}/*.mseed")
                station_ids = glob(f"{waveform_path}/{year}/{jday}/{event_id}/*.mseed")
                station_ids = [x.split("/")[-1].replace(".mseed", "") for x in station_ids]
                for station_id in station_ids:
                    ds = gp.create_dataset(station_id, (3, gp.attrs["nt"]), dtype=np.float32)
                    tr = st.select(id=station_id + "?")
                    for t in tr:
                        if t.stats.sampling_rate != sampling_rate:
                            t.resample(sampling_rate)

                    chn = [tr.stats.channel for tr in tr]
                    chn = sorted(chn, key=lambda x: order[x[-1]])
                    components = []
                    if len(chn) == 3:
                        for i, t in enumerate(tr):
                            index0 = int(
                                round(
                                    (
                                        t.stats.starttime.datetime.replace(tzinfo=timezone.utc) - begin_time
                                    ).total_seconds()
                                    * sampling_rate
                                )
                            )
                            if index0 > 3000:
                                del ds
                                break
                            ds[i, index0 : index0 + len(t.data)] = (t.data - np.mean(t.data))[
                                : len(ds[i, index0:])
                            ] * 1e6
                            components.append(t.stats.channel[-1])
                    else:
                        for t in tr:
                            index0 = int(
                                round(
                                    (
                                        t.stats.starttime.datetime.replace(tzinfo=timezone.utc) - begin_time
                                    ).total_seconds()
                                    * sampling_rate
                                )
                            )
                            if index0 > 3000:
                                del ds
                                break
                            i = comp2idx[t.stats.channel[-1]]
                            ds[i, index0 : index0 + len(t.data)] = (t.data - np.mean(t.data))[
                                : len(ds[i, index0:])
                            ] * 1e6
                            components.append(t.stats.channel[-1])

                    if index0 > 3000:
                        continue
                    network, station, location, instrument = station_id.split(".")
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
                    # p_picks = phases.loc[[(event_id, station_id, "P")]].sort_values("phase_score").iloc[0]
                    # s_picks = phases.loc[[(event_id, station_id, "S")]].sort_values("phase_score").iloc[0]
                    # p_picks = phases.loc[[(event_id, station_id, "P")]].sort_values("phase_score").iloc[0]
                    # s_picks = phases.loc[[(event_id, station_id, "S")]].sort_values("phase_score").iloc[0]

                    ## pick not exist
                    if (event_id, station_id, "P") not in phases.index:
                        del ds
                        # print(f"{jday.name}.{event_id}.{network}.{station}.{location} not in P index")
                        # print(f"{jday.name}.{event_id}.{network}.{station}.{location} not in P index")
                        continue
                    if (event_id, station_id, "S") not in phases.index:
                        del ds
                        # print(f"{jday.name}.{event_id}.{network}.{station}.{location} not in S index")
                        # print(f"{jday.name}.{event_id}.{network}.{station}.{location} not in S index")
                        continue

                    p_picks = phases_by_station.loc[[(station_id, "P")]]
                    p_picks = p_picks[(p_picks["phase_time"] > begin_time) & (p_picks["phase_time"] < end_time)]
                    # p_picks = p_picks.loc[p_picks.groupby("event_id")["phase_score"].idxmin()]

                    s_picks = phases_by_station.loc[[(station_id, "S")]]
                    s_picks = s_picks[(s_picks["phase_time"] > begin_time) & (s_picks["phase_time"] < end_time)]
                    # s_picks = s_picks.loc[s_picks.groupby("event_id")["phase_score"].idxmin()]

                    if len(p_picks[p_picks["event_id"] == event_id]) == 0:
                        print(f"{jday.name}.{event_id}.{network}.{station}.{location}: no picks")
                        del ds
                        continue

                    pick = p_picks[p_picks["event_id"] == event_id].iloc[0]
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
                        # print(f"{jday.name}.{event_id}.{network}.{station}.{location}: snr={snr}")
                        del ds
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
                    ) = extract_pick([p_picks, s_picks], begin_time, sampling_rate)

                    ds.attrs["phase_type"] = phase_type
                    ds.attrs["phase_index"] = phase_index
                    ds.attrs["phase_score"] = phase_score
                    ds.attrs["phase_time"] = phase_time
                    ds.attrs["phase_remark"] = phase_remark
                    ds.attrs["phase_polarity"] = phase_polarity
                    ds.attrs["event_id"] = phase_event_id


if __name__ == "__main__":
    # years = sorted(list(waveform_path.glob("*")), reverse=True)
    # years = sorted(fs.ls(f"{bucket}/waveform_mseed/"), reverse=True)
    years = sorted(glob(f"{waveform_path}/*"))[::-1]
    years = [x.split("/")[-1] for x in years]
    # years = [x for x in years if x in ["1990"]]

    # for x in enumerate(years):
    #     convert(*x)

    ncpu = len(years)
    ctx = mp.get_context("spawn")
    with ctx.Pool(ncpu) as pool:
        pool.starmap(convert, [x for x in enumerate(years)])

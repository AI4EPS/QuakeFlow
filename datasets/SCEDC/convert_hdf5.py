# %%
import multiprocessing as mp
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm

os.environ["OPENBLAS_NUM_THREADS"] = "2"

# %%
root_path = Path("dataset")
waveform_path = root_path / "waveform"
catalog_path = root_path / "catalog"
station_path = root_path / "station"

# %%
comp = ["3", "2", "1", "U", "V", "E", "N", "Z"]
order = {key: i for i, key in enumerate(comp)}
comp2idx = {"3": 0, "2": 1, "1": 2, "U": 0, "V": 1, "E": 0, "N": 1, "Z": 2}  ## only for cases less than 3 components

sampling_rate = 100


# %%
stations = []
for csv in station_path.glob("*.csv"):
    stations.append(
        pd.read_csv(
            csv,
            parse_dates=["begin_time", "end_time"],
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
            phase_remark.append(pick.phase_remark)
            phase_polarity.append(pick.phase_polarity)
            event_id.append(pick.event_id)

    return phase_type, phase_index, phase_score, phase_time, phase_remark, phase_polarity, event_id


# %%
# for year in sorted(list(waveform_path.glob("*")), reverse=True)[1:]:
def convert(i, year):
    # %%
    with h5py.File(root_path / f"{year.name}.h5", "w") as fp:
        jdays = sorted(list(year.glob("*")))
        for jday in tqdm(jdays, total=len(jdays), desc=f"{year.name}", position=i):
            # tmp = pd.Timestamp(datetime.strptime(jday.name, "%Y.%j"))
            year, dayofyear = jday.name.split(".")

            # events = pd.read_csv(catalog_path / f"{tmp.year:04d}.{tmp.month:02d}.event.csv", parse_dates=["event_time"])
            events = pd.read_csv(
                catalog_path / f"{year}" / f"{year}_{dayofyear}.event.csv",
                parse_dates=["event_time"],
            )
            events.set_index("event_id", inplace=True)

            # phases = pd.read_csv(
            #     catalog_path / f"{tmp.year:04d}.{tmp.month:02d}.phase.csv",
            #     parse_dates=["phase_time"],
            #     dtype={"location": str},
            # )
            phases = pd.read_csv(
                catalog_path / f"{year}" / f"{year}_{dayofyear}.phase.csv",
                parse_dates=["phase_time"],
                dtype={"location": str},
            )
            phases["phase_polarity"] = phases["phase_polarity"].fillna("N")
            phases["location"] = phases["location"].fillna("")
            phases["station_id"] = phases["network"] + "." + phases["station"] + "." + phases["location"]
            phases.sort_values(["event_id", "phase_time"], inplace=True)
            phases_by_station = phases.copy()
            phases_by_station.set_index(["station_id", "phase_type"], inplace=True)
            phases.set_index(["event_id", "station_id", "phase_type"], inplace=True)
            phases = phases.sort_index()

            for event_id in sorted(list(jday.glob("*"))):
                gp = fp.create_group(event_id.name)
                gp.attrs["event_id"] = event_id.name
                gp.attrs["event_time"] = events.loc[event_id.name, "event_time"].isoformat()
                gp.attrs["latitude"] = events.loc[event_id.name, "latitude"]
                gp.attrs["longitude"] = events.loc[event_id.name, "longitude"]
                gp.attrs["depth_km"] = events.loc[event_id.name, "depth_km"]
                gp.attrs["magnitude"] = events.loc[event_id.name, "magnitude"]
                gp.attrs["magnitude_type"] = events.loc[event_id.name, "magnitude_type"]
                gp.attrs["source"] = "NC"

                st = obspy.read(str(event_id / "*.mseed"))
                begin_time = min([tr.stats.starttime for tr in st]).datetime
                begin_time = begin_time.replace(tzinfo=timezone.utc)
                end_time = max([tr.stats.endtime for tr in st]).datetime
                end_time = end_time.replace(tzinfo=timezone.utc)
                gp.attrs["begin_time"] = begin_time.isoformat()
                gp.attrs["end_time"] = end_time.isoformat()
                gp.attrs["event_time_index"] = int(
                    (events.loc[event_id.name, "event_time"] - begin_time).total_seconds() * 100
                )
                gp.attrs["sampling_rate"] = sampling_rate
                gp.attrs["nt"] = 12000  # default 120s
                gp.attrs["nx"] = len(list(event_id.glob("*")))
                gp.attrs["delta"] = 1 / sampling_rate

                for station_id in event_id.glob("*"):
                    ds = gp.create_dataset(station_id.stem, (3, gp.attrs["nt"]), dtype=np.float32)
                    tr = st.select(id=station_id.stem + "?")
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
                            ds[i, index0 : index0 + len(t.data)] = t.data[: len(ds[i, index0:])] * 1e6
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
                            ds[i, index0 : index0 + len(t.data)] = t.data[: len(ds[i, index0:])] * 1e6
                            components.append(t.stats.channel[-1])

                    if index0 > 3000:
                        continue
                    network, station, location, instrument = station_id.stem.split(".")
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
                    # p_picks = phases.loc[[(event_id.name, station_id, "P")]].sort_values("phase_score").iloc[0]
                    # s_picks = phases.loc[[(event_id.name, station_id, "S")]].sort_values("phase_score").iloc[0]

                    ## pick not exist
                    if (event_id.name, station_id, "P") not in phases.index:
                        del ds
                        # print(f"{jday.name}.{event_id.name}.{network}.{station}.{location} not in P index")
                        continue
                    if (event_id.name, station_id, "S") not in phases.index:
                        del ds
                        # print(f"{jday.name}.{event_id.name}.{network}.{station}.{location} not in S index")
                        continue

                    p_picks = phases_by_station.loc[[(station_id, "P")]]
                    p_picks = p_picks[(p_picks["phase_time"] > begin_time) & (p_picks["phase_time"] < end_time)]
                    p_picks = p_picks.loc[p_picks.groupby("event_id")["phase_score"].idxmin()]

                    s_picks = phases_by_station.loc[[(station_id, "S")]]
                    s_picks = s_picks[(s_picks["phase_time"] > begin_time) & (s_picks["phase_time"] < end_time)]
                    s_picks = s_picks.loc[s_picks.groupby("event_id")["phase_score"].idxmin()]

                    if len(p_picks[p_picks["event_id"] == event_id.name]) == 0:
                        print(f"{jday.name}.{event_id.name}.{network}.{station}.{location}: no picks")
                        del ds
                        continue

                    pick = p_picks[p_picks["event_id"] == event_id.name].iloc[0]

                    if "distance_km" in pick.index:
                        ds.attrs["distance_km"] = pick.distance_km
                    if "back_azimuth" in pick.index:
                        ds.attrs["back_azimuth"] = pick.back_azimuth
                    if "takeoff_angle" in pick.index:
                        ds.attrs["takeoff_angle"] = pick.takeoff_angle
                    snr = calc_snr(ds[:, :], int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate)))
                    tmp = int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate))
                    if ((tmp - 300) < 0) or ((tmp + 300) > 12000):
                        print(
                            f"{jday.name}.{event_id.name}.{network}.{station}.{location}: tmp={tmp}, {pick.phase_time}, {begin_time}"
                        )

                    if max(snr) == 0:
                        # print(f"{jday.name}.{event_id.name}.{network}.{station}.{location}: snr={snr}")
                        del ds
                        continue

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
    years = sorted(list(waveform_path.glob("*")), reverse=True)
    # select years in [2013, 2014, 2015]
    years = [x for x in years if x.name in ["2021", "2020", "2019", "2018"]]

    # for x in enumerate(years):
    #     convert(*x)

    ncpu = len(years)
    with mp.get_context("spawn").Pool(ncpu) as pool:
        pool.starmap(convert, [x for x in enumerate(years)])

# %%
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm

# %%
root_path = Path("dataset")
waveform_path = root_path / "waveform"
catalog_path = root_path / "catalog"
station_path = Path("../station")

# %%
comp = ["3", "2", "1", "U", "V", "E", "N", "Z"]
order = {key: i for i, key in enumerate(comp)}
comp2idx = {"3": 0, "2": 1, "1": 2, "U": 0, "V": 1, "E": 0, "N": 1, "Z": 2}  ## only for cases less than 3 components

sampling_rate = 100


# %%
def parse_inventory_csv(inventory):
    channel_list = []
    for network in inventory:
        for station in network:
            for channel in station:
                if channel.sensor is None:
                    sensor_description = ""
                else:
                    sensor_description = channel.sensor.description
                channel_list.append(
                    {
                        "network": network.code,
                        "station": station.code,
                        "location": channel.location_code,
                        "instrument": channel.code[:-1],
                        "component": channel.code[-1],
                        "channel": channel.code,
                        "longitude": channel.longitude,
                        "latitude": channel.latitude,
                        "elevation_m": channel.elevation,
                        "depth_km": -channel.elevation / 1e3,
                        # "depth_km": channel.depth,
                        "begin_time": channel.start_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                        if channel.start_date is not None
                        else None,
                        "end_time": channel.end_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                        if channel.end_date is not None
                        else None,
                        "azimuth": channel.azimuth,
                        "dip": channel.dip,
                        "sensitivity": channel.response.instrument_sensitivity.value,
                        "site": station.site.name,
                        "sensor": sensor_description,
                    }
                )
    channel_list = pd.DataFrame(channel_list)

    print(f"Parse {len(channel_list)} channels into csv")

    return channel_list


for network in station_path.glob("*.info"):
    if network.name in ["broadband.info", "BARD.info", "CISN.info"]:
        continue
    inv = obspy.Inventory()
    for xml in (network / f"{network.stem}.FDSN.xml").glob(f"{network.stem}.*.xml"):
        inv += obspy.read_inventory(xml)
    stations = parse_inventory_csv(inv)
    if not (root_path / "station").exists():
        (root_path / "station").mkdir(parents=True)
    stations.to_csv(root_path / "station" / f"{network.stem}.csv", index=False)


# %%

for year in sorted(list(waveform_path.glob("*")), reverse=True):
    # # %%
    # phases = []
    # for phase_file in catalog_path.glob(f"{year.name}.??.phase.csv"):
    #     phases.append(pd.read_csv(phase_file, parse_dates=["phase_time"], dtype={"location": str}))
    # phases = pd.concat(phases)
    # phases["phase_polarity"] = phases["phase_polarity"].fillna("N")
    # phases["location"] = phases["location"].fillna("")
    # phases["station_id"] = phases["network"] + "." + phases["station"] + "." + phases["location"]
    # phases.set_index(["event_id", "station_id", "phase_type"], inplace=True)
    # events = []
    # for event_file in catalog_path.glob(f"{year.name}.??.event.csv"):
    #     events.append(pd.read_csv(event_file, parse_dates=["event_time"]))
    # events = pd.concat(events)
    # events.set_index("event_id", inplace=True)

    # %%
    year = sorted(list(waveform_path.glob("*")), reverse=True)[0]

    # %%
    with h5py.File(root_path / f"{year.name}.h5", "w") as fp:
        jdays = sorted(list(year.glob("*")))
        for jday in tqdm(jdays, total=len(jdays), desc=f"{year.name}"):
            tmp = datetime.strptime(jday.name, "%Y.%j")

            events = pd.read_csv(catalog_path / f"{tmp.year:04d}.{tmp.month:02d}.event.csv", parse_dates=["event_time"])
            events.set_index("event_id", inplace=True)

            phases = pd.read_csv(
                catalog_path / f"{tmp.year:04d}.{tmp.month:02d}.phase.csv",
                parse_dates=["phase_time"],
                dtype={"location": str},
            )
            phases["phase_polarity"] = phases["phase_polarity"].fillna("N")
            phases["location"] = phases["location"].fillna("")
            phases["station_id"] = phases["network"] + "." + phases["station"] + "." + phases["location"]
            phases_by_station = phases.copy()
            phases_by_station.set_index(["station_id", "phase_type"], inplace=True)
            phases.set_index(["event_id", "station_id", "phase_type"], inplace=True)

            for event_id in jday.glob("*"):
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
                            if index0 > 1:
                                print(f"{jday}/{event_id.name}.{station_id.stem}.{t.stats.channel} index0: {index0}")
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
                            if index0 > 1:
                                print(f"{jday}/{event_id.name}.{station_id.stem}.{t.stats.channel} index0: {index0}")
                            i = comp2idx[t.stats.channel[-1]]
                            ds[i, index0 : index0 + len(t.data)] = t.data[: len(ds[i, index0:])] * 1e6
                            components.append(t.stats.channel[-1])

                    network, station, location, instrument = station_id.stem.split(".")
                    ds.attrs["network"] = network
                    ds.attrs["station"] = station
                    ds.attrs["location"] = location
                    ds.attrs["instrument"] = instrument
                    ds.attrs["component"] = "".join(components)

                    # %%

                    # %%
                    station_id = f"{network}.{station}.{location}"
                    # p_picks = phases.loc[[(event_id.name, station_id, "P")]].sort_values("phase_score").iloc[0]
                    # s_picks = phases.loc[[(event_id.name, station_id, "S")]].sort_values("phase_score").iloc[0]

                    p_picks = phases_by_station.loc[[(station_id, "P")]]
                    p_picks = p_picks[(p_picks["phase_time"] > begin_time) & (p_picks["phase_time"] < end_time)]
                    p_picks = p_picks.loc[p_picks.groupby("event_id")["phase_score"].idxmin()]

                    s_picks = phases_by_station.loc[[(station_id, "S")]]
                    s_picks = s_picks[(s_picks["phase_time"] > begin_time) & (s_picks["phase_time"] < end_time)]
                    s_picks = s_picks.loc[s_picks.groupby("event_id")["phase_score"].idxmin()]

                    # %%
                    raise


# %%

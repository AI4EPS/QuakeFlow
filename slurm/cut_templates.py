# %%
import json
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gamma
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm


def extract_template_numpy(template_array, travel_time_array, travel_time_index_array, travel_time_type_array, snr_array,
                           mseed_path, events, stations, picks, config, output_path, figure_path, lock, ibar):

    # %%
    tmp = str(mseed_path).split("/")
    year_jday, hour = tmp[-2], tmp[-1]
    begin_time = datetime.strptime(f"{year_jday}T{hour}", "%Y-%jT%H").replace(tzinfo=timezone.utc)
    end_time = begin_time + timedelta(hours=1)
    events_ = events[(events["event_time"] > begin_time) & (events["event_time"] < end_time)]

    if len(events_) == 0:
        return 0

    # %%
    waveforms_dict = {}
    for station_id in tqdm(stations["station_id"], desc=f"Loading: ", position=ibar%6, nrows=7, mininterval=5, leave=False):
        for c in config.components:
            if (mseed_path / f"{station_id}{c}.mseed").exists():
                try:
                    stream = obspy.read(mseed_path / f"{station_id}{c}.mseed")
                    # stream.merge(method=1, interpolation_samples=0)
                    stream.merge(fill_value="latest")
                    if len(stream) > 1:
                        print(f"More than one trace: {stream}")
                    trace = stream[0]
                    if trace.stats.sampling_rate != config.sampling_rate:
                        if trace.stats.sampling_rate % config.sampling_rate == 0:
                            trace.decimate(int(trace.stats.sampling_rate / config.sampling_rate))
                        else:
                            trace.resample(config.sampling_rate)
                    trace.detrend("linear")
                    # trace.taper(max_percentage=0.05, type="cosine")
                    trace.filter("bandpass", freqmin=1.0, freqmax=15.0, corners=2, zerophase=True)
                    waveforms_dict[f"{station_id}{c}"] = trace
                except Exception as e:
                    print(e)
                    continue

    # %%
    picks["station_component_index"] = picks.apply(lambda x: f"{x.station_id}.{x.phase_type}", axis=1)

    # %%
    num_event = 0
    for event_index in tqdm(events_["event_index"], desc=f"Cutting event {year_jday}T{hour}", position=ibar%6, nrows=7, mininterval=5, leave=False):

        if event_index not in picks.index:
            continue

        picks_ = picks.loc[[event_index]]
        picks_ = picks_.set_index("station_component_index")

        event_loc = events_.loc[event_index][["x_km", "y_km", "z_km"]].to_numpy().astype(np.float32)
        event_loc = np.hstack((event_loc, [0]))[np.newaxis, :]
        station_loc = stations[["x_km", "y_km", "z_km"]].to_numpy()

        template_ = np.zeros((6, len(stations), config.nt), dtype=np.float32)
        snr_ = np.zeros((6, len(stations)), dtype=np.float32)
        travel_time_ = np.zeros((2, len(stations)), dtype=np.float32)
        travel_time_type_ = np.zeros((2, len(stations)), dtype=np.int32)

        for i, phase_type in enumerate(["P", "S"]):

            travel_time = gamma.seismic_ops.calc_time(
                event_loc,
                station_loc,
                [phase_type.lower() for _ in range(len(station_loc))],
                vel={"p": 6.0, "s": 6.0 / 1.73},
            ).squeeze()

            phase_timestamp_pred = events_.loc[event_index]["event_timestamp"] + travel_time
            # predicted_phase_time = [events_.loc[event_index]["event_time"] + pd.Timedelta(seconds=x) for x in travel_time]

            mean_shift = []
            for j, station_id in enumerate(stations["station_id"]):
                if f"{station_id}.{phase_type}" in picks_.index:
                    ## TODO: check if multiple phases for the same station
                    phase_timestamp = picks_.loc[f"{station_id}.{phase_type}"]["phase_timestamp"]
                    phase_timestamp_pred[j] = phase_timestamp
                    mean_shift.append(phase_timestamp - (events_.loc[event_index]["event_timestamp"] + travel_time[j]))
                    
                    travel_time[j] = phase_timestamp - events_.loc[event_index]["event_timestamp"]
                    travel_time_type_[i, j] = 1
                else:
                    travel_time_type_[i, j] = 0

            # if len(mean_shift) > 0:
            #     mean_shift = float(np.median(mean_shift))
            # else:
            #     mean_shift = 0
            # phase_timestamp_pred[travel_time_type_[i, :] == 0] += mean_shift
            # travel_time[travel_time_type_[i, :] == 0] += mean_shift
            # travel_time_[i, :] = travel_time

            for c in config.components:
                
                c_index = i*3 + config.component_mapping[c]
                empty_data = True

                for j, station_id in enumerate(stations["station_id"]):

                    if f"{station_id}{c}" in waveforms_dict:

                        trace = waveforms_dict[f"{station_id}{c}"]

                        begin_time = (
                            phase_timestamp_pred[j]
                            - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
                            - config.time_before
                        )
                        end_time = (
                            phase_timestamp_pred[j]
                            - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
                            + config.time_after
                        )
                        

                        
                        trace_data = trace.data[
                            max(0, int(begin_time * trace.stats.sampling_rate)) : max(0, int(
                                end_time * trace.stats.sampling_rate)
                            )
                        ].astype(np.float32)


                        if len(trace_data) < config.nt:
                            continue
                        std = np.std(trace_data)
                        if std == 0:
                            continue


                        empty_data = False
                        template_[c_index, j, : config.nt] = trace_data[: config.nt]
                        s = np.std(trace_data[-int(config.time_after * config.sampling_rate) :])
                        n = np.std(trace_data[: int(config.time_before * config.sampling_rate)])
                        if n == 0:
                            snr_[c_index, j] = 0
                        else:
                            snr_[c_index, j] = s / n


        template_array[event_index] = template_
        travel_time_array[event_index] = travel_time_
        travel_time_index_array[event_index] = np.round(travel_time_ * config.sampling_rate).astype(np.int32)
        travel_time_type_array[event_index] = travel_time_type_
        snr_array[event_index] = snr_

        with lock:
            template_array.flush()
            travel_time_array.flush()
            travel_time_index_array.flush()
            travel_time_type_array.flush()
            snr_array.flush()

        # num_event += 1
        # if num_event > 20:
        #     break

# %%
result_path = Path("results/")
mseed_path = Path("waveforms/")
figure_path = Path("figures/")
output_path = Path("templates/")
if not figure_path.exists():
    figure_path.mkdir(parents=True)
if not output_path.exists():
    output_path.mkdir()

# %%
@dataclass
class Config:
    sampling_rate: int = 100
    time_before: float = 0.25
    time_after: float = 1.0
    components: str = "ENZ123"
    degree2km: float = 111.2

    def __init__(self, **kwargs):
        self.nt = int((self.time_before + self.time_after) * self.sampling_rate)
        # self.component_mapping = {"E": "E", "N": "N", "Z": "Z", "3": "E", "2": "N", "1": "Z"}
        self.component_mapping = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}
        self.__dict__.update(kwargs)

with open(result_path / "config.json", "r") as f:
    config = Config(**json.load(f))
print(config.__dict__)

# %%
stations = pd.read_json(result_path / "stations.json", orient="index")
stations["station_id"] = stations.index
stations = stations[
    (stations["longitude"] > config.xlim_degree[0])
    & (stations["longitude"] < config.xlim_degree[1])
    & (stations["latitude"] > config.ylim_degree[0])
    & (stations["latitude"] < config.ylim_degree[1])
]
# stations["distance_km"] = stations.apply(
#     lambda x: math.sqrt((x.latitude - config.center[1]) ** 2 + (x.longitude - config.center[0]) ** 2)
#     * config.degree2km,
#     axis=1,
# )
# stations.sort_values(by="distance_km", inplace=True)
# stations.drop(columns=["distance_km"], inplace=True)
# stations.sort_values(by="latitude", inplace=True)
stations["x_km"] = stations.apply(
    lambda x: (x.longitude - config.center[0]) * np.cos(np.deg2rad(config.center[1])) * config.degree2km, axis=1
)
stations["y_km"] = stations.apply(lambda x: (x.latitude - config.center[1]) * config.degree2km, axis=1)
stations["z_km"] = stations.apply(lambda x: -x.elevation_m / 1e3, axis=1)

# %%
events = pd.read_csv(
    result_path/"gamma_catalog.csv", parse_dates=["time"], date_parser=lambda x: pd.to_datetime(x, utc=True)
)
events["depth_km"] = events["depth(m)"]/1e3
events = events[events["time"].notna()]
events.sort_values(by="time", inplace=True)
events.rename(columns={"time": "event_time"}, inplace=True)
events["event_timestamp"] = events["event_time"].apply(lambda x: x.timestamp())
events["x_km"] = events.apply(
    lambda x: (x.longitude - config.center[0]) * np.cos(np.deg2rad(config.center[1])) * config.degree2km, axis=1
)
events["y_km"] = events.apply(lambda x: (x.latitude - config.center[1]) * config.degree2km, axis=1)
events["z_km"] = events.apply(lambda x: x.depth_km, axis=1)

# %%
event_index = list(events["event_index"])
with open(output_path/"event_index.txt", "w") as f:
    for i in event_index:
        f.write(f"{i}\n")

# %%
picks = pd.read_csv(
    result_path/"gamma_picks.csv", parse_dates=["phase_time"], date_parser=lambda x: pd.to_datetime(x, utc=True)
)
picks = picks[picks["event_index"] != -1]
picks["phase_timestamp"] = picks["phase_time"].apply(lambda x: x.timestamp())

picks_ = picks.groupby("station_id").size()
station_id_ = picks_[picks_ > (picks_.sum() / len(picks_) * 0.1)].index
stations = stations[stations["station_id"].isin(station_id_)]
stations.to_json(output_path/"stations_filtered.json", orient="index", indent=4)

# %%
picks = picks.merge(stations, on="station_id")
picks = picks.merge(events, on="event_index", suffixes=("_station", "_event"))

# %%
events["index"] = events["event_index"]
events = events.set_index("index")
picks["index"] = picks["event_index"]
picks = picks.set_index("index")

# %%
nt = config.nt
nch = 6 ## For [P,S] phases and [E,N,Z] components
nev = int(events.index.max()) + 1
nst = len(stations)
print(f"nev: {nev}, nst: {nst}, nch: {nch}, nt: {nt}")
template_shape = (nev, nch, nst, nt)
traveltime_shape = (nev, nch//3, nst)
snr_shape = (nev, nch, nst)
config.template_shape = template_shape
config.traveltime_shape = traveltime_shape
config.snr_shape = snr_shape
template_array = np.memmap(output_path/"template.dat", dtype=np.float32, mode="w+", shape=template_shape)
travel_time_array = np.memmap(output_path/"travel_time.dat", dtype=np.float32, mode="w+", shape=traveltime_shape)
travel_time_index_array = np.memmap(output_path/"travel_time_index.dat", dtype=np.int32, mode="w+", shape=traveltime_shape)
travel_time_type_array = np.memmap(output_path/"travel_time_type.dat", dtype=np.int32, mode="w+", shape=traveltime_shape)
snr_array = np.memmap(output_path/"snr.dat", dtype=np.float32, mode="w+", shape=snr_shape)

# %%
with open(output_path/"config.json", "w") as f:
    json.dump(config.__dict__, f, indent=4)

# %%
dirs = [hour_dir  for jday_dir in sorted(list(mseed_path.iterdir()))[::-1] for hour_dir in sorted(list(jday_dir.iterdir()))]
ncpu = mp.cpu_count()
lock = mp.Lock()
processes = []
for i, d in enumerate(dirs):
    proc = mp.Process(target=extract_template_numpy, args=(
        template_array, travel_time_array, travel_time_index_array, travel_time_type_array, snr_array,
        d, events, stations, picks, config, output_path, figure_path, lock, i%ncpu))
    processes.append(proc)
    proc.start()
    if len(processes) == ncpu:
        for proc in processes:
            proc.join()
        processes = []
for proc in processes:
    proc.join()
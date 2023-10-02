# %%
from typing import Dict, List

from kfp import dsl


@dsl.component(base_image="zhuwq0/quakeflow:latest")
def cut_templates(
    root_path: str,
    region: str,
    config: Dict,
    index: int = 0,
    model_path: str = "../PhaseNet/",
    mseed_list: List = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:
    import json
    import multiprocessing as mp
    import os
    from dataclasses import dataclass
    from datetime import datetime, timedelta, timezone
    from glob import glob
    from multiprocessing.pool import ThreadPool
    from pathlib import Path

    import gamma
    import numpy as np
    import obspy
    import pandas as pd
    from tqdm import tqdm

    def extract_template_numpy(
        template_fname,
        traveltime_fname,
        traveltime_index_fname,
        traveltime_type_fname,
        arrivaltime_index_fname,
        snr_fname,
        mseed_path,
        events,
        stations,
        picks,
        config,
        lock,
        ibar,
    ):
        template_array = np.memmap(template_fname, dtype=np.float32, mode="r+", shape=tuple(config["template_shape"]))
        traveltime_array = np.memmap(
            traveltime_fname, dtype=np.float32, mode="r+", shape=tuple(config["traveltime_shape"])
        )
        traveltime_index_array = np.memmap(
            traveltime_index_fname, dtype=np.int32, mode="r+", shape=tuple(config["traveltime_shape"])
        )
        traveltime_type_array = np.memmap(
            traveltime_type_fname, dtype=np.int32, mode="r+", shape=tuple(config["traveltime_shape"])
        )
        arrivaltime_index_array = np.memmap(
            arrivaltime_index_fname, dtype=np.int64, mode="r+", shape=tuple(config["traveltime_shape"])
        )
        snr_array = np.memmap(snr_fname, dtype=np.float32, mode="r+", shape=tuple(config["snr_shape"]))

        # %%
        tmp = mseed_path.split("/")
        year_jday, hour = tmp[-2], tmp[-1]
        begin_time = datetime.strptime(f"{year_jday}T{hour}", "%Y-%jT%H").replace(tzinfo=timezone.utc)
        end_time = begin_time + timedelta(hours=1)
        events_ = events[(events["event_time"] > begin_time) & (events["event_time"] < end_time)]

        if len(events_) == 0:
            return 0

        # %%
        waveforms_dict = {}
        for station_id in tqdm(
            stations["station_id"], desc=f"Loading waveform: ", position=ibar % 6, nrows=7, mininterval=5, leave=True
        ):
            for c in config["components"]:
                if os.path.exists(f"{mseed_path}/{station_id}{c}.mseed"):
                    try:
                        stream = obspy.read(f"{mseed_path}/{station_id}{c}.mseed")
                        stream.merge(fill_value="latest")
                        if len(stream) > 1:
                            print(f"More than one trace: {stream}")
                        trace = stream[0]
                        if trace.stats.sampling_rate != config["sampling_rate"]:
                            if trace.stats.sampling_rate % config["sampling_rate"] == 0:
                                trace.decimate(int(trace.stats.sampling_rate / config["sampling_rate"]))
                            else:
                                trace.resample(config["sampling_rate"])
                        # trace.detrend("linear")
                        # trace.taper(max_percentage=0.05, type="cosine")
                        # trace.filter("bandpass", freqmin=1.0, freqmax=15.0, corners=2, zerophase=True)
                        waveforms_dict[f"{station_id}{c}"] = trace
                    except Exception as e:
                        print(e)
                        continue

        # %%
        picks["station_component_index"] = picks.apply(lambda x: f"{x.station_id}.{x.phase_type}", axis=1)

        # %%
        num_event = 0
        for ii, event_index in tqdm(
            enumerate(events_["event_index"]),
            total=len(events_),
            desc=f"Cutting event {year_jday}T{hour}",
            position=ibar % 6,
            nrows=7,
            mininterval=5,
            leave=True,
        ):
            if event_index not in picks.index:
                continue

            picks_ = picks.loc[[event_index]]
            picks_ = picks_.set_index("station_component_index")

            event_loc = events_.loc[event_index][["x_km", "y_km", "z_km"]].to_numpy().astype(np.float32)
            event_loc = np.hstack((event_loc, [0]))[np.newaxis, :]
            station_loc = stations[["x_km", "y_km", "z_km"]].to_numpy()

            template_ = np.zeros((6, len(stations), config["nt"]), dtype=np.float32)
            snr_ = np.zeros((6, len(stations)), dtype=np.float32)
            traveltime_ = np.zeros((2, len(stations)), dtype=np.float32)
            traveltime_type_ = np.zeros((2, len(stations)), dtype=np.int32)
            arrivaltime_index_ = np.zeros((2, len(stations)), dtype=np.int64)

            for i, phase_type in enumerate(["P", "S"]):
                traveltime = gamma.seismic_ops.calc_time(
                    event_loc,
                    station_loc,
                    [phase_type.lower() for _ in range(len(station_loc))],
                    vel={"p": 6.0, "s": 6.0 / 1.73},
                ).squeeze()

                phase_timestamp_pred = events_.loc[event_index]["event_timestamp"] + traveltime
                # predicted_phase_time = [events_.loc[event_index]["event_time"] + pd.Timedelta(seconds=x) for x in traveltime]

                mean_shift = []
                for j, station_id in enumerate(stations["station_id"]):
                    if f"{station_id}.{phase_type}" in picks_.index:
                        ## TODO: check if multiple phases for the same station
                        phase_timestamp = picks_.loc[f"{station_id}.{phase_type}"]["phase_timestamp"]
                        phase_timestamp_pred[j] = phase_timestamp
                        mean_shift.append(
                            phase_timestamp - (events_.loc[event_index]["event_timestamp"] + traveltime[j])
                        )

                        traveltime[j] = phase_timestamp - events_.loc[event_index]["event_timestamp"]
                        traveltime_type_[i, j] = 1
                        arrivaltime_index_[i, j] = int(round(phase_timestamp * config["sampling_rate"]))
                        # arrivaltime_index_[i, j] = phase_timestamp
                    else:
                        traveltime_type_[i, j] = 0

                # if len(mean_shift) > 0:
                #     mean_shift = float(np.median(mean_shift))
                # else:
                #     mean_shift = 0
                # phase_timestamp_pred[traveltime_type_[i, :] == 0] += mean_shift
                # traveltime[traveltime_type_[i, :] == 0] += mean_shift
                traveltime_[i, :] = traveltime

                for j, station in enumerate(stations.iloc):
                    station_id = station["station_id"]

                    empty_data = True
                    for c in station["component"]:
                        c_index = i * 3 + config["component_mapping"][c]

                        if f"{station_id}{c}" in waveforms_dict:
                            trace = waveforms_dict[f"{station_id}{c}"]

                            begin_time = (
                                phase_timestamp_pred[j]
                                - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
                                - config["time_before"]
                            )
                            end_time = (
                                phase_timestamp_pred[j]
                                - trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()
                                + config["time_after"]
                            )

                            trace_data = trace.data[
                                max(0, int(begin_time * trace.stats.sampling_rate)) : max(
                                    0, int(end_time * trace.stats.sampling_rate)
                                )
                            ].astype(np.float32)

                            if len(trace_data) < config["nt"]:
                                continue
                            std = np.std(trace_data)
                            if std == 0:
                                continue

                            empty_data = False
                            template_[c_index, j, : config["nt"]] = trace_data[: config["nt"]]
                            s = np.std(trace_data[-int(config["time_after"] * config["sampling_rate"]) :])
                            n = np.std(trace_data[: int(config["time_before"] * config["sampling_rate"])])
                            if n == 0:
                                snr_[c_index, j] = 0
                            else:
                                snr_[c_index, j] = s / n

            # template_array[event_index] = template_
            # traveltime_array[event_index] = traveltime_
            # traveltime_index_array[event_index] = np.round(traveltime_ * config["sampling_rate"]).astype(np.int32)
            # traveltime_type_array[event_index] = traveltime_type_
            # arrivaltime_index_array[event_index] = arrivaltime_index_
            # snr_array[event_index] = snr_
            template_array[ii] = template_
            traveltime_array[ii] = traveltime_
            traveltime_index_array[ii] = np.round(traveltime_ * config["sampling_rate"]).astype(np.int32)
            traveltime_type_array[ii] = traveltime_type_
            arrivaltime_index_array[ii] = arrivaltime_index_
            snr_array[ii] = snr_

            with lock:
                template_array.flush()
                traveltime_array.flush()
                traveltime_index_array.flush()
                traveltime_type_array.flush()
                arrivaltime_index_array.flush()
                snr_array.flush()

            # num_event += 1
            # if num_event > 20:
            #     break

    # %%
    result_path = f"{region}/cctorch"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    stations = pd.read_json(f"{root_path}/{region}/obspy/stations.json", orient="index")
    stations["station_id"] = stations.index
    # stations = stations[
    #     (stations["longitude"] >= config.xlim_degree[0])
    #     & (stations["longitude"] =< config.xlim_degree[1])
    #     & (stations["latitude"] >= config.ylim_degree[0])
    #     & (stations["latitude"] <= config.ylim_degree[1])
    # ]
    # stations["distance_km"] = stations.apply(
    #     lambda x: math.sqrt((x.latitude - config.latitude0) ** 2 + (x.longitude - config.longitude0) ** 2)
    #     * config.degree2km,
    #     axis=1,
    # )
    # stations.sort_values(by="distance_km", inplace=True)
    # stations.drop(columns=["distance_km"], inplace=True)
    # stations.sort_values(by="latitude", inplace=True)
    stations["x_km"] = stations.apply(
        lambda x: (x.longitude - config["longitude0"]) * np.cos(np.deg2rad(config["latitude0"])) * config["degree2km"],
        axis=1,
    )
    stations["y_km"] = stations.apply(lambda x: (x.latitude - config["latitude0"]) * config["degree2km"], axis=1)
    stations["z_km"] = stations.apply(lambda x: -x["elevation_m"] / 1e3, axis=1)

    # %%
    events = pd.read_csv(f"{root_path}/{region}/gamma/gamma_events.csv", parse_dates=["time"])
    events = events[events["time"].notna()]
    events.sort_values(by="time", inplace=True)
    events.rename(columns={"time": "event_time"}, inplace=True)
    events["event_time"] = events["event_time"].apply(lambda x: pd.to_datetime(x, utc=True))
    events["event_timestamp"] = events["event_time"].apply(lambda x: x.timestamp())
    events["x_km"] = events.apply(
        lambda x: (x.longitude - config["longitude0"]) * np.cos(np.deg2rad(config["latitude0"])) * config["degree2km"],
        axis=1,
    )
    events["y_km"] = events.apply(lambda x: (x.latitude - config["latitude0"]) * config["degree2km"], axis=1)
    events["z_km"] = events.apply(lambda x: x.depth_km, axis=1)

    # %%
    if "event_index" not in events.columns:
        event_index = events.index
    else:
        event_index = list(events["event_index"])
    event_index_fname = f"{root_path}/{result_path}/event_index.txt"
    with open(event_index_fname, "w") as f:
        for i, idx in enumerate(event_index):
            f.write(f"{i},{idx}\n")
    config["cctorch"]["event_index_file"] = event_index_fname

    # %%
    picks = pd.read_csv(
        f"{root_path}/{region}/gamma/gamma_picks.csv",
        parse_dates=["phase_time"],
    )
    picks = picks[picks["event_index"] != -1]
    picks["phase_timestamp"] = picks["phase_time"].apply(lambda x: x.timestamp())

    picks_ = picks.groupby("station_id").size()
    # station_id_ = picks_[picks_ > (picks_.sum() / len(picks_) * 0.1)].index
    # stations = stations[stations["station_id"].isin(station_id_)]
    stations = stations[stations["station_id"].isin(picks_.index)]

    stations.to_json(f"{root_path}/{result_path}/stations_filtered.json", orient="index", indent=4)
    stations.to_csv(f"{root_path}/{result_path}/stations_filtered.csv", index=True, index_label="station_id")

    station_index_fname = f"{root_path}/{result_path}/station_index.txt"
    with open(station_index_fname, "w") as f:
        for i, sta in enumerate(stations.iloc):
            f.write(f"{i},{sta['station_id']},{sta['component']}\n")
    config["cctorch"]["station_index_file"] = station_index_fname

    # %%
    picks = picks.merge(stations, on="station_id")
    picks = picks.merge(events, on="event_index", suffixes=("_station", "_event"))

    # %%
    events["index"] = events["event_index"]
    events.set_index("index", inplace=True)
    picks["index"] = picks["event_index"]
    picks.set_index("index", inplace=True)

    # %%
    nt = int((config["cctorch"]["time_before"] + config["cctorch"]["time_after"]) * config["cctorch"]["sampling_rate"])
    config["cctorch"]["nt"] = nt
    nch = 6  ## For [P,S] phases and [E,N,Z] components
    # nev = int(events.index.max()) + 1
    nev = len(events)
    nst = len(stations)
    print(f"nev: {nev}, nch: {nch}, nst: {nst}, nt: {nt}")
    template_shape = (nev, nch, nst, nt)
    traveltime_shape = (nev, nch // 3, nst)
    snr_shape = (nev, nch, nst)
    config["cctorch"]["template_shape"] = template_shape
    config["cctorch"]["traveltime_shape"] = traveltime_shape
    config["cctorch"]["snr_shape"] = snr_shape

    template_fname = f"{root_path}/{result_path}/template.dat"
    traveltime_fname = f"{root_path}/{result_path}/traveltime.dat"
    traveltime_index_fname = f"{root_path}/{result_path}/traveltime_index.dat"
    traveltime_type_fname = f"{root_path}/{result_path}/traveltime_type.dat"
    arrivaltime_index_fname = f"{root_path}/{result_path}/arrivaltime_index.dat"
    snr_fname = f"{root_path}/{result_path}/snr.dat"
    config["cctorch"]["template_file"] = template_fname
    config["cctorch"]["traveltime_file"] = traveltime_fname
    config["cctorch"]["traveltime_index_file"] = traveltime_index_fname
    config["cctorch"]["traveltime_type_file"] = traveltime_type_fname
    config["cctorch"]["arrivaltime_index_file"] = arrivaltime_index_fname
    config["cctorch"]["snr_file"] = snr_fname

    template_array = np.memmap(template_fname, dtype=np.float32, mode="w+", shape=template_shape)
    traveltime_array = np.memmap(traveltime_fname, dtype=np.float32, mode="w+", shape=traveltime_shape)
    traveltime_index_array = np.memmap(traveltime_index_fname, dtype=np.int32, mode="w+", shape=traveltime_shape)
    traveltime_type_array = np.memmap(traveltime_type_fname, dtype=np.int32, mode="w+", shape=traveltime_shape)
    arrivaltime_index_array = np.memmap(arrivaltime_index_fname, dtype=np.int64, mode="w+", shape=traveltime_shape)
    snr_array = np.memmap(snr_fname, dtype=np.float32, mode="w+", shape=snr_shape)

    with open(f"{root_path}/{result_path}/config.json", "w") as f:
        json.dump(config["cctorch"], f, indent=4, sort_keys=True)

    # %%
    dirs = sorted(glob(f"{root_path}/{region}/waveforms/????-???/??"))
    ncpu = mp.cpu_count()
    lock = mp.Lock()

    # with mp.get_context("spawn").Pool(ncpu) as pool:
    with ThreadPool(ncpu) as pool:
        pool.starmap(
            extract_template_numpy,
            [
                (
                    template_fname,
                    traveltime_fname,
                    traveltime_index_fname,
                    traveltime_type_fname,
                    arrivaltime_index_fname,
                    snr_fname,
                    d,
                    events,
                    stations,
                    picks,
                    config["cctorch"],
                    lock,
                    i,
                )
                for i, d in enumerate(dirs)
            ],
        )


if __name__ == "__main__":
    import json
    import os

    root_path = "local"
    region = "demo"
    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    cut_templates.python_func(root_path, region=region, config=config)

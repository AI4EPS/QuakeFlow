# %%
import json
import multiprocessing as mp
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from glob import glob

import gamma
import numpy as np
import obspy
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# %%
def extract_template_numpy(
    template_fname,
    traveltime_fname,
    traveltime_index_fname,
    traveltime_type_fname,
    snr_fname,
    mseed_path,
    events,
    picks,
    stations,
    config,
    lock,
):
    template_array = np.memmap(template_fname, dtype=np.float32, mode="r+", shape=tuple(config["template_shape"]))
    traveltime_array = np.memmap(traveltime_fname, dtype=np.float32, mode="r+", shape=tuple(config["traveltime_shape"]))
    traveltime_index_array = np.memmap(
        traveltime_index_fname, dtype=np.int32, mode="r+", shape=tuple(config["traveltime_shape"])
    )
    traveltime_type_array = np.memmap(
        traveltime_type_fname, dtype=np.int32, mode="r+", shape=tuple(config["traveltime_shape"])
    )
    snr_array = np.memmap(snr_fname, dtype=np.float32, mode="r+", shape=tuple(config["snr_shape"]))

    # %%
    tmp = mseed_path.split("/")
    year_jday, hour = tmp[-2], tmp[-1]
    begin_time = datetime.strptime(f"{year_jday}T{hour}", "%Y-%jT%H").replace(tzinfo=timezone.utc)
    # TODO: make timedelta a parameter
    end_time = begin_time + timedelta(hours=1)
    picks_ = picks[(picks["phase_time"] >= begin_time) & (picks["phase_time"] < end_time)]

    if len(picks_) == 0:
        return mseed_path

    # %%
    waveforms_dict = {}
    # for station_id in tqdm(
    #     stations["station_id"], desc=f"Loading waveform: ", position=ibar % 6, nrows=7, mininterval=5, leave=True
    # ):
    for station_id in stations["station_id"]:
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
                    trace.filter("bandpass", freqmin=2.0, freqmax=15.0, corners=4, zerophase=True)
                    waveforms_dict[f"{station_id}{c}"] = trace
                except Exception as e:
                    print(e)
                    continue

    # %%
    stations = stations.set_index("station_id")
    for i, pick in picks_.iterrows():
        station_id = pick["station_id"]
        station = stations.loc[station_id]
        event = events.loc[pick["event_index"]]
        empty_data = True
        for c in station["component"]:
            c_idx = config["component_mapping"][c]  # 012 for P, 345 for S

            if f"{station_id}{c}" in waveforms_dict:
                trace = waveforms_dict[f"{station_id}{c}"]
                trace_starttime = trace.stats.starttime.datetime.replace(tzinfo=timezone.utc).timestamp()

                begin_time = pick["phase_timestamp"] - trace_starttime - config["time_before"]
                end_time = pick["phase_timestamp"] - trace_starttime + config["time_after"]
                if begin_time < 0:
                    continue
                begin_time_index = max(0, int(begin_time * trace.stats.sampling_rate))
                end_time_index = max(0, int(end_time * trace.stats.sampling_rate))
                traveltime_array[i, 0, 0] = (
                    begin_time_index / trace.stats.sampling_rate
                    + config["time_before"]
                    + trace_starttime
                    - event["event_timestamp"]
                )  ## define traveltime at the exact data point
                traveltime_index_array[i, 0, 0] = begin_time_index + int(
                    config["time_before"] * trace.stats.sampling_rate
                )
                trace_data = trace.data[begin_time_index:end_time_index].astype(np.float32)

                if len(trace_data) < config["nt"]:
                    continue
                std = np.std(trace_data)
                if std == 0:
                    continue

                empty_data = False
                template_array[i, c_idx, 0, : config["nt"]] = trace_data[: config["nt"]]
                ################## debuging ##################
                # import matplotlib.pyplot as plt
                # import scipy.interpolate

                # if (i == 0) and (j in [3, 4, 5]):
                #     # template_[c_index, j, 1 : config["nt"]] = trace_data[: config["nt"] - 1]
                #     t = np.linspace(0, 1, (config["nt"] - 1) + 1)
                #     t_interp = np.linspace(0, 1, (config["nt"] - 1) * 10 + 1)
                #     x = trace_data[: config["nt"]]
                #     x_interp = scipy.interpolate.interp1d(t, x, kind="cubic")(t_interp)
                #     # print(x - x_interp[0::10])
                #     # plt.figure()
                #     # plt.plot(t, x)
                #     # plt.plot(t_interp, x_interp)
                #     # plt.plot(t, x_interp[0::10])
                #     # plt.savefig("debug.png")
                #     # raise
                #     template_[c_index, j, :] = np.roll(x_interp, -1)[::10]
                ################################################

                s = np.std(trace_data[-int(config["time_after"] * config["sampling_rate"]) :])
                n = np.std(trace_data[: int(config["time_before"] * config["sampling_rate"])])
                if n == 0:
                    snr_array[i, c_idx, 0] = 0
                else:
                    snr_array[i, c_idx, 0] = s / n

        with lock:
            template_array.flush()
            traveltime_array.flush()
            traveltime_index_array.flush()
            traveltime_type_array.flush()
            snr_array.flush()

    return mseed_path


def generate_pairs(picks, mseeds, fname="mseed_pick_pairs.txt"):
    # %%
    # mseeds_dict = {x.split("/")[-1].replace(".mseed", "")[:-1]: x for x in mseeds}
    mseeds_df = pd.DataFrame(mseeds, columns=["fname"])
    mseeds_df["station_id"] = mseeds_df["fname"].apply(lambda x: x.split("/")[-1].replace(".mseed", "")[:-1])
    picks["index"] = picks.index
    picks = picks.set_index("station_id")

    # %%
    with open(fname, "w") as f:
        for i, m in tqdm(mseeds_df.iterrows(), desc="Generating pairs", total=len(mseeds_df)):
            for j, p in picks.loc[m["station_id"]].iterrows():
                f.write(f"{i},{p['index']}\n")

    return fname


if __name__ == "__main__":
    # %%
    root_path = "local"
    region = "demo"
    rank = 0
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    # %%
    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    # %%
    result_path = f"{region}/qtm"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    print(json.dumps(config, indent=4, sort_keys=True))

    # %%
    picks = pd.read_csv(
        # f"{root_path}/{region}/results/phase_association/picks_{rank:03d}.csv",
        f"{root_path}/{region}/results/phase_association/picks.csv",
        parse_dates=["phase_time"],
    )
    picks = picks[picks["event_index"] != -1]
    picks["phase_time"] = picks["phase_time"].apply(lambda x: pd.to_datetime(x, utc=True))
    picks["phase_timestamp"] = picks["phase_time"].apply(lambda x: x.timestamp())
    print(f"{len(picks) = }")
    print(picks.iloc[:5])

    # DEBUG TM
    # picks = picks[
    #     (picks["phase_time"] > pd.to_datetime("2019-07-04T17:00:00", utc=True))
    #     & (picks["phase_time"] < pd.to_datetime("2019-07-04T18:00:00", utc=True))
    # ]
    # picks = picks[(picks["event_index"] == 1253.0) | (picks["event_index"] == 270.0) | (picks["event_index"] == 272.0)]
    # picks = picks[(picks["event_index"] == 1253.0)] # 270
    # picks = picks[(picks["event_index"] == 270.0)]  # 270
    # print(picks)
    # print(f"{len(picks) = }")
    # print(f"{len(picks['event_index'].unique()) = }")
    ################## debuging ##################
    # tmp = picks[picks["event_index"] == 1528]
    # tmp["event_index"] = 1527
    # # tmp["phase_time"] = tmp["phase_time"] + pd.Timedelta(seconds=0.1)
    # # tmp["phase_timestamp"] = tmp["phase_time"].apply(lambda x: x.timestamp())
    # picks = pd.concat([picks[picks["event_index"] != 1527], tmp])
    ################################################

    # %%
    stations = pd.read_json(f"{root_path}/{region}/results/data/stations.json", orient="index")
    stations["station_id"] = stations.index
    # %% filter stations without picks
    stations = stations[stations["station_id"].isin(picks["station_id"].unique())]
    stations.reset_index(drop=True, inplace=True)  # index used in memmap array
    stations.to_json(f"{root_path}/{result_path}/stations.json", orient="index", indent=4)
    stations.to_csv(f"{root_path}/{result_path}/stations.csv", index=True)
    print(f"{len(stations) = }")
    print(stations.iloc[:5])

    # %%
    # events = pd.read_csv(f"{root_path}/{region}/results/phase_association/events_{rank:03d}.csv", parse_dates=["time"])
    events = pd.read_csv(f"{root_path}/{region}/results/phase_association/events.csv", parse_dates=["time"])
    events = events[events["time"].notna()]
    # events.sort_values(by="time", inplace=True)
    events.rename(columns={"time": "event_time"}, inplace=True)
    events["event_time"] = events["event_time"].apply(lambda x: pd.to_datetime(x, utc=True))
    events["event_timestamp"] = events["event_time"].apply(lambda x: x.timestamp())
    if "x(km)" in events.columns:
        events.rename(columns={"x(km)": "x_km"}, inplace=True)
    if "y(km)" in events.columns:
        events.rename(columns={"y(km)": "y_km"}, inplace=True)
    if "z(km)" in events.columns:
        events.rename(columns={"z(km)": "z_km"}, inplace=True)
    events.reset_index(drop=True, inplace=True)  # index used in memmap array
    events.to_csv(f"{root_path}/{result_path}/events.csv", index=True)
    print(f"{len(events) = }")
    print(events.iloc[:5])

    ################## debuging ####################
    # mask = events["event_index"] == 1528
    # events.loc[mask, "event_time"] = events.loc[mask, "event_time"] + pd.Timedelta(seconds=-0.001)
    # events.loc[mask, "event_timestamp"] = events.loc[mask, "event_time"].apply(lambda x: x.timestamp())
    # print(events[events["event_index"] == 1528], events[events["event_index"] == 1527])
    ################################################

    # %%
    event_pair_fname = f"{root_path}/{result_path}/event_pairs.txt"
    # generate_pairs(events, min_pair_dist=config["cctorch"]["min_pair_dist_km"], fname=event_pair_fname)
    config["cctorch"]["event_pair_file"] = event_pair_fname

    # %%
    event_index_fname = f"{root_path}/{result_path}/event_index.txt"
    with open(event_index_fname, "w") as f:
        for i, event in enumerate(events.iloc):
            f.write(f"{i},{event['event_index']}\n")
    config["cctorch"]["event_index_file"] = event_index_fname

    station_index_fname = f"{root_path}/{result_path}/station_index.txt"
    with open(station_index_fname, "w") as f:
        for i, sta in enumerate(stations.iloc):
            f.write(f"{i},{sta['station_id']},{sta['component']}\n")
    config["cctorch"]["station_index_file"] = station_index_fname

    # %%
    picks = picks.merge(stations, on="station_id")
    picks = picks.merge(events, on="event_index", suffixes=("_station", "_event"))
    # picks.set_index("event_index", inplace=True)
    picks["event_index"] = picks["event_index"].astype(int)
    picks["event_phase_station_id"] = (
        picks["event_index"].astype(str) + "/" + picks["phase_type"] + "/" + picks["station_id"]
    )
    picks.to_csv(f"{root_path}/{result_path}/picks.csv", index=False)
    picks["event_phase_station_id"].to_csv(
        f"{root_path}/{result_path}/event_phase_station_id.txt", index=False, header=False
    )

    # %%
    nt = int((config["cctorch"]["time_before"] + config["cctorch"]["time_after"]) * config["cctorch"]["sampling_rate"])
    config["cctorch"]["nt"] = nt
    nch = 3  ## For [P,S] phases and [E,N,Z] components  ## Seperate P and S phases
    npk = len(picks)
    print(f"npk: {npk}, nch: {nch}, nt: {nt}")
    template_shape = (npk, nch, 1, nt)
    traveltime_shape = (npk, 1, 1)
    snr_shape = (npk, nch, 1)
    config["cctorch"]["template_shape"] = template_shape
    config["cctorch"]["traveltime_shape"] = traveltime_shape
    config["cctorch"]["snr_shape"] = snr_shape

    template_fname = f"{root_path}/{result_path}/template.dat"
    traveltime_fname = f"{root_path}/{result_path}/traveltime.dat"
    traveltime_index_fname = f"{root_path}/{result_path}/traveltime_index.dat"
    traveltime_type_fname = f"{root_path}/{result_path}/traveltime_type.dat"
    snr_fname = f"{root_path}/{result_path}/snr.dat"
    config["cctorch"]["template_file"] = template_fname
    config["cctorch"]["traveltime_file"] = traveltime_fname
    config["cctorch"]["traveltime_index_file"] = traveltime_index_fname
    config["cctorch"]["traveltime_type_file"] = traveltime_type_fname
    config["cctorch"]["snr_file"] = snr_fname

    template_array = np.memmap(template_fname, dtype=np.float32, mode="w+", shape=template_shape)
    traveltime_array = np.memmap(traveltime_fname, dtype=np.float32, mode="w+", shape=traveltime_shape)
    traveltime_index_array = np.memmap(traveltime_index_fname, dtype=np.int32, mode="w+", shape=traveltime_shape)
    traveltime_type_array = np.memmap(traveltime_type_fname, dtype=np.int32, mode="w+", shape=traveltime_shape)
    snr_array = np.memmap(snr_fname, dtype=np.float32, mode="w+", shape=snr_shape)

    with open(f"{root_path}/{result_path}/config.json", "w") as f:
        json.dump(config["cctorch"], f, indent=4, sort_keys=True)

    # %%
    dirs = sorted(glob(f"{root_path}/{region}/waveforms/????-???/??"))
    ncpu = min(32, mp.cpu_count())
    print(f"Using {ncpu} cores")
    pbar = tqdm(total=len(dirs), desc="Cutting templates")

    # %%
    # for d in dirs:
    #     extract_template_numpy(
    #         template_fname,
    #         traveltime_fname,
    #         traveltime_index_fname,
    #         traveltime_type_fname,
    #         snr_fname,
    #         d,
    #         events,
    #         picks,
    #         stations,
    #         config["cctorch"],
    #         nullcontext(),
    #     )
    #     pbar.update()

    # raise

    def pbar_update(x):
        """
        x is the return value of extract_template_numpy
        """
        pbar.update()
        pbar.set_description(f"Cutting templates: {'/'.join(x.split('/')[-2:])}")

    ctx = mp.get_context("spawn")
    # ctx = mp.get_context("fork")
    with ctx.Manager() as manager:
        lock = manager.Lock()
        with ctx.Pool(ncpu) as pool:
            for d in dirs:
                pool.apply_async(
                    extract_template_numpy,
                    (
                        template_fname,
                        traveltime_fname,
                        traveltime_index_fname,
                        traveltime_type_fname,
                        snr_fname,
                        d,
                        events,
                        picks,
                        stations,
                        config["cctorch"],
                        lock,
                    ),
                    callback=pbar_update,
                )
            pool.close()
            pool.join()

    pbar.close()

    # # %%
    # mseeds = sorted(glob(f"{root_path}/{region}/waveforms/????-???/??/*.mseed"))
    # mseeds = ["/".join(x.split("/")[-3:]) for x in mseeds]
    # generate_pairs(picks, mseeds, fname=f"{root_path}/{result_path}/mseed_pick_pairs.txt")

# %%

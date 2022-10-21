# %%
import json
import multiprocessing as mp
import os
import threading
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from scipy.cluster.hierarchy import leaves_list, linkage

warnings.filterwarnings("ignore")


# %% [markdown]
# ## Sort events based on locations

# %%
def sort_events(events):

    X = []
    deg2km = 111.1949
    for index, event in events.iterrows():
        # X.append([(event["longitude"])*deg2km*np.cos(np.deg2rad( event["latitude"])), event["latitude"]*deg2km, event["depth_km"]])
        X.append([event["longitude"], event["latitude"]])
    X = np.array(X)

    Z = linkage(X, "ward", optimal_ordering=True)
    leaves_index = leaves_list(Z)

    events_ = events.iloc[leaves_index].reset_index(drop=True)

    return events_


def projection(θ, φ, c0, xyz, degree=True):
    if degree:
        θ = θ / 180.0 * np.pi
        ϕ = ϕ / 180.0 * np.pi
    e = np.array([np.sin(θ) * np.cos(ϕ), np.sin(θ) * np.sin(ϕ), np.cos(θ)])
    e1 = np.array([-np.sin(ϕ), np.cos(ϕ), 0])
    e2 = np.array([-np.cos(θ) * np.cos(ϕ), -np.cos(θ) * np.sin(ϕ), np.sin(θ)])
    # r = np.abs(np.dot((xyz - c0), e))
    r = np.dot((xyz - c0), e)
    x1 = np.dot((xyz - c0), e1)
    y1 = np.dot((xyz - c0), e2)
    return x1, y1, r


def filt_events(events, config, w=2):

    c0 = np.array(
        [(config["minlongitude"] + config["maxlongitude"]) / 2, (config["minlatitude"] + config["maxlatitude"]) / 2, 0]
    )
    deg2km = 111.1949
    xyz = events[["longitude", "latitude", "depth_km"]].values
    xyz[:, 0] = (xyz[:, 0] - c0[0]) * deg2km * np.cos(c0[1] / 180.0 * np.pi)
    xyz[:, 1] = (xyz[:, 1] - c0[1]) * deg2km
    θ = 90
    ϕ = -10
    x1, y1, r = projection(θ, ϕ, [0, 0, 0], xyz)
    r_ = np.abs(r)
    idx = np.argsort(-r_)
    x1_ = x1[idx]
    y1_ = y1[idx]
    r_ = r_[idx]

    return events.iloc[idx[r_ < w]]


# %% [markdown]
# ## Download stations

# %%
def download_stations(config, station_json="stations.json", edge=0.2):

    stations = Client(config["client"]).get_stations(
        network=",".join(config["networks"]),
        station="*",
        starttime=config["starttime"],
        endtime=config["endtime"],
        minlongitude=config["minlongitude"] - edge,
        maxlongitude=config["maxlongitude"] + edge,
        minlatitude=config["minlatitude"] - edge,
        maxlatitude=config["maxlatitude"] + edge,
        channel=config["channels"],
        level="response",
        # filename="stations.xml"
    )

    print("Number of stations: {}".format(sum([len(x) for x in stations])))
    # stations.plot('local', outfile="stations.png")
    #     stations.plot('local')

    ####### Save stations ########
    station_locs = defaultdict(dict)
    for network in stations:
        for station in network:
            for chn in station:
                sid = f"{network.code}.{station.code}.{chn.location_code}.{chn.code[:-1]}"
                if sid in station_locs:
                    if chn.code[-1] not in station_locs[sid]["component"]:
                        station_locs[sid]["component"].append(chn.code[-1])
                        station_locs[sid]["response"].append(round(chn.response.instrument_sensitivity.value, 2))
                else:
                    tmp_dict = {
                        "longitude": chn.longitude,
                        "latitude": chn.latitude,
                        "elevation(m)": chn.elevation,
                        "component": [
                            chn.code[-1],
                        ],
                        "response": [
                            round(chn.response.instrument_sensitivity.value, 2),
                        ],
                        "unit": chn.response.instrument_sensitivity.input_units.lower(),
                    }
                    station_locs[sid] = tmp_dict

    with open(station_json, "w") as fp:
        json.dump(station_locs, fp, indent=2)

    station_locs = pd.DataFrame.from_dict(station_locs, orient="index")
    station_locs["id"] = station_locs.index
    return station_locs


# %% [markdown]
# ## Download waveforms

# %%
def download_waveform(config, events, stations, window_length=30, data_path="output"):

    if isinstance(data_path, str):
        data_path = Path(data_path)
    waveform_dir = data_path / "waveforms"
    if not waveform_dir.exists():
        waveform_dir.mkdir(parents=True)

    ####### Download data ########
    client = Client(config["client"])

    def download(event):

        eventtime = obspy.UTCDateTime(event["time"])
        starttime = eventtime - 5
        endtime = eventtime + config["window_length"] - 5
        fname = "{}.mseed".format(starttime.datetime.isoformat(timespec="milliseconds"))

        if (waveform_dir / fname).exists():
            print(f"{fname} already exists.")
            return

        max_retry = 10
        stream = obspy.Stream()
        print(f"{fname} download starts")
        num_sta = 0

        for index, station in stations.iterrows():

            print(f"********{station['id']}********")
            network, station, location, channel = station["id"].split(".")
            retry = 0
            while retry < max_retry:
                try:
                    tmp = client.get_waveforms(
                        network,
                        station,
                        "*",
                        config["channels"],
                        starttime,
                        endtime,
                    )
                    #  for trace in tmp:
                    #      if trace.stats.sampling_rate != 100:
                    #          print(trace)
                    #          trace = trace.interpolate(100, method="linear")
                    #      trace = trace.detrend("spline", order=2, dspline=5*trace.stats.sampling_rate)
                    #      stream.append(trace)
                    stream += tmp
                    num_sta += len(tmp)
                    break

                except Exception as err:
                    print("Error {}.{}: {}".format(network, station, err))
                    message = "No data available for request."
                    if str(err)[: len(message)] == message:
                        break
                    retry += 1
                    time.sleep(5)
                    continue

            if retry == max_retry:
                print(f"{fname}: MAX {max_retry} retries reached : {network}.{station}")

        if len(stream) > 0:
            stream = stream.merge(fill_value="interpolate")
            stream = stream.trim(starttime + 0.1, endtime - 0.1, pad=True, fill_value=0)
            stream.write(waveform_dir / fname)
            print(f"{fname} download succeeds")

        else:
            print(f"{fname} empty data")

    threads = []
    MAX_THREADS = 3
    # MAX_THREADS = 1

    download(events.iloc[0])

    for index, event in events.iterrows():
        t = threading.Thread(target=download, args=(event,))
        t.start()
        # time.sleep(0.1)
        threads.append(t)
        if index % MAX_THREADS == MAX_THREADS - 1:
            for t in threads:
                t.join()
            threads = []
    for t in threads:
        t.join()

    return


# %% [markdown]
# ## Plot waveforms

# %%
def traveltime(tvel_file, src_depth_km, dist_km, rec_depth_km=0, phases=["P", "p"], shift=0):

    try:
        model = TauPyModel(model=os.path.basename(tvel_file).split(".")[0])
    except:
        obspy.taup.taup_create.build_taup_model(tvel_file)
        model = TauPyModel(model=os.path.basename(tvel_file).split(".")[0])

    km2deg = 1.0 / 111.2

    if not isinstance(model, TauPyModel):
        model = TauPyModel(model=model)

    arrivals = model.get_travel_times(
        source_depth_in_km=src_depth_km + shift,
        distance_in_degree=dist_km * km2deg,
        receiver_depth_in_km=rec_depth_km + shift,
        phase_list=phases,
    )

    arr_times = []
    for arr in arrivals:
        # print(arr)
        arr_times.append(arr.time)

    if len(arr_times) == 0:
        return None
    return min(arr_times)


# %%
# vz_file = "ak135_debug.tvel"
# vz_file = "vz_hawaii2.tvel"
# config["taup_model"] = vz_file
# traveltime(str(input_path / vz_file), 16.524579427374235, -1.429, phases=["P","p"], shift=10)

# %%
def plot_waveforms(
    events,
    stations,
    config,
    waveform_path="waveforms",
    input_path="input",
    output_path="output",
    events_background=None,
):

    vp, vs = 6.0, 6.0 / 1.73
    if isinstance(waveform_path, str):
        waveform_path = Path(waveform_path)
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    normalize = lambda x: (x - x.mean()) / x.std()
    if events_background is None:
        events_background = events

    for i, event in events.iterrows():

        eventtime = obspy.UTCDateTime(event["time"])
        starttime = eventtime - 5
        endtime = eventtime + config["window_length"] - 5
        fname = "{}.mseed".format(starttime.datetime.isoformat(timespec="milliseconds"))
        if not (waveform_path / fname).exists():
            print(f"{waveform_path / fname} does not exists.")
            continue

        stream = obspy.read(waveform_path / fname)
        stream = stream.filter("highpass", freq=1.0)
        stream = stream.trim(starttime + 1.0, endtime - 1.0, pad=True, fill_value=0)

        stations["dist_xy_km"] = stations.apply(
            lambda x: obspy.geodetics.base.gps2dist_azimuth(
                x["latitude"], x["longitude"], event["latitude"], event["longitude"]
            )[0]
            / 1000,
            axis=1,
        )
        stations["dist_xyz_km"] = stations.apply(
            lambda x: np.sqrt(x["dist_xy_km"] ** 2 + (x["elevation(m)"] / 1e3 - event["depth_km"]) ** 2), axis=1
        )
        stations = stations.sort_values(by="dist_xyz_km")

        fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [1, 8]})
        # axes[0].scatter(stations["longitude"], stations["latitude"], s=20, marker="^")
        axes[0].scatter(events_background["longitude"], events_background["latitude"], s=100, marker=".", color="k")
        axes[0].scatter(event["longitude"], event["latitude"], s=100, marker=".", color="red")

        ii = 0
        for j, station in stations.iterrows():
            network, station_, location, channel = station["id"].split(".")
            stream_selected = stream.select(network=network, station=station_, location=location, channel=channel + "Z")
            if len(stream_selected) > 0:
                for trace in stream_selected:
                    t = trace.times(reftime=eventtime)
                    axes[1].plot(t, normalize(trace.data) / 6 + ii, linewidth=0.7, color="k")
                    tp = traveltime(
                        str(input_path / config["taup_model"]),
                        event["depth_km"],
                        station["dist_xy_km"],
                        -station["elevation(m)"] / 1e3,
                        phases=["P", "p", "Pg"],
                        shift=10,
                    )
                    ts = traveltime(
                        str(input_path / config["taup_model"]),
                        event["depth_km"],
                        station["dist_xy_km"],
                        -station["elevation(m)"] / 1e3,
                        phases=["S", "s", "Sg"],
                        shift=10,
                    )
                    if tp is None:
                        tp = station["dist_xyz_km"] / vp
                    if ts is None:
                        ts = station["dist_xyz_km"] / vs
                    axes[1].plot([tp, tp], [ii - 0.7, ii + 0.7], color="blue")
                    axes[1].plot([ts, ts], [ii - 0.7, ii + 0.7], color="red")

                ii += 1

        plt.close(fig)
        fig.savefig(output_path / f"{i:04d}_{starttime.datetime.isoformat(timespec='milliseconds')}.png")
        print(f"Saving {i:04d}_{starttime.datetime.isoformat(timespec='milliseconds')}.png")


# %%
def plot_waveform_by_station(
    events,
    stations,
    config,
    waveform_path="waveforms",
    input_path="input",
    output_path="output",
    events_background=None,
):

    vp, vs = 6.0, 6.0 / 1.73
    if isinstance(waveform_path, str):
        waveform_path = Path(waveform_path)
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    normalize = lambda x: (x - x.mean()) / x.std()
    if events_background is None:
        events_background = events

    for i, station in stations.iterrows():

        network, station_, location, channel = station["id"].split(".")

        # fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [1, 8]})
        # # axes[0].scatter(stations["longitude"], stations["latitude"], s=20, marker="^")
        # axes[0].scatter(events_background["longitude"], events_background["latitude"], s=100, marker=".", color="k")
        # axes[0].scatter(event["longitude"], event["latitude"], s=100, marker=".", color="red")

        fig, ax = plt.subplots(1, 1, figsize=(10, 50))

        events["dist_xy_km"] = events.apply(
            lambda x: obspy.geodetics.base.gps2dist_azimuth(
                station["latitude"], station["longitude"], x["latitude"], x["longitude"]
            )[0]
            / 1000,
            axis=1,
        )
        events["dist_xyz_km"] = events.apply(
            lambda x: np.sqrt(x["dist_xy_km"] ** 2 + (x["depth_km"] / 1e3 - station["elevation(m)"] / 1e3) ** 2),
            axis=1,
        )
        # events = events.sort_values(by="dist_xyz_km").reset_index(drop=True)
        events = events.sort_values(by="latitude").reset_index(drop=True)

        ii = 0
        for j, event in events.iterrows():

            eventtime = obspy.UTCDateTime(event["time"])
            starttime = eventtime - 5
            endtime = eventtime + config["window_length"] - 5
            fname = "{}.mseed".format(starttime.datetime.isoformat(timespec="milliseconds"))

            if not (waveform_path / fname).exists():
                print(f"{waveform_path / fname} does not exists.")
                continue

            stream = obspy.read(waveform_path / fname)
            stream = stream.detrend("linear")
            stream = stream.filter("highpass", freq=2.0)
            stream = stream.trim(starttime + 1.0, endtime - 1.0, pad=True, fill_value=0)

            # stream_selected = stream.select(network=network, station=station_, location=location, channel=channel + "Z")
            stream_selected = stream.select(network=network, station=station_, location=location, channel=channel + "Z")
            if len(stream_selected) > 0:
                for trace in stream_selected:
                    t = trace.times(reftime=eventtime)
                    ax.plot(t, normalize(trace.data) / 6 + ii, linewidth=1.0, color="k")
                    # tp = traveltime(
                    #     str(input_path / config["taup_model"]),
                    #     event["depth_km"],
                    #     event["dist_xy_km"],
                    #     -station["elevation(m)"] / 1e3,
                    #     phases=["P", "p", "Pg"],
                    #     shift=10,
                    # )
                    # ts = traveltime(
                    #     str(input_path / config["taup_model"]),
                    #     event["depth_km"],
                    #     event["dist_xy_km"],
                    #     -station["elevation(m)"] / 1e3,
                    #     phases=["S", "s", "Sg"],
                    #     shift=10,
                    # )
                    # if tp is None:
                    #     tp = event["dist_xyz_km"] / vp
                    # if ts is None:
                    #     ts = event["dist_xyz_km"] / vs
                    # ax.plot([tp, tp], [ii - 0.7, ii + 0.7], color="blue")
                    # ax.plot([ts, ts], [ii - 0.7, ii + 0.7], color="red")

                    ii += 1

            # if j > 100:
            #     break

        ax.set_ylim([-1, ii])
        ax.grid("on")
        ax.autoscale(enable=True, axis="x", tight=True)
        # plt.tight_layout()
        plt.close(fig)
        fig.savefig(output_path / f"{i}.png", bbox_inches="tight", dpi=600)
        print(f"Saving {i}.png")


def save_waveform_by_station(
    events,
    stations,
    config,
    waveform_path="waveforms",
    input_path="input",
    output_path="output",
    events_background=None,
):

    vp, vs = 6.0, 6.0 / 1.73
    if isinstance(waveform_path, str):
        waveform_path = Path(waveform_path)
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    normalize = lambda x: (x - x.mean()) / x.std()
    if events_background is None:
        events_background = events

    for i, station in stations.iterrows():

        network, station_, location, channel = station["id"].split(".")

        events["dist_xy_km"] = events.apply(
            lambda x: obspy.geodetics.base.gps2dist_azimuth(
                station["latitude"], station["longitude"], x["latitude"], x["longitude"]
            )[0]
            / 1000,
            axis=1,
        )
        events["dist_xyz_km"] = events.apply(
            lambda x: np.sqrt(x["dist_xy_km"] ** 2 + (x["depth_km"] / 1e3 - station["elevation(m)"] / 1e3) ** 2),
            axis=1,
        )
        # events = events.sort_values(by="dist_xyz_km").reset_index(drop=True)
        events = events.sort_values(by="latitude").reset_index(drop=True)

        ii = 0
        waveforms = []
        max_nt = 0
        event_index = []
        for j, event in events.iterrows():
            event_index.append(j)

            eventtime = obspy.UTCDateTime(event["time"])
            starttime = eventtime - 5
            endtime = eventtime + config["window_length"] - 5
            fname = "{}.mseed".format(starttime.datetime.isoformat(timespec="milliseconds"))

            if not (waveform_path / fname).exists():
                print(f"{waveform_path / fname} does not exists.")
                continue

            stream = obspy.read(waveform_path / fname)
            stream = stream.detrend("linear")
            stream = stream.filter("highpass", freq=2.0)
            stream = stream.trim(starttime + 1.0, endtime - 1.0, pad=True, fill_value=0)

            stream_selected = stream.select(network=network, station=station_, location=location, channel=channel + "Z")

            if len(stream_selected) > 0:
                waveforms.append(stream_selected[0].data)
                max_nt = max([len(stream_selected[0].data), max_nt])
                ii += 1

        waveforms_ = np.zeros([len(waveforms), max_nt])
        for jj in range(len(waveforms)):
            waveforms_[jj, : len(waveforms[jj])] = waveforms[jj]
        waveforms = np.array(waveforms_)
        # np.save(output_path / f"{i}.npy", waveforms)
        np.savez(output_path / f"{i}.npz", data=waveforms, event_index=event_index)
        print(ii)


# %% [markdown]
# ## Read Growclust catalog
def load_catalog_glowclust(file_path, config):

    if isinstance(file_path, str):
        file_path = Path(file_path)
    # %%
    catalog = pd.read_csv(
        file_path,
        sep="\s+",
        header=None,
        names=[
            "yr",
            "mon",
            "day",
            "hr",
            "min",
            "sec",
            "evid",
            "latR",
            "lonR",
            "depR",
            "mag",
            "qID",
            "cID",
            "nbranch",
            "qnpair",
            "qndiffP",
            "qndiffS",
            "rmsP",
            "rmsS",
            "eh",
            "ez",
            "et",
            "latC",
            "lonC",
            "depC",
        ],
    )
    catalog = catalog[catalog["nbranch"] > 1]

    catalog["latitude"] = catalog["latR"]
    catalog["longitude"] = catalog["lonR"]
    catalog["depth_km"] = catalog["depR"]
    catalog["time"] = catalog.apply(
        lambda x: datetime.fromisoformat(
            f"{x['yr']:04.0f}-{x['mon']:02.0f}-{x['day']:02.0f}T{x['hr']:02.0f}:{x['min']:02.0f}:{min(x['sec'],59.999):06.3f}"
        ),
        axis=1,
    )
    catalog["timestamp"] = catalog.apply(lambda x: x.time.timestamp(), axis=1)
    catalog_selected = catalog[
        (catalog["longitude"] > config["minlongitude"])
        & (catalog["longitude"] < config["maxlongitude"])
        & (catalog["latitude"] > config["minlatitude"])
        & (catalog["latitude"] < config["maxlatitude"])
    ]

    return catalog_selected


# %%
if __name__ == "__main__":

    # %%
    center = (-155.32, 19.39)
    client = "IRIS"
    # minlongitude, maxlongitude, minlatitude, maxlatitude = (-155.55, -155.45, 19.52, 19.8)
    # minlongitude, maxlongitude, minlatitude, maxlatitude = (-155.57, -155.44, 19.50, 19.82)
    # minlongitude, maxlongitude, minlatitude, maxlatitude = (-155.57, -155.44, 19.50, 19.9)
    # minlongitude, maxlongitude, minlatitude, maxlatitude = (-155.57, -155.44, 19.50, 19.825)
    minlongitude, maxlongitude, minlatitude, maxlatitude = (-155.57, -155.44, 19.50, 19.83)
    network_list = ["HV", "PT"]
    channel_list = "HH*,BH*,EH*,HN*"
    starttime = obspy.UTCDateTime("2018-01-01T00")
    endtime = obspy.UTCDateTime("2022-08-12T00")

    config = {}
    config["client"] = client
    config["minlongitude"] = minlongitude
    config["maxlongitude"] = maxlongitude
    config["minlatitude"] = minlatitude
    config["maxlatitude"] = maxlatitude
    config["networks"] = network_list
    config["channels"] = channel_list
    config["starttime"] = starttime.datetime.isoformat(timespec="milliseconds")
    config["endtime"] = endtime.datetime.isoformat(timespec="milliseconds")
    config["window_length"] = 30
    config["taup_model"] = "vz_hawaii2.tvel"

    # %%
    input_path = Path("input")
    output_path = Path("output")
    if not output_path.exists():
        output_path.mkdir()

    catalog = load_catalog_glowclust(input_path / "catalogs/out.growclust_cat", config)

    # %%
    events = catalog[["time", "latitude", "longitude", "depth_km"]]
    print(f"Selected {len(events)} events.")

    events.to_csv(output_path / "selected_events.csv")

    plt.figure()
    plt.scatter(events["longitude"], events["latitude"], c=events["depth_km"], s=5.0)
    plt.axis("scaled")
    plt.tight_layout()
    plt.savefig(output_path / "events.png")
    plt.show()

    # %%
    events = filt_events(events, config, w=5)
    print(f"After filtering {len(events)} events.")

    events.to_csv(output_path / "filt_events.csv")

    plt.figure()
    plt.scatter(events["longitude"], events["latitude"], c=events["depth_km"], s=5.0)
    plt.axis("scaled")
    plt.tight_layout()
    plt.savefig(output_path / "filt_events.png")
    plt.show()

    # %%
    # events = sort_events(events)

    # %%
    stations = download_stations(config, output_path / "stations.json", edge=0.15)

    plt.figure()
    plt.scatter(stations["longitude"], stations["latitude"], s=10, marker="^")
    plt.axis("scaled")
    plt.tight_layout()
    plt.savefig(output_path / "stations.png")
    plt.show()

    # %%
    download_waveform(config, events, stations, window_length=30, data_path=output_path)

    # %%
    # plot_waveforms(events, stations, config, output_path/"waveforms", output_path/"figures")

    # ctx = mp.get_context("spawn")
    # ncpu = mp.cpu_count()
    # print("Number of CPUs: {}".format(ncpu))
    # processes = []
    # for i in range(ncpu):
    #     p = ctx.Process(
    #         target=plot_waveforms,
    #         args=(
    #             events.iloc[i::ncpu],
    #             stations,
    #             config,
    #             output_path / "waveforms",
    #             input_path,
    #             output_path / "figures",
    #             events,
    #         ),
    #     )
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    # plot_waveform_by_station(
    #     events, stations, config, output_path / "waveforms", input_path, output_path / "figures_by_station"
    # )

    # ctx = mp.get_context("spawn")
    # ncpu = mp.cpu_count()
    # print("Number of CPUs: {}".format(ncpu))
    # processes = []
    # for i in range(ncpu):
    #     p = ctx.Process(
    #         target=plot_waveform_by_station,
    #         args=(
    #             events,
    #             stations.iloc[i::ncpu],
    #             config,
    #             output_path / "waveforms",
    #             input_path,
    #             output_path / "figures_by_station",
    #             events,
    #         ),
    #     )
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    # save_waveform_by_station(
    #     events, stations, config, output_path / "waveforms", input_path, output_path / "waveforms_by_station"
    # )
    ctx = mp.get_context("spawn")
    ncpu = mp.cpu_count()
    print("Number of CPUs: {}".format(ncpu))
    processes = []
    for i in range(ncpu):
        p = ctx.Process(
            target=save_waveform_by_station,
            args=(
                events,
                stations.iloc[i::ncpu],
                config,
                output_path / "waveforms",
                input_path,
                output_path / "waveforms_by_station",
                events,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

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

from obspy.signal.rotate import rotate2zne
from obspy import Inventory
from obspy import read_inventory

# warnings.filterwarnings("error")
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# %%
protocol = "gs"
token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
# token = "cloud"
bucket = "quakeflow_dataset"
fs = fsspec.filesystem(protocol=protocol, token=token)

# root_path = "dataset"
region = "NC"
root_path = f"{bucket}/{region}"
mseed_path = f"{root_path}/waveform_mseed2"
catalog_path = f"{root_path}/catalog"
#station_path = f"{root_path}/station"
station_path = f"{root_path}/FDSNstationXML"
xml_files = fs.glob(f"{station_path}/*.info/*.FDSN.xml/*.*.xml")
inventory = Inventory()
print("reading stationxml files to create an channel-level inventory")
print("it takes around 15 minutes")
for xml_file in tqdm(xml_files):
    with fs.open(xml_file, 'rb') as f:
        inventory.extend(read_inventory(f, level='channel'))
result_path = f"waveform_h5"
if not os.path.exists(result_path):
    os.makedirs(result_path)

sampling_rate = 100
NT = 12000  # 120 s

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
    phase_picking_channel = []
    event_id = []
    for idx, pick in picks.sort_values("phase_time").iterrows():
        phase_type.append(pick.phase_type)
        phase_index.append(int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate)))
        phase_score.append(pick.phase_score)
        phase_time.append(pick.phase_time.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        phase_remark.append(pick.phase_remark)
        phase_polarity.append(pick.phase_polarity)
        phase_picking_channel.append(pick.instrument+pick.component)
        event_id.append(pick.event_id)

    return phase_type, phase_index, phase_score, phase_time, phase_remark, phase_polarity, phase_picking_channel, event_id

# %%
def flip_polarity(phase_polarity, pha_channel_dips):
    pol_out = []
    for pol, cha_dip in zip(phase_polarity, pha_channel_dips):
        if pol == 'U' or pol == '+':
            if cha_dip == -90:
                pol_out.append('U')
            elif cha_dip == 90:
                pol_out.append('D')
            else:
                pol_out.append('N')
        elif pol == 'D' or pol == '-':
            if cha_dip == -90:
                pol_out.append('D')
            elif cha_dip == 90:
                pol_out.append('U')
            else:
                pol_out.append('N')
        else:
            pol_out.append('N')
    return pol_out
    
# Refer to: https://github.com/ltauxe/Python-for-Earth-Science-Students/blob/master/Lecture_22.ipynb
def dir2cart(Dir):
    """
    converts polar directions to cartesian coordinates
    Parameters: 
        Dir[Azimuth,Plunge]:  directions in degreess
    Returns:
        [X,Y,Z]: cartesian coordinates
    """
    Az,Pl=np.radians(Dir[0]),np.radians(Dir[1])
    return np.array([np.cos(Az)*np.cos(Pl),np.sin(Az)*np.cos(Pl),np.sin(Pl)])
def cart2dir(X):
    """
    converts cartesian coordinates to polar azimuth and plunge
    Parameters:
        X: list of X,Y,Z coordinates
    Returns: 
        [Az,Pl]: list of polar coordinates in degrees
    """
    R=np.sqrt(X[0]**2+X[1]**2+X[2]**2) # calculate resultant vector length
    Az=np.degrees(np.arctan2(X[1],X[0]))%360. # calculate declination taking care of correct quadrants (arctan2) and making modulo 360.
    Pl=np.degrees(np.arcsin(X[2]/R)) # calculate inclination (converting to degrees) #
    return [Az,Pl]
    
# %%
def convert(i, year):
    # %%
    fs_ = fsspec.filesystem(protocol=protocol, token=token)
    
    # %%
    with h5py.File(f"{result_path}/{year}.h5", "w") as fp, open(f"{result_path}/{year}.skipped.csv", "w") as cw:
        jdays = sorted(fs_.ls(f"{mseed_path}/{year}"), reverse=False)
        jdays = [x.split("/")[-1] for x in jdays][0:1]
        for jday in tqdm(jdays, total=len(jdays), desc=f"{year}", position=i, leave=True):
            tmp = datetime.strptime(jday, "%Y.%j")

            with fs_.open(f"{catalog_path}/{tmp.year:04d}.{tmp.month:02d}.event.csv", "rb") as f:
                events = pd.read_csv(f, parse_dates=["time"], date_format="%Y-%m-%dT%H:%M:%S.%f")
            events["time"] = pd.to_datetime(events["time"])
            events.set_index("event_id", inplace=True)
            with fs_.open(f"{catalog_path}/{tmp.year:04d}.{tmp.month:02d}.phase.csv", "rb") as f:
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

            event_ids = sorted(fs_.ls(f"{mseed_path}/{year}/{jday}"), reverse=True)
            event_ids = [x.split("/")[-1] for x in event_ids][0:10]
            for tmp_event_id in event_ids:
                event_id, tmp_time_string = tmp_event_id.split('_')
                if event_id not in events.index:
                    continue

                gp = fp.create_group(event_id)
                gp.attrs["event_id"] = event_id
                gp.attrs["event_time"] = events.loc[event_id, "time"].strftime("%Y-%m-%dT%H:%M:%S.%f")
                gp.attrs["latitude"] = events.loc[event_id, "latitude"]
                gp.attrs["longitude"] = events.loc[event_id, "longitude"]
                gp.attrs["depth_km"] = events.loc[event_id, "depth_km"]
                gp.attrs["magnitude"] = events.loc[event_id, "magnitude"]
                gp.attrs["magnitude_type"] = events.loc[event_id, "magnitude_type"]
                gp.attrs["source"] = "NC"

                mseed_list = sorted(list(fs_.glob(f"{mseed_path}/{year}/{jday}/{event_id}_{tmp_time_string}/*.mseed")))
                st = obspy.Stream()
                for file in mseed_list:
                    with fs_.open(file, "rb") as f:
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
                gp.attrs["nt"] = NT  # default 120s
                gp.attrs["nx"] = len(mseed_list)
                gp.attrs["delta"] = 1 / sampling_rate

                has_station = False
                station_channel_ids = [x.split("/")[-1].replace(".mseed", "") for x in mseed_list]
                for station_channel_id in station_channel_ids:
                    # select traces
                    network, station, location, instrument = station_channel_id.split(".")
                    tr = st.select(id=station_channel_id + "?")
                    # create dataset
                    ds = gp.create_dataset(station_channel_id, (3, gp.attrs["nt"]), dtype=np.float32)
                    # resampe traces
                    for t in tr:
                        if t.stats.sampling_rate != sampling_rate:
                            t.resample(sampling_rate)
                    # extract data from traces
                    for i, t in enumerate(tr):
                        index0 = int(
                            round(
                                (t.stats.starttime.datetime.replace(tzinfo=timezone.utc) - begin_time).total_seconds()
                                * sampling_rate
                            )
                        )
                        if index0 > 3000:
                            del fp[f"{event_id}/{station_channel_id}"]
                            cw.writelines(f"{event_id},{station_channel_id},data_coverage_reason\n")
                            break

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

                    if index0 > 3000:
                        continue
                    # get channel orientations
                    orientations = []
                    for t in tr:
                        try:
                            orientations.append(inventory.get_channel_metadata(t.id, arrival_time))
                        except:
                            pass
                    # if channel metadata is missing or no data
                    if len(orientations) != len(tr) or len(tr) == 0:
                        del fp[f"{event_id}/{station_channel_id}"]
                        cw.writelines(f"{event_id},{station_channel_id},metadata_not_found_reason\n")
                        continue
                    # rotate to ENZ directions
                    # if 3 components available
                    if len(orientations) == 3:
                        azimuth1, dip1 = orientations[0]['azimuth'], orientations[0]['dip']
                        azimuth2, dip2 = orientations[1]['azimuth'], orientations[1]['dip']
                        azimuth3, dip3 = orientations[2]['azimuth'], orientations[2]['dip']
                        vec1, vec2, vec3 = dir2cart([azimuth1, dip1]), dir2cart([azimuth2, dip2]), dir2cart([azimuth3, dip3])
                    # if 2 components available
                    if len(orientations) == 2:
                        azimuth1, dip1 = orientations[0]['azimuth'], orientations[0]['dip']
                        azimuth2, dip2 = orientations[1]['azimuth'], orientations[1]['dip']
                        vec1 = dir2cart([azimuth1, dip1])
                        vec2 = dir2cart([azimuth2, dip2])
                        vec3 = np.cross(vec1, vec2)
                        azimuth3, dip3 = cart2dir(vec3)
                    # if 1 component available
                    if len(orientations) == 1:
                        azimuth1, dip1 = orientations[0]['azimuth'], orientations[0]['dip']
                        vec1 = dir2cart([azimuth1, dip1])
                        vec2 = np.random.randn(3)
                        vec2 -= vec2.dot(vec1) * vec1
                        vec2 /= np.linalg.norm(vec2)
                        azimuth2, dip2 = cart2dir(vec2)
                        vec3 = np.cross(vec1, vec2)
                        azimuth3, dip3 = cart2dir(vec3)
                    # E->[90, 0]
                    # N->[0, 0]
                    # Z->[0, -90]
                    nearzero = 1e-4
                    if abs(np.dot(vec1, vec2)) > nearzero or abs(np.dot(vec1, vec3)) > nearzero or abs(np.dot(vec2, vec3)) > nearzero:
                        del fp[f"{event_id}/{station_channel_id}"]
                        cw.writelines(f"{event_id},{station_channel_id},channel_not_orthogonal_reason\n")
                        continue
                    znewaveforms = rotate2zne(ds[0, :], azimuth1, dip1,
                                            ds[1, :], azimuth2, dip2,
                                            ds[2, :], azimuth3, dip3)
                    ds[:] = np.array(znewaveforms)[::-1]
                    ds.attrs["component"] = 'ENZ'
                    # assign zeros if one component has very low values
                    asmallvalue = np.max(np.abs(ds[:])) / 1e10
                    for i in range(3):
                        if np.max(np.abs(ds[i, :])) < asmallvalue:
                            ds[i, :] = 0
                    # other attibutes
                    ds.attrs["network"] = network
                    ds.attrs["station"] = station
                    ds.attrs["location"] = location
                    ds.attrs["instrument"] = instrument
                    ds.attrs["unit"] = "1e-6m/s" if instrument[-1] != "N" else "1e-6m/s**2"
                    ds.attrs["dt_s"] = 0.01
                    # at least one channel is available
                    ds.attrs["longitude"] = orientations[0]["longitude"]
                    ds.attrs["latitude"] = orientations[0]["latitude"]
                    ds.attrs["elevation_m"] = orientations[0]["elevation"]
                    ds.attrs["local_depth_m"] = orientations[0]["local_depth"]
                    ds.attrs["depth_km"] = round(-0.001*(ds.attrs["elevation_m"]-ds.attrs["local_depth_m"]), 4)
                    station_id = f"{network}.{station}.{location}"
                    if station_id not in phases_by_station.index:
                        del fp[f"{event_id}/{station_channel_id}"]
                        cw.writelines(f"{event_id},{station_channel_id},no_phase_found_reason1\n")
                        continue
                    picks_ = phases_by_station.loc[[station_id]]
                    picks_ = picks_[(picks_["phase_time"] > begin_time) & (picks_["phase_time"] < end_time)]
                    if len(picks_[picks_["event_id"] == event_id]) == 0:
                        del fp[f"{event_id}/{station_channel_id}"]
                        cw.writelines(f"{event_id},{station_channel_id},no_phase_found_reason2\n")
                        continue

                    pick = picks_[picks_["event_id"] == event_id].iloc[0]  # after sort_value
                    ds.attrs["azimuth"] = pick.azimuth
                    ds.attrs["distance_km"] = pick.distance_km
                    ds.attrs["takeoff_angle"] = pick.takeoff_angle

                    tmp = int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate))
                    if (tmp - 300 < 0) or (tmp + 300 >= NT):
                        del fp[f"{event_id}/{station_channel_id}"]
                        cw.writelines(f"{event_id},{station_channel_id},snr_timewindow_reason\n")
                        continue

                    snr = calc_snr(ds[:, :], int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate)))
                    if max(snr) == 0:
                        del fp[f"{event_id}/{station_channel_id}"]
                        cw.writelines(f"{event_id},{station_channel_id},zero_snr_reason\n")
                        continue
                    ds.attrs["snr"] = snr

                    (
                        phase_type,
                        phase_index,
                        phase_score,
                        phase_time,
                        phase_remark,
                        phase_polarity,
                        phase_picking_channel, 
                        phase_event_id,
                    ) = extract_pick(picks_, begin_time, sampling_rate)
                    # flip the P polarity if the vertical channel is reversed
                    phase_picking_channel_x = [".".join([station_id, x]) for x in phase_picking_channel]
                    pha_channel_dips = []
                    for x in phase_picking_channel_x:
                        try:
                            pha_channel_dips.append(inventory.get_channel_metadata(x, arrival_time)['dip'])
                        except:
                            pha_channel_dips.append('none')
                    phase_polarity = flip_polarity(phase_polarity, pha_channel_dips)
                    # 
                    ds.attrs["phase_type"] = phase_type
                    ds.attrs["phase_index"] = phase_index
                    ds.attrs["phase_score"] = phase_score
                    ds.attrs["phase_time"] = phase_time
                    ds.attrs["phase_remark"] = phase_remark
                    ds.attrs["phase_polarity"] = phase_polarity
                    ds.attrs["phase_picking_channel"] = phase_picking_channel
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

                    has_station = True

                if not has_station:
                    print(f"{event_id} has no station")
                    del fp[event_id]


if __name__ == "__main__":
    # %%
    years = sorted(fs.ls(mseed_path), reverse=False)
    years = [x.split("/")[-1] for x in years]

    ncpu = len(years)
    ctx = mp.get_context("spawn")
    # ctx = mp.get_context("fork")
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

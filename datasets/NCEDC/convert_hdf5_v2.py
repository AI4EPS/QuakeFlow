# %%
import logging
import multiprocessing as mp
import os
import threading
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path

import fsspec
import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import obspy.geodetics.base
from obspy.signal.rotate import rotate2zne
from tqdm import tqdm

with open("convert_hdf5.log", "w") as f:
    f.write("")
logging.basicConfig(
    filename="convert_hdf5.log", level=logging.INFO, filemode="a", format="%(asctime)s - %(levelname)s - %(message)s"
)

os.environ["OPENBLAS_NUM_THREADS"] = "2"
# raise warnings as exceptions
warnings.filterwarnings("error")

# %%
protocol = "gs"
token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
# token = "cloud"
bucket = "quakeflow_dataset"
fs = fsspec.filesystem(protocol=protocol, token=token)

# root_path = "dataset"
region = "NC"
root_path = f"{bucket}/{region}"
mseed_path = f"{root_path}/waveform_mseed"
catalog_path = f"{root_path}/catalog"
station_path = f"FDSNstationXML"

# %%
result_path = f"waveform_h5"
if not os.path.exists(result_path):
    os.makedirs(result_path)

sampling_rate = 100
NT = 120*sampling_rate  # 120 s


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
def extract_pick(picks, begin_time, sampling_rate, main_event_id):
    # FIXME: check if the arrival time is earlier than the event origin time
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
        phase_picking_channel.append(pick.instrument + pick.component)
        event_id.append(pick.event_id)
    
    phase_type = np.array(phase_type)
    phase_index = np.array(phase_index)
    phase_score = np.array(phase_score)
    phase_time = np.array(phase_time)
    phase_polarity = np.array(phase_polarity)
    phase_remark = np.array(phase_remark)
    phase_picking_channel = np.array(phase_picking_channel)
    event_id = np.array(event_id)
    # fix the duplicate picks
    uniqued_same_phase, idx_same_phase, counts = np.unique(np.array([phase_type, event_id]), axis=1, return_index=True, return_counts=True) # for this case, keep the first one
    if uniqued_same_phase.shape[1] != len(phase_index):
        logging.warning(f"{event_id[idx_same_phase[counts>1][0]]}/{picks.iloc[0].name+picks.iloc[0].location+'.'+picks.iloc[0].instrument} has duplicate picks for the same phase in the same event")
        idxs = np.sort(idx_same_phase)
        phase_type = phase_type[idxs]
        phase_index = phase_index[idxs]
        phase_score = phase_score[idxs]
        phase_time = phase_time[idxs]
        phase_remark = phase_remark[idxs]
        phase_polarity = phase_polarity[idxs]
        phase_picking_channel = phase_picking_channel[idxs]
        event_id = event_id[idxs]
        
    bad_e_ids = []
    for event_idxs, ee_id in zip([event_id == e_id for e_id in np.unique(event_id)], np.unique(event_id)):
        assert len(phase_type[event_idxs]) in [1, 2], f"{event_id}/{picks.iloc[0].name+picks.iloc[0].location+'.'+picks.iloc[0].instrument} has {len(phase_type[event_idxs])} picks, should be 1 or 2"
        pick_types = phase_type[event_idxs]
        if len(pick_types) == 1:
            continue
        p_index = phase_index[event_idxs][pick_types == "P"]
        s_index = phase_index[event_idxs][pick_types == "S"]
        if s_index - p_index <= 5: # if S pick is earlier than P pick
            bad_e_ids.append(ee_id)
    if len(bad_e_ids) > 0:
        logging.warning(f"{picks.iloc[0].name+picks.iloc[0].location+'.'+picks.iloc[0].instrument}/{bad_e_ids} has P pick later than S pick")
        idxs = np.array([e_id not in bad_e_ids for e_id in event_id])
        phase_type = phase_type[idxs]
        phase_index = phase_index[idxs]
        phase_score = phase_score[idxs]
        phase_time = phase_time[idxs]
        phase_remark = phase_remark[idxs]
        phase_polarity = phase_polarity[idxs]
        phase_picking_channel = phase_picking_channel[idxs]
        event_id = event_id[idxs]
    
    # detect too close pick indexs between different events, keep the main one or the first one
    p_indexs = phase_index[phase_type == "P"]
    bad_e_ids = []
    ee_ids = event_id[phase_type == "P"]
    too_close_p_pairs = np.where(np.abs(p_indexs[:, None] - p_indexs) <= 5)
    # filter the same event
    too_close_p_pairs = np.unique(np.sort(
        np.array([too_close_p_pairs[0][too_close_p_pairs[0]!=too_close_p_pairs[1]], too_close_p_pairs[1][too_close_p_pairs[0]!=too_close_p_pairs[1]]]), 
        axis=0), axis=1).T
    for idx1, idx2 in too_close_p_pairs:
        if ee_ids[idx1] == ee_ids[idx2]:
            assert idx1 != idx2, f"{ee_ids[idx1]} has the same index {idx1} with itself"
        if ee_ids[idx1] == main_event_id:
            bad_e_ids.append(ee_ids[idx2])
        elif ee_ids[idx2] == main_event_id:
            bad_e_ids.append(ee_ids[idx1])
        else:
            bad_e_ids.append(ee_ids[idx2])
    if len(bad_e_ids) > 0:
        logging.warning(f"{picks.iloc[0].name+picks.iloc[0].location+'.'+picks.iloc[0].instrument}/{bad_e_ids} has too close P picks")
        idxs = np.array([e_id not in bad_e_ids for e_id in event_id])
        phase_type = phase_type[idxs]
        phase_index = phase_index[idxs]
        phase_score = phase_score[idxs]
        phase_time = phase_time[idxs]
        phase_remark = phase_remark[idxs]
        phase_polarity = phase_polarity[idxs]
        phase_picking_channel = phase_picking_channel[idxs]
        event_id = event_id[idxs]

    return (
        phase_type.tolist(),
        phase_index.tolist(),
        phase_score.tolist(),
        phase_time.tolist(),
        phase_remark.tolist(),
        phase_polarity.tolist(),
        phase_picking_channel.tolist(),
        event_id.tolist(),
    )


# %%
def flip_polarity(phase_polarity, channel_dip):
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


# Refer to: https://github.com/ltauxe/Python-for-Earth-Science-Students/blob/master/Lecture_22.ipynb
def dir2cart(Dir):
    """
    converts polar directions to cartesian coordinates
    Parameters:
        Dir[Azimuth,Plunge]:  directions in degreess
    Returns:
        [X,Y,Z]: cartesian coordinates
    """
    Az, Pl = np.radians(Dir[0]), np.radians(Dir[1])
    return np.array([np.cos(Az) * np.cos(Pl), np.sin(Az) * np.cos(Pl), np.sin(Pl)])


def cart2dir(X):
    """
    converts cartesian coordinates to polar azimuth and plunge
    Parameters:
        X: list of X,Y,Z coordinates
    Returns:
        [Az,Pl]: list of polar coordinates in degrees
    """
    R = np.sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)  # calculate resultant vector length
    Az = (
        np.degrees(np.arctan2(X[1], X[0])) % 360.0
    )  # calculate declination taking care of correct quadrants (arctan2) and making modulo 360.
    Pl = np.degrees(np.arcsin(X[2] / R))  # calculate inclination (converting to degrees) #
    return [Az, Pl]

# %%
def convert_jday(jday, catalog_path, result_path, protocol, token):
    inv_dict = {}
    fs_ = fsspec.filesystem(protocol=protocol, token=token)
    ## NCEDC
    tmp = datetime.strptime(jday, "%Y.%j")
    year, month, day = f"{tmp.year:04d}", f"{tmp.month:02d}", f"{tmp.day:02d}"
    year, dayofyear = jday.split(".")
    if not os.path.exists(f"{result_path}/{year}"):
        os.makedirs(f"{result_path}/{year}", exist_ok=True)
    with h5py.File(f"{result_path}/{year}/{dayofyear}.h5", "w") as fp:

        ## NCEDC
        with fs.open(f"{catalog_path}/{year}.{month}.event.csv", "rb") as f:
        ## SCEDC
        # with fs_.open(f"{catalog_path}/{year}/{year}_{dayofyear}.event.csv", "rb") as f:
            events = pd.read_csv(f, parse_dates=["time"], date_format="%Y-%m-%dT%H:%M:%S.%f%z")
        events["time"] = pd.to_datetime(events["time"])
        events.set_index("event_id", inplace=True)

        ## NCEDC
        with fs.open(f"{catalog_path}/{year}.{month}.phase.csv", "rb") as f:
        ## SCEDC
        # with fs_.open(f"{catalog_path}/{year}/{year}_{dayofyear}.phase.csv", "rb") as f:
            phases = pd.read_csv(
                    f,
                    parse_dates=["phase_time"],
                    date_format="%Y-%m-%dT%H:%M:%S.%f%z",
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

        event_ids = sorted(fs_.ls(f"{mseed_path}/{year}/{jday}"), reverse=False)
        event_fnames = [x.split("/")[-1] for x in event_ids]
        event_ids = [x.split("/")[-1].split("_")[0] for x in event_ids]
        # pbar = tqdm(event_ids, total=len(event_ids), desc=f"{year}/{dayofyear}", leave=True, position=int(dayofyear))
        for event_id, event_fname in zip(event_ids, event_fnames):
            # pbar.update(1)

            if event_id not in events.index or event_id not in phases.index:
                continue

            if event_id in fp:
                logging.warning(f"Duplicate {event_id}: {event_fname}")
                continue

            gp = fp.create_group(event_id)
            # event info
            gp.attrs["event_id"] = event_id
            gp.attrs["event_time"] = events.loc[event_id, "time"].strftime("%Y-%m-%dT%H:%M:%S.%f")
            gp.attrs["latitude"] = events.loc[event_id, "latitude"]
            gp.attrs["longitude"] = events.loc[event_id, "longitude"]
            gp.attrs["depth_km"] = events.loc[event_id, "depth_km"]
            gp.attrs["magnitude"] = events.loc[event_id, "magnitude"]
            gp.attrs["magnitude_type"] = events.loc[event_id, "magnitude_type"]
            gp.attrs["source"] = region

            # waveform info
            mseed_list = sorted(list(fs_.glob(f"{mseed_path}/{year}/{jday}/{event_fname}/*.mseed")))
            arrival_time = phases.loc[event_id, "phase_time"].min()
            begin_time = arrival_time - pd.Timedelta(seconds=30)
            end_time = arrival_time + pd.Timedelta(seconds=90)
            gp.attrs["begin_time"] = begin_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            gp.attrs["end_time"] = end_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            gp.attrs["event_time_index"] = int(
                    round((events.loc[event_id, "time"] - begin_time).total_seconds() * sampling_rate)
                )
            gp.attrs["sampling_rate"] = sampling_rate
            gp.attrs["nt"] = NT  # default 120s
            #gp.attrs["nx"] = len(mseed_list)

            # read mseed
            num_stations = 0
            has_station = False
            for mseed in mseed_list:
                # pbar.set_description(f"{year}/{dayofyear}/{event_id}/{mseed.split('/')[-1]}")

                with fs_.open(mseed, "rb") as f:
                    st = obspy.read(f)
                    # st.detrend("constant")
                    # st.merge(fill_value=0)
                if st[0].stats.sampling_rate != sampling_rate:
                    st.resample(sampling_rate)
                st.sort()
                components = "".join([tr.stats.channel[-1] for tr in st])
                if len(st) > 3:
                    logging.warning(f"{event_id}/{mseed} has {len(st)} components: {components}")
                    continue

                array = np.zeros((3, NT))
                for i, t in enumerate(st):
                    index0 = int(
                            round(
                                (t.stats.starttime.datetime.replace(tzinfo=timezone.utc) - begin_time).total_seconds()
                                * sampling_rate
                            )
                        )
                    if index0 > 30*sampling_rate:
                        logging.warning(f"{event_id}/{mseed} has index0 > {30*sampling_rate}")
                        break

                    if index0 > 0:
                        i_trace = 0
                        i_array = index0
                        ll = min(len(t.data), len(array[i, i_array:]))  # data length
                    elif index0 < 0:
                        i_trace = -index0
                        i_array = 0
                        ll = min(len(t.data[i_trace:]), len(array[i, :]))
                    else:
                        i_trace = 0
                        i_array = 0
                        ll = min(len(t.data), len(array[i, :]))
                    array[i, i_array : i_array + ll] = t.data[i_trace : i_trace + ll] * 1e6  # convert to 1e-6m/s

                if index0 > 30*sampling_rate:
                    continue

                station_channel_id = mseed.split("/")[-1].replace(".mseed", "")
                network, station, location, instrument = station_channel_id.split(".")

                if not os.path.exists(f"{station_path}/{network}/{network}.{station}.xml"):
                ## NCEDC
                    if fs_.exists(
                        f"{root_path}/FDSNstationXML/{network}.info/{network}.FDSN.xml/{network}.{station}.xml"
                    ):

                        fs_.get(
                            f"{root_path}/FDSNstationXML/{network}.info/{network}.FDSN.xml/{network}.{station}.xml",
                            f"{station_path}/{network}/{network}_{station}.xml",
                        )
                ## SCEDC
                #     if fs_.exists(f"{root_path}/FDSNstationXML/{network}/{network}_{station}.xml"):
                #         fs_.get(
                #                 f"{root_path}/FDSNstationXML/{network}/{network}_{station}.xml",
                #                 f"{station_path}/{network}/{network}_{station}.xml",
                #             )
                #     elif fs_.exists(f"{root_path}/FDSNstationXML/unauthoritative-XML/{network}_{station}.xml"):
                #         fs_.get(
                #                 f"{root_path}/FDSNstationXML/unauthoritative-XML/{network}_{station}.xml",
                #                 f"{station_path}/{network}/{network}_{station}.xml",
                #             )
                    else:
                        logging.warning(
                                f"{event_id}/{station_channel_id} has no station metadata: {station_path}/{network}/{network}_{station}.xml"
                            )
                        continue
                
                if f"{network}.{station}" not in inv_dict:
                    try:
                        inv = obspy.read_inventory(f"{station_path}/{network}/{network}.{station}.xml")
                        inv_dict[f"{network}.{station}"] = inv
                    except Exception as e:
                        try:
                            ## NCEDC
                            fs_.get(
                                f"{root_path}/FDSNstationXML/{network}.info/{network}.FDSN.xml/{network}.{station}.xml",
                                f"{station_path}/{network}/{network}.{station}.xml",
                            )
                            ## SCEDC
                            # fs_.get(
                            #         f"{root_path}/FDSNstationXML/{network}/{network}_{station}.xml",
                            #         f"{station_path}/{network}/{network}_{station}.xml",
                            #     )
                        except:
                            logging.error(f"{event_id}/{station_channel_id} has invalid station metadata: {e}")
                        continue
                else:
                    inv = inv_dict[f"{network}.{station}"]
                inv = inv.select(starttime=obspy.UTCDateTime(begin_time))

                # get channel orientations
                try:
                    orientations = [inv.get_channel_metadata(tr.id, tr.stats.starttime) for tr in st]
                except:
                    logging.error(
                            f"{event_id}/{station_channel_id} {[(tr.id, tr.stats.starttime) for tr in st]} has invalid channel metadata"
                        )
                # if channel metadata is missing or no data
                if len(orientations) != len(st) or len(orientations) == 0:
                    logging.error(f"{event_id}/{station_channel_id} has no channel metadata or no data")
                    continue
                # rotate to ENZ directions
                # if 3 components available
                if len(orientations) == 3:
                    azimuth1, dip1 = orientations[0]["azimuth"], orientations[0]["dip"]
                    azimuth2, dip2 = orientations[1]["azimuth"], orientations[1]["dip"]
                    azimuth3, dip3 = orientations[2]["azimuth"], orientations[2]["dip"]
                    vec1, vec2, vec3 = (
                            dir2cart([azimuth1, dip1]),
                            dir2cart([azimuth2, dip2]),
                            dir2cart([azimuth3, dip3]),
                        )
                    components = "ENZ"
                # if 2 components available
                if len(orientations) == 2:
                    azimuth1, dip1 = orientations[0]["azimuth"], orientations[0]["dip"]
                    azimuth2, dip2 = orientations[1]["azimuth"], orientations[1]["dip"]
                    vec1 = dir2cart([azimuth1, dip1])
                    vec2 = dir2cart([azimuth2, dip2])
                    vec3 = np.cross(vec1, vec2)
                    azimuth3, dip3 = cart2dir(vec3)
                # if 1 component available
                if len(orientations) == 1:
                    azimuth1, dip1 = orientations[0]["azimuth"], orientations[0]["dip"]
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
                epsilon = 1e-4
                # if (
                #     abs(np.dot(vec1, vec2)) > epsilon
                #     or abs(np.dot(vec1, vec3)) > epsilon
                #     or abs(np.dot(vec2, vec3)) > epsilon
                # ):
                #     logging.warning(f"{event_id}/{station_channel_id} has invalid channel orientations")
                #     continue
                if (
                        (abs(azimuth1 - 90) < epsilon)
                        and (abs(dip1) < epsilon)
                        and (abs(azimuth2) < epsilon)
                        and (abs(dip2) < epsilon)
                        and (abs(azimuth3) < epsilon)
                        and (abs(dip3 + 90) < epsilon)
                    ):
                    # logging.info(f"{event_id}/{station_channel_id} is already in ENZ")
                    pass
                elif (len(orientations) == 1) and (abs(azimuth1) < epsilon) and (abs(dip1 + 90) < epsilon):
                    array = array[[1, 2, 0], :]  # Zxx -> xxZ
                    # logging.info(f"{event_id}/{station_channel_id} has one componet Z")
                    pass
                else:
                    #FIXME: Error: The given directions are not linearly independent, at least within numerical precision. Determinant of the base change matrix: 0
                    try:
                        znewaveforms = rotate2zne(
                                array[0, :], azimuth1, dip1, array[1, :], azimuth2, dip2, array[2, :], azimuth3, dip3
                            )
                    except:
                        logging.error(
                                f"{event_id}/{station_channel_id} has invalid channel orientations: ({azimuth1}, {dip1}; {azimuth2}, {dip2}; {azimuth3}, {dip3})"
                            )
                        continue
                    array[:] = np.array(znewaveforms)[[2, 1, 0], :]  # ZNE -> ENZ
                    logging.warning(
                            f"{event_id}/{station_channel_id} rotate from {components}:({azimuth1}, {dip1}; {azimuth2}, {dip2}; {azimuth3}, {dip3}) to ENZ: (90, 0; 0, 0; 0, -90)"
                        )
                    # assign zeros if one component has very low values
                    # maxv = np.max(np.abs(array)) / 1e10
                    # array[np.max(np.abs(array), axis=1) < maxv, :] = 0

                station_id = f"{network}.{station}.{location}"
                if station_id not in phases_by_station.index:
                    logging.warning(f"{event_id}/{station_id} sation not in phase picks")
                    continue
                picks_ = phases_by_station.loc[[station_id]]
                picks_ = picks_[(picks_["phase_time"] > begin_time) & (picks_["phase_time"] < end_time)]
                if len(picks_[picks_["event_id"] == event_id]) == 0:
                    logging.warning(f"{event_id}/{station_id} no phase picks")
                    continue

                pick = picks_[picks_["event_id"] == event_id].iloc[0]  # after sort_value
                tmp = int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate))
                if (tmp - 300 < 0) or (tmp + 300 >= NT):
                    # logging.warning(f"{event_id}/{station_id} picks out of time range")
                    continue

                snr = calc_snr(array, int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate)))
                if max(snr) == 0:
                    continue
                (
                    phase_type,
                    phase_index,
                    phase_score,
                    phase_time,
                    phase_remark,
                    phase_polarity,
                    phase_picking_channel,
                    phase_event_id,
                ) = extract_pick(picks_, begin_time, sampling_rate, event_id)
                if len(phase_type) == 0:
                    logging.warning(f"{event_id}/{station_id} has no valid phase picks")
                    continue

                # flip the P polarity if the vertical channel is reversed
                phase_picking_channel_x = [".".join([station_id, x]) for x in phase_picking_channel]
                channel_dip = []
                for x in phase_picking_channel_x:
                    try:
                        channel_dip.append(inv.get_channel_metadata(x, arrival_time)["dip"])
                    except:
                        channel_dip.append("none")
                if 90.0 in channel_dip:
                    logging.warning(
                            f"{event_id}/{station_id}: {phase_picking_channel_x} {channel_dip} has 90.0 dip"
                        )
                phase_polarity = flip_polarity(phase_polarity, channel_dip)

                # save to hdf5
                ds = gp.create_dataset(station_channel_id, data=array, dtype=np.float32)
                ds.attrs["network"] = network
                ds.attrs["station"] = station
                ds.attrs["location"] = location
                ds.attrs["instrument"] = instrument
                ds.attrs["component"] = components
                ds.attrs["unit"] = "1e-6m/s" if instrument[-1] != "N" else "1e-6m/s**2"
                ds.attrs["dt_s"] = 1/sampling_rate
                # at least one channel is available
                ds.attrs["longitude"] = orientations[0]["longitude"]
                ds.attrs["latitude"] = orientations[0]["latitude"]
                ds.attrs["elevation_m"] = orientations[0]["elevation"]
                ds.attrs["local_depth_m"] = orientations[0]["local_depth"]
                ds.attrs["depth_km"] = round(-0.001 * (ds.attrs["elevation_m"] - ds.attrs["local_depth_m"]), 4)
                dist_azi = obspy.geodetics.base.gps2dist_azimuth(
                    gp.attrs["latitude"], # event_latitude
                    gp.attrs["longitude"], # event_longitude
                    ds.attrs["latitude"], # station_latitude
                    ds.attrs["longitude"], # station_longitude
                    #a=6371000.0, f=0.0
                )
                ds.attrs["azimuth"] = round(dist_azi[1], 6)
                ds.attrs["back_azimuth"] = round(dist_azi[2], 6)
                ds.attrs["distance_km"] = round(dist_azi[0] / 1000, 6)
                #assert np.abs(pick.azimuth - dist_azi[1]) < 1, f"{pick.azimuth} != {dist_azi[1]}"
                # the azimuth can be +- 1 degree
                #assert np.abs(pick.distance_km - dist_azi[0] / 1000) < 5e-1, f"{pick.distance_km} != {dist_azi[0] / 1000}"
                # the distance can be +- 2 km
                #if "azimuth" in pick:
                #    ds.attrs["azimuth"] = pick.azimuth
                #if "distance_km" in pick:
                #    ds.attrs["distance_km"] = pick.distance_km
                if "takeoff_angle" in pick:
                    ds.attrs["takeoff_angle"] = pick.takeoff_angle
                ds.attrs["snr"] = snr
                ds.attrs["phase_type"] = phase_type
                ds.attrs["phase_index"] = phase_index
                ds.attrs["phase_score"] = phase_score
                ds.attrs["phase_time"] = phase_time
                ds.attrs["phase_remark"] = phase_remark
                ds.attrs["phase_polarity"] = phase_polarity
                ds.attrs["phase_picking_channel"] = phase_picking_channel
                ds.attrs["event_id"] = phase_event_id
                main_p_slice = np.logical_and(
                        np.array(phase_event_id) == event_id, np.array(phase_type) == "P"
                    )
                main_s_slice = np.logical_and(
                        np.array(phase_event_id) == event_id, np.array(phase_type) == "S"
                    )
                main_phase_index_p = np.array(phase_index)[main_p_slice]
                main_phase_index_s = np.array(phase_index)[main_s_slice]
                assert len(main_phase_index_p) <=1 and len(main_phase_index_s) <= 1
                if len(main_phase_index_p) == 1 and len(main_phase_index_s) == 1:
                    ds.attrs["phase_status"] = "manual"
                    has_station = True
                else:
                    ##FIXME: keep the stations if it don't have main phase pair but have other pair
                    #if (len(np.unique(np.array(phase_type)[(np.array(phase_event_id) != event_id)]))>= 2):
                    #    has_station = True
                    ds.attrs["phase_status"] = "automatic"
                main_p_idx = np.where(main_p_slice)[0][0] if len(main_phase_index_p) == 1 else None
                main_s_idx = np.where(main_s_slice)[0][0] if len(main_phase_index_s) == 1 else None
                if main_p_idx is not None:
                    ds.attrs['p_phase_index'] = main_phase_index_p[0]
                    ds.attrs['p_phase_score'] = phase_score[main_p_idx]
                    ds.attrs['p_phase_time'] = phase_time[main_p_idx]
                    ds.attrs['p_phase_polarity'] = phase_polarity[main_p_idx]
                    ds.attrs['p_phase_status'] = ds.attrs["phase_status"]
                if main_s_idx is not None:
                    ds.attrs['s_phase_index'] = main_phase_index_s[0]
                    ds.attrs['s_phase_score'] = phase_score[main_s_idx]
                    ds.attrs['s_phase_time'] = phase_time[main_s_idx]
                    ds.attrs['s_phase_polarity'] = phase_polarity[main_s_idx]
                    ds.attrs['s_phase_status'] = "manual"
                
                num_stations += 1

            gp.attrs["nx"] = num_stations
            if not has_station:
                logging.warning(f"{event_id} has no stations")    
                del fp[event_id]

    return None

if __name__ == "__main__":
    # %%
    years = sorted(fs.ls(mseed_path), reverse=True)
    years = [x.split("/")[-1] for x in years]
    # years = ["2022"]
    MAX_THREADS = 32
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=MAX_THREADS) as pool:
        for year in years:
            jdays = sorted(fs.ls(f"{mseed_path}/{year}"), reverse=False)
            jdays = [x.split("/")[-1] for x in jdays]
            pbar = tqdm(jdays, total=len(jdays), desc=f"{year}", leave=True)
            # for jday in jdays[1:]:
            #     convert_jday(jday, catalog_path, result_path, protocol, token)
            #     pbar.update(1)
            # raise

            processes = []
            for jday in jdays:
                p = pool.apply_async(convert_jday, args=(jday, catalog_path, result_path, protocol, token), callback=lambda x: pbar.update(1))
                processes.append(p)
            for p in processes:
                try:
                    out = p.get()
                    if out is not None:
                        print(out)
                except Exception as e:
                    print(f"{type(e)}: {e}")
            pbar.close()
        
            with h5py.File(f"{result_path}/{year}.h5", "w") as fp:
                for jday in tqdm(jdays):
                    year, dayofyear = jday.split(".")
                    with h5py.File(f"{result_path}/{year}/{dayofyear}.h5", "r") as f:
                        for event_id in f:
                            f.copy(event_id, fp)
            
            os.system(f"rm -rf {result_path}/{year}")

    # convert(0, "2019")
    # raise

    # ncpu = len(years)
    # ctx = mp.get_context("spawn")
    # # ctx = mp.get_context("fork")
    # with ctx.Pool(ncpu) as pool:
    #     pool.starmap(convert, [x for x in enumerate(years)])

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

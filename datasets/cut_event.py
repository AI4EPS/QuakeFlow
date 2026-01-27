# %%
import argparse
import json
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import fsspec
import h5py
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm

np.random.seed(42)

# Constants
GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
SAMPLING_RATE = 100.0
TIME_BEFORE = 40.96  # seconds before first arrival
TIME_AFTER = 40.96 * 2  # seconds after first arrival
NT = int(round((TIME_BEFORE + TIME_AFTER) * SAMPLING_RATE))  # 12288 samples


def set_attr(obj, key, value):
    """Set an attribute only if the value is not empty."""
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
        # Check if all elements are empty strings
        if all(v == "" for v in value):
            return
    obj.attrs[key] = value


# %%
def calc_snr(data, index0, noise_window=300, signal_window=300, gap_window=50):
    """Calculate signal-to-noise ratio for each channel.

    Args:
        data: Array of shape (3, nt) with waveform data
        index0: Sample index of the first arrival
        noise_window: Number of samples for noise window
        signal_window: Number of samples for signal window
        gap_window: Gap between noise and signal windows

    Returns:
        List of SNR values for each channel
    """
    snr = []
    for i in range(data.shape[0]):
        j = index0
        noise_start = max(0, j - noise_window - gap_window)
        noise_end = max(0, j - gap_window)
        signal_start = min(data.shape[1], j)
        signal_end = min(data.shape[1], j + signal_window)

        if noise_end <= noise_start or signal_end <= signal_start:
            snr.append(0)
            continue

        noise = np.std(data[i, noise_start:noise_end])
        signal = np.std(data[i, signal_start:signal_end])

        if noise > 0 and signal > 0:
            snr.append(signal / noise)
        else:
            snr.append(0)

    return snr


def set_phase_attrs(ds, phase_prefix, pick_df):
    """Set phase attributes for P or S picks on a dataset.

    Args:
        ds: HDF5 dataset to set attributes on
        phase_prefix: 'p' or 's' for the attribute prefix
        pick_df: DataFrame containing the pick data
    """
    if len(pick_df) == 0:
        return

    set_attr(ds, f"{phase_prefix}_phase_time", pick_df["phase_time"].iloc[0].strftime("%Y-%m-%dT%H:%M:%S.%f"))
    set_attr(ds, f"{phase_prefix}_phase_index", pick_df["phase_index"].values[0])
    set_attr(ds, f"{phase_prefix}_phase_score", pick_df["phase_score"].values[0])
    set_attr(ds, f"{phase_prefix}_phase_polarity", pick_df["phase_polarity"].values[0])
    set_attr(ds, f"{phase_prefix}_phase_remark", pick_df["phase_remark"].values[0])

    optional_cols = [
        ("time_residual", f"{phase_prefix}_time_residual"),
        ("phase_weight", f"{phase_prefix}_phase_weight"),
        ("review_status", f"{phase_prefix}_phase_status"),
    ]
    for col, attr_name in optional_cols:
        if col in pick_df.columns:
            set_attr(ds, attr_name, pick_df[col].values[0])


def set_optional_attrs(ds, df, column_mappings):
    """Set optional attributes from DataFrame columns.

    Args:
        ds: HDF5 dataset to set attributes on
        df: DataFrame containing the data
        column_mappings: List of (column_name, attr_name) tuples, or just column names
    """
    for mapping in column_mappings:
        if isinstance(mapping, tuple):
            col, attr_name = mapping
        else:
            col = attr_name = mapping
        if col in df.columns:
            set_attr(ds, attr_name, df[col].values)


# %%
def extract_template_numpy(
    picks_group,
    events,
    mechanisms,
    config,
    fp,
    lock,
):

    region = config["region"]

    for picks_df in picks_group:

        waveforms_dict = {}

        # for _, pick in picks_df.iterrows():
        for key, picks_ in picks_df.groupby(["event_id", "network", "station", "location", "instrument"]):

            event_id, network, station, location, instrument = key
            station_id = f"{network}.{station}.{location}.{instrument}"

            if event_id not in events.index:
                print(f"Event_id {event_id} does not have event information")
                continue

            begin_time = picks_.iloc[0]["begin_time"]
            end_time = picks_.iloc[0]["end_time"]

            p_pick = picks_[picks_["phase_type"] == "P"]
            s_pick = picks_[picks_["phase_type"] == "S"]

            pick = p_pick.iloc[0] if len(p_pick) > 0 else s_pick.iloc[0]

            if len(picks_) > 2:
                print(f"More than two picks: {picks_}")
                continue
                
            event = events.loc[event_id]

            mseed_3c = pick["mseed_3c"].split("|")

            if pick["mseed_3c"] not in waveforms_dict:
                stream_3c = obspy.Stream()
                for c in mseed_3c:
                    try:
                        with fsspec.open(c, "rb", anon=True) as f:
                            stream = obspy.read(f)
                            stream.merge(fill_value="latest")
                            if len(stream) > 1:
                                print(f"More than one trace: {stream}")
                            trace = stream[0]
                            if trace.stats.sampling_rate != config["sampling_rate"]:
                                if trace.stats.sampling_rate % config["sampling_rate"] == 0:
                                    trace.decimate(int(trace.stats.sampling_rate / config["sampling_rate"]))
                                else:
                                    trace.resample(config["sampling_rate"])
                            stream_3c.append(trace)
                    except Exception as err:
                        print(f"Error reading mseed: {err} on {c}")
                        continue

                try:
                    fs = fsspec.filesystem("gs", token=GCS_CREDENTIALS_PATH)
                    if region == "SC":
                        with fs.open(
                            f"quakeflow_dataset/SCEDC/FDSNstationXML/{network}/{network}.{station}.xml", "r"
                        ) as f:
                            inv = obspy.read_inventory(f)
                    elif region == "NC":
                        with fs.open(
                            f"quakeflow_dataset/NCEDC/FDSNstationXML/{network}/{network}.{station}.xml", "r"
                        ) as f:
                            inv = obspy.read_inventory(f)
                    else:
                        raise ValueError(f"Invalid region: {region}")

                except Exception as err:
                    print(f"Error reading inventory: {err} on {network}.{station}.xml")
                    continue

                try:
                    stream_3c.remove_sensitivity(inv)
                    stream_3c.detrend("constant")
                    stream_3c.rotate("->ZNE", inventory=inv)
                except Exception as err:
                    print(f"Error removing sensitivity: {err} on {pick['mseed_3c']}")
                    continue

                # Cache the processed stream for reuse
                waveforms_dict[pick["mseed_3c"]] = stream_3c

            else:
                stream_3c = waveforms_dict[pick["mseed_3c"]]

            tmp = stream_3c.slice(
                obspy.UTCDateTime(begin_time), obspy.UTCDateTime(end_time), keep_empty_traces=False, nearest_sample=True
            )

            template = np.zeros((3, config["nt"]), dtype=np.float32)
            component = []
            for i, ch in enumerate(["E", "N", "Z"]): # rotation to mseed_3c
                trace = tmp.select(component=f"{ch}")
                if len(trace) == 0:
                    continue
                elif len(trace) > 1:
                    print(f"More than one trace: {trace}")
                    continue
                else:
                    component.append(ch)
                    template[i, : len(trace[0].data)] = trace[0].data[: config["nt"]] * 1e6  # to micro m/s

            # Calculate SNR based on first arrival
            first_arrival_index = pick["phase_index"]
            snr = calc_snr(template, first_arrival_index)

            # Skip if all channels have zero SNR
            if max(snr) == 0:
                continue

            with lock:
                if f"{event_id}" not in fp:
                    gp = fp.create_group(f"{event_id}")
                    for key, value in event.items():
                        if key in ["event_time", "begin_time", "end_time"]:
                            value = value.strftime("%Y-%m-%dT%H:%M:%S.%f")
                        set_attr(gp, key, value)

                    gp.attrs["begin_time"] = begin_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
                    gp.attrs["end_time"] = end_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

                    if mechanisms is not None and event_id in mechanisms.index:
                        mech = mechanisms.loc[event_id]
                        mech_columns = [
                            "strike", "dip", "rake",
                            "num_first_motions", "first_motion_misfit",
                            "num_sp_ratios", "sp_ratio_misfit",
                            "plane1_uncertainty", "plane2_uncertainty", "quality",
                        ]
                        for key in mech_columns:
                            if key in mech.index and pd.notna(mech[key]):
                                set_attr(gp, key, mech[key])

                ds = fp.create_dataset(f"{event_id}/{station_id}", data=template)
                for key, value in pick.items():
                    if key in ["event_id", "mseed_3c", "begin_time", "end_time"]:
                        continue
                    if key in ["phase_time", "phase_type", "phase_index", "phase_score", "phase_polarity", "phase_remark", "phase_weight", "time_residual", "pick_component"]:
                        continue # Handled later
                    #     value = value.strftime("%Y-%m-%dT%H:%M:%S.%f")
                    set_attr(ds, key, value)

                set_phase_attrs(ds, "p", p_pick)
                set_phase_attrs(ds, "s", s_pick)

                ds.attrs["unit"] = "micro m/s"
                ds.attrs["component"] = "".join(component)
                ds.attrs["snr"] = round(max(snr),3)

                picks_in_window = picks_df[(picks_df["phase_time"] >= begin_time) & (picks_df["phase_time"] <= end_time)].copy()
                picks_in_window.sort_values(by="phase_time", inplace=True)
                set_attr(ds, "event_id", picks_in_window["event_id"].values)
                set_attr(ds, "phase_type", picks_in_window["phase_type"].values)
                set_attr(ds, "phase_time", picks_in_window["phase_time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f")).values)
                set_attr(ds, "phase_index", picks_in_window["phase_index"].values)
                set_attr(ds, "phase_score", picks_in_window["phase_score"].values)
                set_attr(ds, "phase_polarity", picks_in_window["phase_polarity"].values)
                set_attr(ds, "phase_remark", picks_in_window["phase_remark"].values)
                set_optional_attrs(ds, picks_in_window, [
                    "time_residual",
                    "phase_weight",
                    ("review_status", "phase_status"),
                    "azimuth",
                    "back_azimuth",
                    "takeoff_angle",
                ])

    return 

# %%
def cut_templates(jdays, root_path, data_path, result_path, region, config, bucket, protocol, token):

    # %%
    fs = fsspec.filesystem(protocol, token=token)

    config["nt"] = NT
    config["sampling_rate"] = SAMPLING_RATE
    config["time_before"] = TIME_BEFORE
    config["time_after"] = TIME_AFTER

    for jday in jdays:
        year, jday = jday.split(".")
        year, jday = int(year), int(jday)

        os.makedirs(f"{root_path}/{result_path}/{year:04d}/{jday:03d}", exist_ok=True)

        if protocol == "file":
            events = pd.read_csv(f"{root_path}/{data_path}/{year:04d}/{jday:03d}/events.csv", dtype=str)
        else:
            with fs.open(f"{bucket}/{data_path}/{year:04d}/{jday:03d}/events.csv", "r") as fp:
                events = pd.read_csv(fp, dtype=str)

        events.fillna("", inplace=True)
        events.rename(columns={"time": "event_time"}, inplace=True)
        events["event_time"] = pd.to_datetime(events["event_time"])

        print(f"{len(events) = }")
        print(events.head())

        # %%
        if protocol == "file":
            picks = pd.read_csv(f"{root_path}/{data_path}/{year:04d}/{jday:03d}/phases.csv", dtype=str)
        else:
            with fs.open(f"{bucket}/{data_path}/{year:04d}/{jday:03d}/phases.csv", "r") as fp:
                picks = pd.read_csv(fp, dtype=str)

        picks.fillna("", inplace=True)
        picks.rename(columns={"component": "pick_component"}, inplace=True)
        picks["phase_time"] = pd.to_datetime(picks["phase_time"])
        picks["begin_time"] = picks.groupby("event_id")["phase_time"].transform("min") - pd.Timedelta(
            seconds=config["time_before"]
        )
        picks["end_time"] = picks.groupby("event_id")["phase_time"].transform("min") + pd.Timedelta(
            seconds=config["time_after"]
        )
        picks["phase_index"] = (
            (picks["phase_time"] - picks["begin_time"]).dt.total_seconds() * config["sampling_rate"]
        ).astype(int)

        # Set proper data types
        picks["event_id"] = picks["event_id"].astype(str)
        picks["phase_type"] = picks["phase_type"].astype(str)
        picks["phase_score"] = picks["phase_score"].astype(float)
        picks["phase_polarity"] = picks["phase_polarity"].fillna("").astype(str)
        picks["phase_remark"] = picks["phase_remark"].fillna("").astype(str)
        if "time_residual" in picks.columns:
            picks["time_residual"] = pd.to_numeric(picks["time_residual"], errors="coerce")
        if "phase_weight" in picks.columns:
            picks["phase_weight"] = pd.to_numeric(picks["phase_weight"], errors="coerce")
        if "review_status" in picks.columns:
            picks["review_status"] = picks["review_status"].fillna("").astype(str)

        print(f"{len(picks) = }")
        print(picks.head())

        tmp = picks.groupby("event_id")["begin_time"].first()
        # events = events.merge(tmp, on="event_id", how="left")
        events = events.merge(tmp, on="event_id", how="inner")
        picks = picks[picks["event_id"].isin(events["event_id"])]
        events["event_time_index"] = (
            (events["event_time"] - events["begin_time"]).dt.total_seconds() * config["sampling_rate"]
        ).astype(int)
            
        events.set_index("event_id", inplace=True)

        # Read focal mechanisms
        mechanisms = None
        try:
            if protocol == "file":
                mechanisms = pd.read_csv(
                    f"{root_path}/{data_path}/{year:04d}/{jday:03d}/focal_mechanisms.csv", dtype=str
                )
            else:
                with fs.open(f"{bucket}/{data_path}/{year:04d}/{jday:03d}/focal_mechanisms.csv", "r") as fp:
                    mechanisms = pd.read_csv(fp, dtype=str)
            mechanisms.set_index("event_id", inplace=True)
            print(f"{len(mechanisms) = }")
        except FileNotFoundError:
            print(f"No focal mechanisms file for {year:04d}/{jday:03d}")

        ############################# CLOUD #########################################
        # Use separate filesystem for quakeflow_dataset bucket (contains mseed list and station XML)
        dataset_bucket = "quakeflow_dataset"
        dataset_token_json = GCS_CREDENTIALS_PATH
        with open(dataset_token_json, "r") as fp:
            dataset_token = json.load(fp)
        dataset_fs = fsspec.filesystem(protocol="gs", token=dataset_token)

        # Map region to folder name
        region_folder = "SCEDC" if region == "SC" else "NCEDC"

        # Read mseed list for this specific jday (new format: {year}/{jday}.txt)
        with dataset_fs.open(f"{dataset_bucket}/{region_folder}/mseed/{year}/{jday:03d}.txt", "r") as f:
            mseeds = f.readlines()
        mseeds = [x.strip("\n") for x in mseeds]
        mseeds_df = pd.DataFrame(mseeds, columns=["mseed_3c"])

        # Parse first file in each 3c group to extract metadata
        # mseed_3c format: "s3://bucket/path/file1.ms|s3://bucket/path/file2.ms|..."
        mseeds_df["fname"] = mseeds_df["mseed_3c"].apply(lambda x: x.split("|")[0].split("/")[-1])

        if region == "SC":
            # SCEDC format: {network}{station}{location}{channel}___{year}{jday}.ms
            # Example: AZCRY__BHE___2026001.ms
            mseeds_df["network"] = mseeds_df["fname"].apply(lambda x: x[:2])
            mseeds_df["station"] = mseeds_df["fname"].apply(lambda x: x[2:7].rstrip("_"))
            mseeds_df["instrument"] = mseeds_df["fname"].apply(lambda x: x[7:9])
            mseeds_df["location"] = mseeds_df["fname"].apply(lambda x: x[10:12].rstrip("_"))
        else:  # NC
            # NCEDC format: {station}.{network}.{channel}.{location}.{quality}.{year}.{jday}
            # Example: ABL.CI.HHE..D.2026.001
            mseeds_df["network"] = mseeds_df["fname"].apply(lambda x: x.split(".")[1])
            mseeds_df["station"] = mseeds_df["fname"].apply(lambda x: x.split(".")[0])
            mseeds_df["instrument"] = mseeds_df["fname"].apply(lambda x: x.split(".")[2][:-1])
            mseeds_df["location"] = mseeds_df["fname"].apply(lambda x: x.split(".")[3])

        print(f"{len(mseeds_df) = }")
        print(mseeds_df.head())

        picks = picks.merge(mseeds_df, on=["network", "station", "location", "instrument"])
        picks.drop(columns=["fname"], inplace=True)

        if len(picks) == 0:
            print(f"No picks found for {year:04d}/{jday:03d}")
            continue

        picks_group = picks.copy()
        picks_group = picks_group.groupby("mseed_3c")

        ############################################################
        with h5py.File(f"{root_path}/{result_path}/{year:04d}/{jday:03d}/waveform.h5", "w") as fp:

            # ## FIXME: For testing purpose, use single processing
            # nsplit = 1
            # group_chunk = np.array_split(list(picks_group.groups.keys()), nsplit)
            # picks_group_chunk = [[picks_group.get_group(g) for g in group] for group in group_chunk]

            # lock = nullcontext()
            # for group in picks_group_chunk:
            #     extract_template_numpy(group, events, picks, mechanisms, config, fp, lock)
            # sys.exit(0)

            ncpu = min(8, multiprocessing.cpu_count())
            nsplit = len(picks_group)
            print(f"Using {ncpu} cores")

            group_chunk = np.array_split(list(picks_group.groups.keys()), nsplit)
            picks_group_chunk = [[picks_group.get_group(g) for g in group] for group in group_chunk]

            lock = threading.Lock()  # Use threading.Lock for ThreadPoolExecutor

            # with ProcessPoolExecutor(max_workers=ncpu) as executor:
            with ThreadPoolExecutor(max_workers=ncpu) as executor:
                futures = [
                    executor.submit(
                        partial(
                            extract_template_numpy,
                            events=events,
                            mechanisms=mechanisms,
                            config=config,
                            fp=fp,
                            lock=lock,
                        ),
                        group
                    )
                    for group in picks_group_chunk
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting templates"):
                    out = future.result()
                    if out is not None:
                        print(out)

        # Upload the HDF5 file to cloud storage
        local_h5_path = f"{root_path}/{result_path}/{year:04d}/{jday:03d}/waveform.h5"
        remote_h5_path = f"{bucket}/{result_path}/{year:04d}/{jday:03d}/waveform.h5"
        print(f"Upload {local_h5_path} to {remote_h5_path}")
        fs.put(local_h5_path, remote_h5_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="SC")
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument("--bucket", type=str, default="quakeflow_dataset")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--year", type=int, default=2024)

    return parser.parse_args()



# %%
if __name__ == "__main__":

    # %%
    protocol = "gs"
    with open(GCS_CREDENTIALS_PATH, "r") as fp:
        token = json.load(fp)

    # protocol = "file"
    # token = None

    # %%
    args = parse_args()
    region = args.region
    root_path = args.root_path
    bucket = args.bucket
    num_nodes = args.num_nodes
    node_rank = args.node_rank
    year = args.year

    num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365
    jdays = range(1, num_jday + 1)

    jdays = np.array_split(jdays, num_nodes)[node_rank]

    processed = []

    jdays = [jday for jday in jdays if f"{jday:03d}" not in processed]
    print(f"Remaining days: {len(jdays)}")

    if len(jdays) == 0:
        print("No days to process")
        exit()

    jdays = [f"{year}.{jday:03d}" for jday in jdays]

    print(f"{node_rank}/{num_nodes}: {jdays[0] = }, {jdays[-1] = }")

    config = vars(args)

    # %%
    if protocol == "file":
        data_path = f"{region}EDC/dataset"
        result_path = f"{region}EDC/dataset"
    else:
        data_path = f"{region}EDC/catalog" # sorry for the inconsistency
        result_path = f"{region}EDC/dataset"

    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    ## FIXME: Hardcode for testing
    jdays = ["2025.001"]
    cut_templates(jdays, root_path, data_path, result_path, region, config, bucket, protocol, token)

# %%
import gzip
import multiprocessing as mp
import os
import re
import shutil
from collections import namedtuple
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import fsspec
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm

# %%
input_protocol = "s3"
input_bucket = "ncedc-pds"
input_fs = fsspec.filesystem(input_protocol, anon=True)

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset"
output_fs = fsspec.filesystem(output_protocol, token=output_token)

# %%
catalog_path = f"{input_bucket}/event_phases"
station_path = f"{input_bucket}/FDSNstationXML"
waveform_path = f"{input_bucket}/continuous_waveforms/"
dataset_path = f"{output_bucket}/NC/catalog"

# %%
result_path = "dataset"
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(f"{result_path}/catalog_raw"):
    os.makedirs(f"{result_path}/catalog_raw")
if not os.path.exists(f"{result_path}/catalog"):
    os.makedirs(f"{result_path}/catalog")

## https://ncedc.org/ftp/pub/doc/ncsn/shadow2000.pdf
# %%
event_columns = {
    "year": (0, 4),
    "month": (4, 6),
    "day": (6, 8),
    "hour": (8, 10),
    "minute": (10, 12),
    "seconds": (12, 16),  # / 100
    "latitude_deg": (16, 18),
    "s_indicator": (18, 19),
    "latitude_min": (19, 23),
    "longitude_deg": (23, 26),
    "e_indicator": (26, 27),
    "longitude_min": (27, 31),
    "depth_km": (31, 36),  # /100
    "magnitude_max_s_amplitude": (36, 39),
    "num_p_s_times_weighted": (39, 42),
    "max_azimuthal_gap": (42, 45),
    "nearest_station_distance_km": (45, 48),
    "rms_travel_time_residual": (48, 52),
    "largest_principal_error_azimuth": (52, 55),
    "largest_principal_error_dip": (55, 57),
    "largest_principal_error_size_km": (57, 61),
    "intermediate_principal_error_azimuth": (61, 64),
    "intermediate_principal_error_dip": (64, 66),
    "intermediate_principal_error_size_km": (66, 70),
    "coda_duration_magnitude": (70, 73),
    "event_location_remark": (73, 76),
    "smallest_principal_error_size_km": (76, 80),
    "auxiliary_remarks": (80, 82),
    "num_s_times_weighted": (82, 85),
    "horizontal_error_km": (85, 89),
    "vertical_error_km": (89, 93),
    "num_p_first_motions": (93, 96),
    "ncsn_s_amplitude_mag_weights": (96, 100),
    "ncsn_duration_mag_weights": (100, 104),
    "median_abs_diff_ncsn_s_amp_magnitudes": (104, 107),
    "median_abs_diff_ncsn_duration_magnitudes": (107, 110),
    "crust_delay_model_code": (110, 113),
    "last_authority_for_earthquake": (113, 114),
    "common_p_s_data_source_code": (114, 115),
    "common_duration_data_source_code": (115, 116),
    "common_amplitude_data_source_code": (116, 117),
    "coda_duration_magnitude_type_code": (117, 118),
    "valid_p_s_readings_num": (118, 121),
    "s_amplitude_magnitude_type_code": (121, 122),
    "external_magnitude_label_or_type_code": (122, 123),
    "external_magnitude": (123, 126),
    "external_magnitude_weights": (126, 129),
    "alternate_amplitude_magnitude_label_or_type_code": (129, 130),
    "alternate_amplitude_magnitude": (130, 133),
    "alternate_amplitude_mag_weights": (133, 136),
    "event_id": (136, 146),
    "preferred_magnitude_label_code": (146, 147),
    "preferred_magnitude": (147, 150),  # /100
    "preferred_mag_weights": (150, 154),
    "alternate_coda_duration_magnitude_label_or_type_code": (154, 155),
    "alternate_coda_duration_magnitude": (155, 158),
    "alternate_coda_duration_magnitude_weights": (158, 162),
    "qdds_version_number": (162, 163),
    "origin_instance_version_number": (163, 164),
}

event_decimal_number = {
    "seconds": 100,
    "latitude_deg": 1,
    "latitude_min": 100,
    "longitude_deg": 1,
    "longitude_min": 100,
    "depth_km": 100,
    "preferred_magnitude": 100,
}

phase_columns = {
    "station": (0, 5),
    "network": (5, 7),
    "station_component_code_one_letter": (8, 9),
    "instrument": (9, 11),
    "component": (11, 12),
    "channel": (9, 12),
    "p_remark": (13, 15),
    "p_polarity": (15, 16),
    "p_weight_code": (16, 17),
    "year": (17, 21),
    "month": (21, 23),
    "day": (23, 25),
    "hour": (25, 27),
    "minute": (27, 29),
    "second_of_p_arrival": (29, 34),  # /100
    "p_travel_time_residual": (34, 38),  # /100
    "normalized_p_weight_actually_used": (38, 41),  # /100
    "second_of_s_arrival": (41, 46),  # /100
    "s_remark": (46, 48),
    "s_weight_code": (49, 50),  # /100
    "s_travel_time_residual": (50, 54),  # /100
    "amplitude": (54, 61),
    "amp_units_code": (61, 63),
    "s_weight_actually_used": (63, 66),
    "p_delay_time": (66, 70),
    "s_delay_time": (70, 74),
    "distance_km": (74, 78),  # /10
    "takeoff_angle": (78, 81),
    "amplitude_magnitude_weight_code": (81, 82),
    "duration_magnitude_weight_code": (82, 83),
    "period": (83, 86),
    "station_remark": (86, 87),
    "coda_duration": (87, 91),
    "azimuth": (91, 94),
    "duration_magnitude_for_this_station": (94, 97),
    "amplitude_magnitude_for_this_station": (97, 100),
    "importance_of_p_arrival": (100, 104),
    "importance_of_s_arrival": (104, 108),
    "data_source_code": (108, 109),
    "label_code_for_duration_magnitude": (109, 110),
    "label_code_for_amplitude_magnitude": (110, 111),
    "location": (111, 113),
    "amplitude_type": (113, 115),
    "alternate_component_code": (115, 118),
    "amplitude_magnitude_not_used": (118, 119),
    "duration_magnitude_not_used": (119, 120),
}

phase_decimal_number = {
    "second_of_p_arrival": 100,
    "second_of_s_arrival": 100,
    "distance_km": 10,
    "p_travel_time_residual": 100,
    "s_travel_time_residual": 100,
    "normalized_p_weight_actually_used": 100,
    "s_weight_actually_used": 100,
}


def read_event_line(line):
    event = {}
    for key, (start, end) in event_columns.items():
        if key in event_decimal_number:
            try:
                event[key] = float(line[start:end]) / event_decimal_number[key]
            except:
                print(key, line[start:end])
        else:
            event[key] = line[start:end]

    if event["seconds"] < 60:
        event["time"] = (
            f"{event['year']}-{event['month']}-{event['day']}T{event['hour']}:{event['minute']}:{event['seconds']:06.3f}"
        )
    else:
        tmp = datetime.fromisoformat(
            f"{event['year']}-{event['month']}-{event['day']}T{event['hour']}:{event['minute']}"
        )
        tmp += timedelta(seconds=event["seconds"])
        event["time"] = tmp.strftime("%Y-%m-%dT%H:%M:%S.%f")

    event["latitude"] = round(event["latitude_deg"] + event["latitude_min"] / 60, 6)
    event["longitude"] = round(-(event["longitude_deg"] + event["longitude_min"] / 60), 6)
    if event["s_indicator"] == "S":
        event["latitude"] = -event["latitude"]
    if event["e_indicator"] == "E":
        event["longitude"] = -event["longitude"]
    event["magnitude"] = event["preferred_magnitude"]
    event["magnitude_type"] = event["preferred_magnitude_label_code"]
    event["event_id"] = event["event_id"].strip()
    return event


def read_phase_line(line):
    ## check p_remark
    phases = []
    start, end = phase_columns["p_remark"]
    if len(line[start:end].strip()) > 0:
        p_phase = {}
        for key, (start, end) in phase_columns.items():
            # ######## filter strange data ############
            # if key == "p_travel_time_residual":
            #     if line[start : end + 3] == " " * 3 + "0" + " " * 2 + "0":
            #         # print(f"strange data: {line}")
            #         return []
            # #########################################
            if key in phase_decimal_number:
                if len(line[start:end].strip()) == 0:
                    p_phase[key] = ""
                else:
                    p_phase[key] = float(line[start:end].strip()) / phase_decimal_number[key]
            else:
                p_phase[key] = line[start:end]
        if (p_phase["second_of_p_arrival"] < 60) and (p_phase["second_of_p_arrival"] >= 0):
            p_phase["phase_time"] = (
                f"{p_phase['year']}-{p_phase['month']}-{p_phase['day']}T{p_phase['hour']}:{p_phase['minute']}:{p_phase['second_of_p_arrival']:06.3f}"
            )
        else:
            tmp = datetime.fromisoformat(
                f"{p_phase['year']}-{p_phase['month']}-{p_phase['day']}T{p_phase['hour']}:{p_phase['minute']}"
            )
            tmp += timedelta(seconds=p_phase["second_of_p_arrival"])
            p_phase["phase_time"] = tmp.strftime("%Y-%m-%dT%H:%M:%S.%f")
        p_phase["phase_polarity"] = p_phase["p_polarity"]
        p_phase["phase_remark"] = p_phase["p_remark"]
        p_phase["phase_score"] = p_phase["p_weight_code"]
        p_phase["phase_type"] = "P"
        p_phase["location_residual_s"] = p_phase["p_travel_time_residual"]
        p_phase["location_weight"] = p_phase["normalized_p_weight_actually_used"]
        phases.append(p_phase)
    start, end = phase_columns["s_remark"]
    if len(line[start:end].strip()) > 0:
        s_phase = {}
        for key, (start, end) in phase_columns.items():
            if key in phase_decimal_number:
                if len(line[start:end].strip()) == 0:
                    s_phase[key] = ""
                else:
                    s_phase[key] = float(line[start:end].strip()) / phase_decimal_number[key]
            else:
                s_phase[key] = line[start:end]
        if (s_phase["second_of_s_arrival"] < 60) and (s_phase["second_of_s_arrival"] >= 0):
            s_phase["phase_time"] = (
                f"{s_phase['year']}-{s_phase['month']}-{s_phase['day']}T{s_phase['hour']}:{s_phase['minute']}:{s_phase['second_of_s_arrival']:06.3f}"
            )
        else:
            tmp = datetime.fromisoformat(
                f"{s_phase['year']}-{s_phase['month']}-{s_phase['day']}T{s_phase['hour']}:{s_phase['minute']}"
            )
            tmp += timedelta(seconds=s_phase["second_of_s_arrival"])
            s_phase["phase_time"] = tmp.strftime("%Y-%m-%dT%H:%M:%S.%f")
        s_phase["phase_remark"] = s_phase["s_remark"]
        s_phase["phase_score"] = s_phase["s_weight_code"]
        s_phase["phase_type"] = "S"
        s_phase["location_residual_s"] = s_phase["s_travel_time_residual"]
        s_phase["location_weight"] = s_phase["s_weight_actually_used"]
        phases.append(s_phase)

    return phases


# %%
# for year in range(1966, 2024)[::-1]:
def process(year):
    # for phase_file in sorted(glob(f"{catalog_path}/{year}/*.phase.Z"))[::-1]:
    # print(f"{catalog_path}/{year}/*.phase.Z")

    for phase_file in sorted(input_fs.glob(f"{catalog_path}/{year}/*.phase.Z"), reverse=True):
        phase_filename = phase_file.split("/")[-1]

        # if not os.path.exists(f"{result_path}/catalog_raw/{phase_filename[:-2]}"):
        # shutil.copy(phase_file, f"{result_path}/catalog_raw/{phase_filename}")
        input_fs.get(phase_file, f"{result_path}/catalog_raw/{phase_filename}")
        os.system(f"uncompress -f {result_path}/catalog_raw/{phase_filename}")

        with open(f"{result_path}/catalog_raw/{phase_filename[:-2]}") as f:
            lines = f.readlines()
        catalog = {}
        event_id = None
        for line in tqdm(lines, desc=phase_filename):
            if len(line) > (180 + 114) / 2:  # event_line
                if event_id is not None:
                    assert event["event_id"] == event_id
                    catalog[event_id] = {"event": event, "picks": picks}
                event = read_event_line(line)
                picks = []
            elif len(line) > (73 + 114) / 2:  # phase_line
                picks.extend(read_phase_line(line))
            else:
                event_id = line.strip().split(" ")[-1]
        catalog[event_id] = {"event": event, "picks": picks}  # last event

        events = []
        phases = []
        for event_id in catalog:
            if len(catalog[event_id]["picks"]) == 0:
                continue
            events.append(catalog[event_id]["event"])
            phase = pd.DataFrame(catalog[event_id]["picks"])
            phase["event_id"] = event_id
            phases.append(phase)

        events = pd.DataFrame(events)
        if len(phases) == 0:
            continue
        phases = pd.concat(phases)
        events = events[["event_id", "time", "latitude", "longitude", "depth_km", "magnitude", "magnitude_type"]]
        events["event_id"] = events["event_id"].apply(lambda x: "nc" + x)
        events["time"] = events["time"].apply(lambda x: x + "+00:00")
        phases = phases[
            [
                "event_id",
                "network",
                "station",
                "location",
                "instrument",
                "component",
                "phase_type",
                "phase_time",
                "phase_score",
                "phase_polarity",
                "phase_remark",
                "distance_km",
                "azimuth",
                "takeoff_angle",
                "location_residual_s",
                "location_weight",
            ]
        ]
        phases["event_id"] = phases["event_id"].apply(lambda x: "nc" + x)
        phases["phase_time"] = phases["phase_time"].apply(lambda x: x + "+00:00")
        phases["network"] = phases["network"].str.strip()
        phases["station"] = phases["station"].str.strip()
        phases["phase_polarity"] = phases["phase_polarity"].str.strip()
        phases["azimuth"] = phases["azimuth"].str.strip()
        phases["takeoff_angle"] = phases["takeoff_angle"].str.strip()
        phases = phases[phases["distance_km"] != ""]
        phases = phases[phases["location_residual_s"].abs() < 9.99]
        phases = phases[~((phases["location_weight"] == 0) & (phases["location_residual_s"] == 0))]
        phases["location"] = phases["location"].apply(lambda x: x if x != "--" else "")

        # %% save all events
        tmp_name = ".".join(phase_filename.split(".")[:2])
        events.to_csv(f"{result_path}/catalog/{tmp_name}.event.csv", index=False)
        output_fs.put(f"{result_path}/catalog/{tmp_name}.event.csv", f"{dataset_path}/{tmp_name}.event.csv")
        phases.to_csv(f"{result_path}/catalog/{tmp_name}.phase_raw.csv", index=False)
        output_fs.put(f"{result_path}/catalog/{tmp_name}.phase_raw.csv", f"{dataset_path}/{tmp_name}.phase_raw.csv")

        # %%
        phases_ps = []
        event_ids = []
        for (event_id, network, station), picks in phases.groupby(["event_id", "network", "station"]):
            if len(picks) >= 2:
                phase_type = picks["phase_type"].unique()
                if ("P" in phase_type) and ("S" in phase_type):
                    phases_ps.append(picks)
                    event_ids.append(event_id)
            if len(picks) >= 3:
                print(event_id, network, station, len(picks))
        if len(phases_ps) == 0:
            continue
        phases_ps = pd.concat(phases_ps)
        # events = events[events.event_id.isin(event_ids)]
        phases = phases[phases.event_id.isin(event_ids)]

        # %%
        phases_ps.to_csv(f"{result_path}/catalog/{tmp_name}.phase_ps.csv", index=False)
        phases.to_csv(f"{result_path}/catalog/{tmp_name}.phase.csv", index=False)
        output_fs.put(f"{result_path}/catalog/{tmp_name}.phase_ps.csv", f"{dataset_path}/{tmp_name}.phase_ps.csv")
        output_fs.put(f"{result_path}/catalog/{tmp_name}.phase.csv", f"{dataset_path}/{tmp_name}.phase.csv")

        # year, month = phase_filename.split("/")[-1].split(".")[0:2]
        # if not os.path.exists(f"{result_path}/catalog/{year}"):
        #     os.makedirs(f"{result_path}/catalog/{year}")
        # events.to_csv(f"{result_path}/catalog/{year}/{year}_{month}.event.csv", index=False)
        # phases_ps.to_csv(f"{result_path}/catalog/{year}/{year}_{month}.phase_ps.csv", index=False)
        # phases.to_csv(f"{result_path}/catalog/{year}/{year}_{month}.phase.csv", index=False)


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    # years = range(2023, 2024)[::-1]
    years = range(1966, 2024)[::-1]
    ncpu = len(years)
    with ctx.Pool(processes=ncpu) as pool:
        pool.map(process, years)

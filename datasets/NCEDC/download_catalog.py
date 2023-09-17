# %%
import gzip
import os
import re
import shutil
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path

import fsspec
import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm

# %%
protocol = "sftp"
host = "range.geo.berkeley.edu"
username = "zhuwq"
keyfile = "/Users/weiqiang/.ssh/id_rsa"
fs = fsspec.filesystem(protocol, host=host, username=username, key_filename=keyfile)
fs.ls(".")

# %%
catalog_path = Path("../catalog/phase2k")
station_path = Path("../station")
waveform_path = Path("../waveform/")
dataset_path = Path("./dataset")
if not dataset_path.exists():
    dataset_path.mkdir()
if not (dataset_path / "catalog").exists():
    (dataset_path / "catalog").mkdir()


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
event_scale_factors = {
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
    "back_azimuth": (91, 94),
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

phase_scale_factors = {
    "second_of_p_arrival": 100,
    "second_of_s_arrival": 100,
    "distance_km": 10,
}


def read_event_line(line):
    event = {}
    for key, (start, end) in event_columns.items():
        if key in event_scale_factors:
            try:
                event[key] = float(line[start:end]) / event_scale_factors[key]
            except:
                print(key, line[start:end])
        else:
            event[key] = line[start:end]

    if event["seconds"] < 60:
        event[
            "event_time"
        ] = f"{event['year']}-{event['month']}-{event['day']}T{event['hour']}:{event['minute']}:{event['seconds']:06.3f}"
    else:
        tmp = datetime.fromisoformat(
            f"{event['year']}-{event['month']}-{event['day']}T{event['hour']}:{event['minute']}"
        )
        tmp += timedelta(seconds=event["seconds"])
        event["event_time"] = tmp.isoformat()

    event["latitude"] = event["latitude_deg"] + event["latitude_min"] / 60
    event["longitude"] = event["longitude_deg"] + event["longitude_min"] / 60
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
            if key in phase_scale_factors:
                # p_phase[key] = float(line[start:end].replace(" ", "0")) / phase_scale_factors[key]
                if len(line[start:end].strip()) == 0:
                    p_phase[key] = ""
                else:
                    p_phase[key] = float(line[start:end]) / phase_scale_factors[key]
            else:
                p_phase[key] = line[start:end]
        if (p_phase["second_of_p_arrival"] < 60) and (p_phase["second_of_p_arrival"] >= 0):
            p_phase[
                "phase_time"
            ] = f"{p_phase['year']}-{p_phase['month']}-{p_phase['day']}T{p_phase['hour']}:{p_phase['minute']}:{p_phase['second_of_p_arrival']:06.3f}"
        else:
            tmp = datetime.fromisoformat(
                f"{p_phase['year']}-{p_phase['month']}-{p_phase['day']}T{p_phase['hour']}:{p_phase['minute']}"
            )
            tmp += timedelta(seconds=p_phase["second_of_p_arrival"])
            p_phase["phase_time"] = tmp.isoformat()
        p_phase["phase_polarity"] = p_phase["p_polarity"]
        p_phase["remark"] = p_phase["p_remark"]
        p_phase["phase_score"] = p_phase["p_weight_code"]
        p_phase["phase_type"] = "P"
        phases.append(p_phase)
    start, end = phase_columns["s_remark"]
    if len(line[start:end].strip()) > 0:
        s_phase = {}
        for key, (start, end) in phase_columns.items():
            if key in phase_scale_factors:
                # s_phase[key] = float(line[start:end].replace(" ", "0")) / phase_scale_factors[key]
                if len(line[start:end].strip()) == 0:
                    s_phase[key] = ""
                else:
                    s_phase[key] = float(line[start:end]) / phase_scale_factors[key]
                # try:
                #     s_phase[key] = float(line[start:end].replace(" ", "0")) / phase_scale_factors[key]
                # except:
                #     print(key, line[start:end])
            else:
                s_phase[key] = line[start:end]
        if (s_phase["second_of_s_arrival"] < 60) and (s_phase["second_of_s_arrival"] >= 0):
            s_phase[
                "phase_time"
            ] = f"{s_phase['year']}-{s_phase['month']}-{s_phase['day']}T{s_phase['hour']}:{s_phase['minute']}:{s_phase['second_of_s_arrival']:06.3f}"
        else:
            tmp = datetime.fromisoformat(
                f"{s_phase['year']}-{s_phase['month']}-{s_phase['day']}T{s_phase['hour']}:{s_phase['minute']}"
            )
            tmp += timedelta(seconds=s_phase["second_of_s_arrival"])
            s_phase["phase_time"] = tmp.isoformat()
        s_phase["remark"] = s_phase["s_remark"]
        s_phase["phase_score"] = s_phase["s_weight_code"]
        s_phase["phase_type"] = "S"
        phases.append(s_phase)

    return phases


# %%
for year in range(1966, 2024)[::-1]:
    for phase_file in sorted(list((catalog_path / f"{year}").glob("*.phase.Z")))[::-1]:
        # if (dataset_path / "catalog" / f"{phase_file.name[:-2-6]}.event.csv").exists():
        #     continue

        if not (dataset_path / "catalog" / phase_file.name[:-2]).exists():
            shutil.copy(phase_file, dataset_path / "catalog" / phase_file.name)
            os.system(f"uncompress {dataset_path / 'catalog' / phase_file.name}")

        with open(dataset_path / "catalog" / phase_file.name[:-2]) as f:
            lines = f.readlines()
        catalog = {}
        event_id = None
        for line in tqdm(lines, desc=phase_file.name):
            if len(line) > (180 + 114) / 2:
                if event_id is not None:
                    assert event["event_id"] == event_id
                    catalog[event_id] = {"event": event, "picks": picks}
                event = read_event_line(line)
                picks = []
            elif len(line) > (73 + 114) / 2:
                picks.extend(read_phase_line(line))
            else:
                event_id = line.strip().split(" ")[-1]

        events = []
        phases = []
        for event_id in catalog:
            events.append(catalog[event_id]["event"])
            phase = pd.DataFrame(catalog[event_id]["picks"])
            phase["event_id"] = event_id
            phases.append(phase)

        events = pd.DataFrame(events)
        if len(phases) == 0:
            continue
        phases = pd.concat(phases)
        events = events[["event_id", "event_time", "latitude", "longitude", "depth_km", "magnitude", "magnitude_type"]]
        events["event_id"] = events["event_id"].apply(lambda x: "nc" + x)
        events["event_time"] = events["event_time"].apply(lambda x: x + "+00:00")
        phases = phases[
            [
                "event_id",
                "network",
                "station",
                "location",
                "component",
                "phase_type",
                "phase_time",
                "phase_score",
                "phase_polarity",
                "remark",
                "distance_km",
                "back_azimuth",
                "takeoff_angle",
            ]
        ]
        phases["event_id"] = phases["event_id"].apply(lambda x: "nc" + x)
        phases["phase_time"] = phases["phase_time"].apply(lambda x: x + "+00:00")
        phases["network"] = phases["network"].str.strip()
        phases["station"] = phases["station"].str.strip()
        phases["phase_polarity"] = phases["phase_polarity"].str.strip()
        phases["back_azimuth"] = phases["back_azimuth"].str.strip()
        phases["takeoff_angle"] = phases["takeoff_angle"].str.strip()
        phases = phases[phases["distance_km"] != ""]
        phases["location"] = phases["location"].apply(lambda x: x if x != "--" else "")

        # %%
        phases_ps = []
        for (event_id, network, station), picks in phases.groupby(["event_id", "network", "station"]):
            if len(picks) >= 2:
                phase_type = picks["phase_type"].unique()
                if "P" in phase_type and "S" in phase_type:
                    phases_ps.append(picks)
            if len(picks) >= 3:
                print(event_id, network, station, len(picks))
        if len(phases_ps) == 0:
            continue
        phases_ps = pd.concat(phases_ps)
        phases = phases_ps

        # %%
        events.to_csv(dataset_path / "catalog" / f"{phase_file.name[:-2-6]}.event.csv", index=False)
        phases.to_csv(dataset_path / "catalog" / f"{phase_file.name[:-2]}.csv", index=False)
    # %%

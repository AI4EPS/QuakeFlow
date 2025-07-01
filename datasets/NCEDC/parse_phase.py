# %%
import multiprocessing as mp
import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import fsspec
import pandas as pd
from tqdm import tqdm

# %%
input_protocol = "s3"
input_bucket = "ncedc-pds"
input_folder = "event_phases"
input_fs = fsspec.filesystem(input_protocol, anon=True)

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset"
output_folder = "NC/catalog"
output_fs = fsspec.filesystem(output_protocol, token=output_token)

# %%
result_path = "dataset"
os.makedirs(result_path, exist_ok=True)
os.makedirs(f"{result_path}/catalog_raw", exist_ok=True)

# %%
## https://ncedc.org/ftp/pub/doc/ncsn/shadow2000.pdf
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

phase_score_mappling = {"0": 1.0, "1": 0.75, "2": 0.5, "3": 0.25, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0}


def parse_event_line(line):
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
    event["event_id"] = "nc" + event["event_id"].strip()
    return event


def parse_phase_line(line):
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
        p_phase["phase_remark"] = p_phase["p_remark"].strip()[0].lower()
        p_phase["phase_score"] = phase_score_mappling[p_phase["p_weight_code"].strip()]
        p_phase["phase_type"] = "P"
        p_phase["time_residual"] = p_phase["p_travel_time_residual"]
        p_phase["phase_weight"] = p_phase["normalized_p_weight_actually_used"]
        p_phase["review_status"] = "manual" if p_phase["data_source_code"] == "J" else "automatic"
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
        s_phase["phase_remark"] = s_phase["s_remark"].strip()[0].lower()
        s_phase["phase_score"] = phase_score_mappling[s_phase["s_weight_code"].strip()]
        s_phase["phase_type"] = "S"
        s_phase["time_residual"] = s_phase["s_travel_time_residual"]
        s_phase["phase_weight"] = s_phase["s_weight_actually_used"]
        s_phase["review_status"] = "manual" if s_phase["data_source_code"] == "J" else "automatic"
        phases.append(s_phase)

    return phases


# %%
def process(file):

    # %%
    input_fs = fsspec.filesystem(input_protocol, anon=True)
    output_fs = fsspec.filesystem(output_protocol, token=output_token)

    filename = file.split("/")[-1]

    input_fs.get(file, f"{result_path}/catalog_raw/{filename}")
    os.system(f"uncompress -f {result_path}/catalog_raw/{filename}")

    with open(f"{result_path}/catalog_raw/{filename.replace('.Z', '')}") as f:
        lines = f.readlines()

    catalog = {}
    event_id = None
    for line in lines:
        if len(line) > (180 + 114) / 2:  # event_line
            if event_id is not None:
                assert event["event_id"] == event_id
                catalog[event_id] = {"event": event, "picks": picks}
            event = parse_event_line(line)
            picks = []
        elif len(line) > (73 + 114) / 2:  # phase_line
            picks.extend(parse_phase_line(line))
        else:
            event_id = "nc" + line.strip().split(" ")[-1]
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


    phase_columns = [
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
            "time_residual",
            "phase_weight",
            "review_status",
        ]
    event_columns = [
        "event_id",
        "time",
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
        "magnitude_type",
        # "event_type",
        # "quality",
    ]
    
    if len(phases) == 0:
        return

    events = pd.DataFrame(events)
    events = events[event_columns]
    events["time"] = pd.to_datetime(events["time"])
    # events["time"] = events["time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00")
    events["time"] = events["time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f"))

    phases = pd.concat(phases)
    phases = phases.reset_index(drop=True)
    phases = phases[phase_columns]
    phases["phase_time"] = pd.to_datetime(phases["phase_time"])
    phases["phase_time"] = phases["phase_time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    # phases["phase_time"] = phases["phase_time"].apply(lambda x: x + "+00:00")

    phases["network"] = phases["network"].str.strip()
    phases["station"] = phases["station"].str.strip()
    phases["phase_polarity"] = phases["phase_polarity"].str.strip()
    phases["azimuth"] = phases["azimuth"].str.strip()
    phases["takeoff_angle"] = phases["takeoff_angle"].str.strip()
    phases = phases[phases["distance_km"] != ""]
    phases = phases[phases["time_residual"].abs() < 9.99]
    phases = phases[~((phases["phase_weight"] == 0) & (phases["time_residual"] == 0))]
    phases["location"] = phases["location"].apply(lambda x: x if x != "--" else "")
    phases["phase_remark"] = phases["phase_remark"].apply(lambda x: "" if x in ["p", "s"] else x)

    # %% 
    # save picks to jday
    phases["tmp_time"] = pd.to_datetime(phases["phase_time"])
    phases["jday"] = phases["tmp_time"].dt.strftime("%j")
    phases["year"] = phases["tmp_time"].dt.strftime("%Y")
    for (year, jday), picks in phases.groupby(["year", "jday"]):
        if len(picks) == 0:
            continue

        os.makedirs(f"{result_path}/{year}/{jday}", exist_ok=True)

        # events.to_csv(f"{result_path}/event/{year}/{jday}.csv", index=False)
        # output_fs.put(
        #     f"{result_path}/event/{year}/{jday}.csv",
        #     f"{output_bucket}/{output_folder}/event/{year}/{jday}.csv",
        # )
        picks = picks[phase_columns]
        picks.to_csv(f"{result_path}/{year}/{jday}/phases.csv", index=False)
        output_fs.put(
            f"{result_path}/{year}/{jday}/phases.csv",
            f"{output_bucket}/{output_folder}/{year}/{jday}/phases.csv",
        )

    # %% save picks with P/S pairs
    phases_ps = []
    phases = phases.loc[phases.groupby(["event_id", "network", "station", "phase_type"])["phase_score"].idxmax()]
    for (event_id, network, station), picks in phases.groupby(["event_id", "network", "station"]):
        if len(picks) >= 2:
            phase_type = picks["phase_type"].unique()
            if ("P" in phase_type) and ("S" in phase_type):
                phases_ps.append(picks)
        if len(picks) >= 3:
            print(event_id, network, station, len(picks))

    if len(phases_ps) == 0:
        return
    phases_ps = pd.concat(phases_ps)

    for (year, jday), picks in phases_ps.groupby(["year", "jday"]):
        if len(picks) == 0:
            return

        picks = picks[phase_columns]
        picks.to_csv(f"{result_path}/{year}/{jday}/phases_ps.csv", index=False)
        output_fs.put(
            f"{result_path}/{year}/{jday}/phases_ps.csv",
            f"{output_bucket}/{output_folder}/{year}/{jday}/phases_ps.csv",
        )



if __name__ == "__main__":


    file_list = []
    for year in tqdm(sorted(input_fs.glob(f"{input_bucket}/{input_folder}/????"), reverse=True)):
        for file in sorted(input_fs.glob(f"{year}/*.phase.Z")):
            file_list.append(file)
        
        if year.split("/")[-1] <= "2024":
            break

    # for file in tqdm(file_list):
    #     process(file)

    ncpu = mp.cpu_count() - 1
    with ProcessPoolExecutor(max_workers=ncpu) as executor:
        
        futures = [executor.submit(process, file) for file in file_list]
        
        for future in tqdm(as_completed(futures), total=len(file_list)):
            result = future.result()
            if result is not None:
                print(result)

# %%

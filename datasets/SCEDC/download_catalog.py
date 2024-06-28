# %%
import multiprocessing as mp
import os
from datetime import datetime, timedelta
from pathlib import Path

import fsspec
import pandas as pd
from tqdm import tqdm

# %%
input_protocol = "s3"
input_bucket = "scedc-pds"
input_fs = fsspec.filesystem(input_protocol, anon=True)

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset"
output_fs = fsspec.filesystem(output_protocol, token=output_token)

# %%
catalog_path = f"{input_bucket}/event_phases"
station_path = f"{input_bucket}/FDSNstationXML"
waveform_path = f"{input_bucket}/continuous_waveforms/"
dataset_path = f"{output_bucket}/SC/catalog"

result_path = "dataset"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# %%
## https://scedc.caltech.edu/data/stp/STP_Manual_v1.01.pdf
# Defining a function to parse event location information
parse_event_time = lambda x: (
    datetime.strptime(":".join(x.split(":")[:-1]), "%Y/%m/%d,%H:%M") + timedelta(seconds=float(x.split(":")[-1]))
)


def parse_event_info(line):
    fields = line.split()
    try:
        event_info = {
            "event_id": "ci" + fields[0],
            "event_type": fields[1],
            # "event_time": parse_event_time(fields[3]),
            "time": parse_event_time(fields[3]),
            "latitude": float(fields[4]),
            "longitude": float(fields[5]),
            "depth_km": float(fields[6]),
            "magnitude": float(fields[7]),
            "magnitude_type": fields[8],
            "event_quality": float(fields[9]),
        }
    except:
        event_info = {
            "event_id": fields[0],
            "event_type": fields[1],
            # "date": fields[2],
            # "event_time": parse_event_time(fields[2]),
            "time": parse_event_time(fields[2]),
            "latitude": float(fields[3]),
            "longitude": float(fields[4]),
            "depth_km": float(fields[5]),
            "magnitude": float(fields[6]),
            "magnitude_type": fields[7],
            "event_quality": float(fields[8]),
        }

    return event_info


def parse_phase_pick(line, event_id, event_time):
    fields = line.split()
    phase_pick = {
        "network": fields[0],
        "station": fields[1],
        "channel": fields[2],
        "instrument": fields[2][:-1],
        "component": fields[2][-1],
        "location": fields[3] if fields[3] != "--" else "",
        "latitude": float(fields[4]),
        "longitude": float(fields[5]),
        "elevation_m": float(fields[6]),
        "depth_km": -float(fields[6]) / 1000,
        "phase_type": fields[7],
        "phase_polarity": fields[8],
        # "signal onset quality": fields[9],
        "phase_remark": fields[9],
        "phase_score": float(fields[10]),
        "distance_km": float(fields[11]),
        "phase_time": (event_time + timedelta(seconds=float(fields[12]))),
        "event_id": event_id,
    }
    if phase_pick["phase_polarity"][0] == ".":
        phase_pick["phase_polarity"] = "N"
    elif phase_pick["phase_polarity"][0] in ["c", "+", "u"]:
        phase_pick["phase_polarity"] = "U"
    elif phase_pick["phase_polarity"][0] in ["d", "-", "r"]:
        phase_pick["phase_polarity"] = "D"
    else:
        print(f"Unknown polarity: {phase_pick['phase_polarity']}")
        phase_pick["phase_polarity"] = "N"
    return phase_pick


# %% NCEDC
def parse(jday):
    events = []
    event_ids = []
    phases = []
    phases_ps = []

    for file in input_fs.glob(f"{jday}/*.phase"):
        with input_fs.open(file, "r") as fp:
            event_line = fp.readline().strip()
            nan_case = "0       1970/01/01,00:00:00.000"
            if event_line.startswith(nan_case):
                continue

            event = parse_event_info(event_line)
            phases_ = [parse_phase_pick(line.strip(), event["event_id"], event["time"]) for line in fp]
            if len(phases_) == 0:
                continue
        
        events.append(event)
        phases_ = pd.DataFrame(phases_)
        phases_ = phases_[phases_["phase_type"].isin(["P", "S"])]

        ## keep best picks
        phases_ = phases_.loc[phases_.groupby(["event_id", "network", "station", "phase_type"])["phase_score"].idxmax()]
        phases.append(phases_)

        ## keep P/S pairs
        for (event_id, network, station), picks in phases_.groupby(["event_id", "network", "station"]):
            if len(picks) >= 2:
                phase_type = picks["phase_type"].unique()
                if "P" in phase_type and "S" in phase_type:
                    phases_ps.append(picks)
                    event_ids.append(event_id)

            if len(picks) >= 3:
                print(jday, event_id, network, station, len(picks))

    # %% save all events
    if len(events) == 0:
        return None
    phases = pd.concat(phases)
    phases["phase_time"] = phases["phase_time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00")
    events = pd.DataFrame(events)
    events["time"] = events["time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00")

    if not os.path.exists(f"{result_path}/{jday.split('/')[-2]}"):
        os.makedirs(f"{result_path}/{jday.split('/')[-2]}", exist_ok=True)
    events.to_csv(f"{result_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.event.csv", index=False)
    output_fs.put(
        f"{result_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.event.csv",
        f"{dataset_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.event.csv",
    )
    phases.to_csv(f"{result_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.phase_raw.csv", index=False)
    output_fs.put(
        f"{result_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.phase_raw.csv",
        f"{dataset_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.phase_raw.csv",
    )

    # %% save only picks with P/S pairs
    if len(phases_ps) == 0:
        return None

    phases_ps = pd.concat(phases_ps)
    # events = events[events.event_id.isin(event_ids)]
    phases = phases[phases.event_id.isin(event_ids)]

    phases.to_csv(f"{result_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.phase.csv", index=False)
    phases_ps.to_csv(f"{result_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.phase_ps.csv", index=False)
    output_fs.put(
        f"{result_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.phase.csv",
        f"{dataset_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.phase.csv",
    )
    output_fs.put(
        f"{result_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.phase_ps.csv",
        f"{dataset_path}/{jday.split('/')[-2]}/{jday.split('/')[-1]}.phase_ps.csv",
    )


# %%
if __name__ == "__main__":
    file_list = []
    for year in tqdm(input_fs.glob(f"{catalog_path}/????")):
        if year.endswith("done"):
            continue

        for jday in input_fs.glob(f"{year}/????_???"):
            file_list.append(jday)

    file_list = sorted(file_list, reverse=True)
    ncpu = mp.cpu_count() * 2
    pbar = tqdm(total=len(file_list))
    processes = []
    with mp.get_context("spawn").Pool(ncpu) as pool:
        for f in file_list:
            proc = pool.apply_async(parse, args=(f,), callback=lambda _: pbar.update())
            processes.append(proc)
        for proc in processes:
            output = proc.get()
            if output is not None:
                print(output)

        #     pool.apply_async(parse, args=(f,), callback=lambda _: pbar.update())
        # pool.close()
        # pool.join()

    pbar.close()

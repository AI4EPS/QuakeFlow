# %%
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path

import fsspec
import pandas as pd
from tqdm import tqdm

# %%
protocol = "s3"
bucket = "scedc-pds"
fs = fsspec.filesystem(protocol, anon=True)

# %%
catalog_path = "event_phases"
station_path = "FDSNstationXML"
waveform_path = "continuous_waveforms/"
dataset_path = Path("./dataset")
if not dataset_path.exists():
    dataset_path.mkdir()
if not (dataset_path / "catalog").exists():
    (dataset_path / "catalog").mkdir()

# %%
## https://scedc.caltech.edu/data/stp/STP_Manual_v1.01.pdf
# Defining a function to parse event location information
parse_event_time = lambda x: datetime.strptime(":".join(x.split(":")[:-1]), "%Y/%m/%d,%H:%M") + timedelta(
    seconds=float(x.split(":")[-1])
)


def parse_event_info(line):
    fields = line.split()
    try:
        event_info = {
            "event_id": "ci" + fields[0],
            "event_type": fields[1],
            "event_time": parse_event_time(fields[3]),
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
            "event_time": parse_event_time(fields[2]),
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
        "phase_time": event_time + timedelta(seconds=float(fields[12])),
        "event_id": event_id,
    }
    if phase_pick["phase_polarity"][0] == ".":
        phase_pick["phase_polarity"] = "N"
    elif phase_pick["phase_polarity"][0] == "c":
        phase_pick["phase_polarity"] = "D"
    elif phase_pick["phase_polarity"][0] == "d":
        phase_pick["phase_polarity"] = "U"
    else:
        print(f"Unknown polarity: {phase_pick['phase_polarity']}")
        phase_pick["phase_polarity"] = "N"
    return phase_pick


# %%
def parse(jday):
    phases_ps = []
    events = []
    for file in fs.glob(f"{jday}/*.phase"):
        with fs.open(file, "r") as f:
            event_line = f.readline().strip()
            nan_case = "0       1970/01/01,00:00:00.000"
            if event_line.startswith(nan_case):
                continue

            event = parse_event_info(event_line)
            phases = [parse_phase_pick(line.strip(), event["event_id"], event["event_time"]) for line in f]
            if len(phases) == 0:
                continue

        events.append(event)
        phases = pd.DataFrame(phases)
        phases = phases[phases["phase_type"].isin(["P", "S"])]

        ## keep best picks
        phases = phases.loc[phases.groupby(["event_id", "network", "station", "phase_type"])["phase_score"].idxmax()]
        ## keep P/S pairs
        for (event_id, network, station), picks in phases.groupby(["event_id", "network", "station"]):
            if len(picks) >= 2:
                phase_type = picks["phase_type"].unique()
                if "P" in phase_type and "S" in phase_type:
                    phases_ps.append(picks)

            if len(picks) >= 3:
                print(jday, event_id, network, station, len(picks))
                print(picks)

    if len(phases_ps) == 0:
        return 0
    phases_ps = pd.concat(phases_ps)
    events = pd.DataFrame(events)

    if not (dataset_path / f"catalog/{jday.split('/')[-2]}").exists():
        (dataset_path / f"catalog/{jday.split('/')[-2]}").mkdir()

    # add timezone utc to phase_time
    phases_ps["phase_time"] = phases_ps["phase_time"].dt.tz_localize("UTC")
    events["event_time"] = events["event_time"].dt.tz_localize("UTC")
    phases_ps.to_csv(dataset_path / f"catalog/{jday.split('/')[-2]}/{jday.split('/')[-1]}.phase.csv", index=False)
    events.to_csv(dataset_path / f"catalog/{jday.split('/')[-2]}/{jday.split('/')[-1]}.event.csv", index=False)


# %%
if __name__ == "__main__":
    file_list = []
    for year in tqdm(fs.glob(f"{bucket}/{catalog_path}/????")):
        if year.endswith("done"):
            continue

        for jday in fs.glob(f"{year}/????_???"):
            file_list.append(jday)

    file_list = sorted(file_list, reverse=True)
    ncpu = mp.cpu_count()
    pbar = tqdm(total=len(file_list))
    with mp.get_context("spawn").Pool(ncpu) as pool:
        for f in file_list:
            pool.apply_async(parse, args=(f,), callback=lambda _: pbar.update())
        pool.close()
        pool.join()

    pbar.close()

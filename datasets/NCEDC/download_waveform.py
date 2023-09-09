# %%
import multiprocessing as mp
import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import obspy
import pandas as pd
from tqdm.auto import tqdm

matplotlib.use("Agg")

warnings.filterwarnings("ignore")

os.environ["OPENBLAS_NUM_THREADS"] = "2"

# %%
catalog_path = Path("dataset/catalog")
station_path = Path("../station")
waveform_path = Path("../waveform/")
dataset_path = Path("./dataset")
if not dataset_path.exists():
    dataset_path.mkdir()


# %%
def cut_data(event, phases):
    arrival_time = phases.loc[event.event_id, "phase_time"].min()
    begin_time = arrival_time - pd.Timedelta(seconds=30)
    end_time = arrival_time + pd.Timedelta(seconds=90)

    for _, pick in phases.loc[event.event_id].iterrows():
        tmp_path = (
            dataset_path
            / "waveform"
            / f"{event.event_time.year}"
            / f"{event.event_time.year}.{event.event_time.dayofyear:03d}"
            / f"{event.event_id}"
        )

        # if (tmp_path / f"{pick.network}.{pick.station}.{pick.location}.{pick.component[:-1]}.mseed").exists():
        #     continue

        inv = obspy.read_inventory(
            station_path / f"{pick.network}.info" / f"{pick.network}.FDSN.xml" / f"{pick.network}.{pick.station}.xml"
        )

        try:
            st = obspy.read(
                waveform_path
                / f"{pick.network}"
                / f"{arrival_time.year}"
                / f"{arrival_time.year}.{arrival_time.dayofyear:03d}"
                / f"{pick.station}.{pick.network}.{pick.component[:-1]}?.{pick.location}.?.{arrival_time.year}.{arrival_time.dayofyear:03d}"
            )
        except Exception as e:
            print(e)
            continue

        st.trim(obspy.UTCDateTime(begin_time), obspy.UTCDateTime(end_time))
        st.merge(fill_value="latest")
        st.remove_sensitivity(inv)

        if len(st) == 0:
            print(f"{event.event_id}.{pick.network}.{pick.station}.{pick.location}.{pick.component[:-1]} is empty")
            continue

        if not tmp_path.exists():
            tmp_path.mkdir(parents=True)
        st.write(
            tmp_path / f"{pick.network}.{pick.station}.{pick.location}.{pick.component[:-1]}.mseed",
            format="MSEED",
        )

        tmp_path = (
            dataset_path
            / "figure"
            / f"{event.event_time.year}"
            / f"{event.event_time.year}.{event.event_time.dayofyear:03d}"
            / f"{event.event_id}"
        )
        if not tmp_path.exists():
            tmp_path.mkdir(parents=True)
        st.plot(outfile=tmp_path / f"{pick.network}.{pick.station}.{pick.location}.{pick.component[:-1]}.png")


# %%
if __name__ == "__main__":
    ncpu = 32
    event_list = sorted(list(catalog_path.glob("*.event.csv")))[::-1]
    start_year = "1966"
    end_year = "2022"
    tmp = []
    for event_file in event_list:
        if event_file.name.split(".")[0] >= start_year and event_file.name.split(".")[0] <= end_year:
            tmp.append(event_file)
    event_list = sorted(tmp)[::-1]

    for event_file in event_list:
        print(event_file)
        events = pd.read_csv(event_file, parse_dates=["event_time"])
        phases = pd.read_csv(
            event_file.parent / (event_file.name.replace("event.csv", "phase.csv")),
            parse_dates=["phase_time"],
            keep_default_na=False,
        )
        phases.set_index("event_id", inplace=True)

        events = events[events.event_id.isin(phases.index)]
        pbar = tqdm(events, total=len(events))
        with mp.get_context("spawn").Pool(ncpu) as p:
            for _, event in events.iterrows():
                p.apply_async(cut_data, args=(event, phases.loc[event.event_id]), callback=lambda _: pbar.update(1))
            p.close()
            p.join()
        pbar.close()

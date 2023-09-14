# %%
import multiprocessing as mp
import os
import warnings
from pathlib import Path

import fsspec
import matplotlib
import obspy
import pandas as pd
from tqdm.auto import tqdm

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
os.environ["OPENBLAS_NUM_THREADS"] = "2"


# %%
protocol = "s3"
bucket = "scedc-pds"
fs = fsspec.filesystem(protocol, anon=True)


# %%
catalog_path = "event_phases"
station_path = "FDSNstationXML"
waveform_path = "continuous_waveforms"
dataset_path = Path("./dataset")
if not dataset_path.exists():
    dataset_path.mkdir()
if not (dataset_path / "catalog").exists():
    (dataset_path / "catalog").mkdir()


# %%
def cut_data(event_file, phase_file):
    events = pd.read_csv(event_file, parse_dates=["event_time"])
    phases = pd.read_csv(phase_file, parse_dates=["phase_time"], keep_default_na=False)
    phases = phases.loc[
        phases.groupby(["event_id", "network", "station", "location", "instrument"]).phase_time.idxmin()
    ]
    phases.set_index("event_id", inplace=True)

    for _, event in events.iterrows():
        arrival_time = phases.loc[event.event_id, "phase_time"].min()
        begin_time = arrival_time - pd.Timedelta(seconds=30)
        end_time = arrival_time + pd.Timedelta(seconds=90)

        for _, pick in phases.loc[event.event_id].iterrows():
            outfile_path = (
                dataset_path
                / "waveform"
                / f"{event.event_time.year}"
                / f"{event.event_time.year}.{event.event_time.dayofyear:03d}"
                / f"{event.event_id}"
            )

            if (outfile_path / f"{pick.network}.{pick.station}.{pick.location}.{pick.instrument}.mseed").exists():
                continue

            # if pick.network == "CI":
            #     inv_path = f"{bucket}/{station_path}/{pick.network}/{pick.network}_{pick.station}.xml"
            # else:
            #     inv_path = f"{bucket}/{station_path}/unauthoritative-XML/{pick.network}.{pick.station}.xml"

            inv_path = f"{bucket}/{station_path}/{pick.network}/{pick.network}_{pick.station}.xml"
            if not fs.exists(inv_path):
                inv_path = f"{bucket}/{station_path}/unauthoritative-XML/{pick.network}.{pick.station}.xml"
            if not fs.exists(inv_path):
                continue
            with fs.open(inv_path) as f:
                inv = obspy.read_inventory(f)

            # if not fs.exists(inv_path):
            #     # print(f"{inv_path} not exists")
            #     continue
            # # print(inv_path)
            # with fs.open(inv_path) as f:
            #     inv = obspy.read_inventory(f)

            location_code = pick.location if pick.location else "__"
            begin_mseed_path = f"{bucket}/{waveform_path}/{begin_time.year}/{begin_time.year}_{begin_time.dayofyear:03d}/{pick.network}{pick.station:_<5}{pick.instrument}?_{location_code}{begin_time.year}{begin_time.dayofyear:03d}.ms"
            end_mseed_path = f"{bucket}/{waveform_path}/{end_time.year}/{end_time.year}_{end_time.dayofyear:03d}/{pick.network}{pick.station:_<5}{pick.instrument}?_{location_code}{end_time.year}{end_time.dayofyear:03d}.ms"
            try:
                st = obspy.Stream()
                for mseed_path in set([begin_mseed_path, end_mseed_path]):
                    for mseed in fs.glob(mseed_path):
                        with fs.open(mseed) as f:
                            st += obspy.read(f)
            except Exception as e:
                print(e)
                continue

            st.trim(obspy.UTCDateTime(begin_time), obspy.UTCDateTime(end_time))
            try:
                st.merge(fill_value="latest")
            except Exception as e:
                print(e)
                continue
            try:
                st.remove_sensitivity(inv)
            except Exception as e:
                print(e)
                continue

            if len(st) == 0:
                # print(f"{event.event_id}.{pick.network}.{pick.station}.{pick.location}.{pick.component[:-1]} is empty")
                continue

            if not outfile_path.exists():
                outfile_path.mkdir(parents=True)
            st.write(
                # outfile_path / f"{pick.network}.{pick.station}.{pick.location}.{pick.component[:-1]}.mseed",
                outfile_path / f"{pick.network}.{pick.station}.{pick.location}.{pick.instrument}.mseed",
                format="MSEED",
            )

            outfile_path = (
                dataset_path
                / "figure"
                / f"{event.event_time.year}"
                / f"{event.event_time.year}.{event.event_time.dayofyear:03d}"
                / f"{event.event_id}"
            )
            if not outfile_path.exists():
                outfile_path.mkdir(parents=True)

            # st.plot(outfile=outfile_path / f"{pick.network}.{pick.station}.{pick.location}.{pick.component[:-1]}.png")
            st.plot(outfile=outfile_path / f"{pick.network}.{pick.station}.{pick.location}.{pick.instrument}.png")

    return 0


# %%
if __name__ == "__main__":
    # %%
    event_list = []
    start_year = "1900"
    end_year = "2021"
    for year in (dataset_path / "catalog").glob("????"):
        if year.name < start_year or year.name > end_year:
            continue
        for jday in year.glob("????_???.event.csv"):
            event_list.append(jday)

    event_list = sorted(event_list)[::-1]

    pbar = tqdm(event_list, total=len(event_list))
    ncpu = mp.cpu_count() * 2
    with mp.get_context("spawn").Pool(ncpu) as p:
        for event_file in event_list:
            phase_file = event_file.parent / (event_file.name.replace("event.csv", "phase.csv"))
            p.apply_async(cut_data, args=(event_file, phase_file), callback=lambda _: pbar.update(1))
        p.close()
        p.join()
    pbar.close()

# %%

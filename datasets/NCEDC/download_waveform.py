# %%
import multiprocessing as mp
import os
import warnings
from glob import glob
from pathlib import Path

import fsspec
import matplotlib
import matplotlib.pyplot as plt
import obspy
import pandas as pd
from tqdm.auto import tqdm

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# %%
root_path = "."
catalog_path = f"dataset/catalog"
station_path = f"{root_path}/station"
waveform_path = f"{root_path}/waveform"
# catalog_path = "/quakeflow_dataset/NC/catalog"
# station_path = "/quakeflow_dataset/NC/FDSNstationXML"
# waveform_path = "/ncedc-pds/continuous_waveforms"

protocol = "gs"
token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
bucket = "quakeflow_dataset"
result_path = f"{bucket}/NC"


# %%
def cut_data(event, phases):
    arrival_time = phases.loc[event.event_id, "phase_time"].min()
    begin_time = arrival_time - pd.Timedelta(seconds=35)
    end_time = arrival_time + pd.Timedelta(seconds=95)

    fs = fsspec.filesystem(protocol=protocol, token=token)

    for _, pick in phases.loc[event.event_id].iterrows():
        outfile_path = f"{result_path}/waveform_mseed/{event.time.year}/{event.time.year}.{event.time.dayofyear:03d}/{event.event_id}_{begin_time.strftime('%Y%m%d%H%M%S')}"
        outfile_name = f"{pick.network}.{pick.station}.{pick.location}.{pick.instrument}.mseed"
        if fs.exists(f"{outfile_path}/{outfile_name}"):
            continue

        inv_path = f"{station_path}/{pick.network}.info/{pick.network}.FDSN.xml/{pick.network}.{pick.station}.xml"
        if not os.path.exists(inv_path):
            continue
        inv = obspy.read_inventory(str(inv_path))

        begin_mseed_path = f"{waveform_path}/{pick.network}/{begin_time.year}/{begin_time.year}.{begin_time.dayofyear:03d}/{pick.station}.{pick.network}.{pick.instrument}?.{pick.location}.?.{begin_time.year}.{begin_time.dayofyear:03d}"
        end_mseed_path = f"{waveform_path}/{pick.network}/{end_time.year}/{end_time.year}.{end_time.dayofyear:03d}/{pick.station}.{pick.network}.{pick.instrument}?.{pick.location}.?.{end_time.year}.{end_time.dayofyear:03d}"
        try:
            st = obspy.Stream()
            for mseed_path in set([begin_mseed_path, end_mseed_path]):
                st += obspy.read(str(mseed_path))
        except Exception as e:
            print(e)
            continue

        if len(st) == 0:
            continue

        try:
            st.merge(fill_value="latest")
            st.remove_sensitivity(inv)
            st.detrend("constant")
            st.trim(obspy.UTCDateTime(begin_time), obspy.UTCDateTime(end_time), pad=True, fill_value=0)
        except Exception as e:
            print(e)
            continue

        # float64 to float32
        for tr in st:
            tr.data = tr.data.astype("float32")

        if not fs.exists(outfile_path):
            fs.makedirs(outfile_path)

        with fs.open(f"{outfile_path}/{outfile_name}", "wb") as f:
            st.write(f, format="MSEED")

        # st.plot(outfile=outfile_path / f"{pick.network}.{pick.station}.{pick.location}.{pick.instrument}.png")
        # outfile_path = (
        #     f"{result_path}/figure/{event.time.year}/{event.time.year}.{event.time.dayofyear:03d}/{event.event_id}"
        # )
        # if not os.path.exists(outfile_path):
        #     os.makedirs(outfile_path)
        # st.plot(outfile=f"{outfile_path}/{pick.network}.{pick.station}.{pick.location}.{pick.instrument}.png")

    return 0


# %%
if __name__ == "__main__":
    ncpu = min(mp.cpu_count() * 4, 32)
    event_list = sorted(list(glob(f"{catalog_path}/*.event.csv")))[::-1]
    start_year = "1967"
    end_year = "2023"
    tmp = []
    for event_file in event_list:
        if (
            event_file.split("/")[-1].split(".")[0] >= start_year
            and event_file.split("/")[-1].split(".")[0] <= end_year
        ):
            tmp.append(event_file)
    event_list = sorted(tmp, reverse=True)

    for event_file in event_list:
        print(event_file)
        events = pd.read_csv(event_file, parse_dates=["time"])
        phases = pd.read_csv(
            f"{event_file.replace('event.csv', 'phase.csv')}",
            parse_dates=["phase_time"],
            keep_default_na=False,
        )
        phases = phases.loc[
            phases.groupby(["event_id", "network", "station", "location", "instrument"]).phase_time.idxmin()
        ]
        phases.set_index("event_id", inplace=True)

        events = events[events.event_id.isin(phases.index)]
        pbar = tqdm(events, total=len(events))
        with mp.get_context("spawn").Pool(ncpu) as p:
            for _, event in events.iterrows():
                p.apply_async(cut_data, args=(event, phases.loc[event.event_id]), callback=lambda _: pbar.update(1))
            p.close()
            p.join()
        pbar.close()

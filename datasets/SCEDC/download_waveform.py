# %%
import multiprocessing as mp
import os
import time
import warnings
from glob import glob

import fsspec
import matplotlib
import obspy
import pandas as pd
from tqdm.auto import tqdm

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
os.environ["OPENBLAS_NUM_THREADS"] = "2"


# %%
input_protocol = "s3"
input_bucket = "scedc-pds"

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset"

result_path = f"{output_bucket}/SC"

# %%
catalog_path = f"{result_path}/catalog"
station_path = f"FDSNstationXML"
waveform_path = f"continuous_waveforms"


# %%
input_fs = fsspec.filesystem(input_protocol, anon=True)
output_fs = fsspec.filesystem(output_protocol, token=output_token)


# %%
def cut_data(event, phases):
    if event.event_id not in phases.index:
        return 0

    arrival_time = phases.loc[[event.event_id], "phase_time"].min()
    begin_time = arrival_time - pd.Timedelta(seconds=35)
    end_time = arrival_time + pd.Timedelta(seconds=95)

    for _, pick in phases.loc[[event.event_id]].iterrows():
        outfile_path = f"{result_path}/waveform_mseed/{event.time.year}/{event.time.year}.{event.time.dayofyear:03d}/{event.event_id}_{begin_time.strftime('%Y%m%d%H%M%S')}"
        outfile_name = f"{pick.network}.{pick.station}.{pick.location}.{pick.instrument}.mseed"
        if output_fs.exists(f"{outfile_path}/{outfile_name}"):
            continue

        ########### NCEDC ###########
        # inv_path = f"{station_path}/{pick.network}.info/{pick.network}.FDSN.xml/{pick.network}.{pick.station}.xml"
        # if not os.path.exists(inv_path):
        #     continue
        # inv = obspy.read_inventory(str(inv_path))

        # begin_mseed_path = f"{waveform_path}/{pick.network}/{begin_time.year}/{begin_time.year}.{begin_time.dayofyear:03d}/{pick.station}.{pick.network}.{pick.instrument}?.{pick.location}.?.{begin_time.year}.{begin_time.dayofyear:03d}"
        # end_mseed_path = f"{waveform_path}/{pick.network}/{end_time.year}/{end_time.year}.{end_time.dayofyear:03d}/{pick.station}.{pick.network}.{pick.instrument}?.{pick.location}.?.{end_time.year}.{end_time.dayofyear:03d}"

        ########### SCEDC ###########
        inv_path = f"{input_bucket}/{station_path}/{pick.network}/{pick.network}_{pick.station}.xml"
        if not input_fs.exists(inv_path):
            inv_path = f"{input_bucket}/{station_path}/unauthoritative-XML/{pick.network}_{pick.station}.xml"
        if not input_fs.exists(inv_path):
            # print(f"{inv_path} not exists")
            continue

        with input_fs.open(inv_path) as f:
            inv = obspy.read_inventory(f)

        location_code = pick.location if pick.location else "__"
        begin_mseed_path = f"{input_bucket}/{waveform_path}/{begin_time.year}/{begin_time.year}_{begin_time.dayofyear:03d}/{pick.network}{pick.station:_<5}{pick.instrument}?_{location_code}{begin_time.year}{begin_time.dayofyear:03d}.ms"
        end_mseed_path = f"{input_bucket}/{waveform_path}/{end_time.year}/{end_time.year}_{end_time.dayofyear:03d}/{pick.network}{pick.station:_<5}{pick.instrument}?_{location_code}{end_time.year}{end_time.dayofyear:03d}.ms"

        #############################

        try:
            st = obspy.Stream()
            for mseed_path in set([begin_mseed_path, end_mseed_path]):
                for mseed in input_fs.glob(mseed_path):
                    with input_fs.open(mseed) as f:
                        st += obspy.read(f)
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

        # if not output_fs.exists(outfile_path):
        #     output_fs.makedirs(outfile_path)

        with output_fs.open(f"{outfile_path}/{outfile_name}", "wb") as f:
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
    mp.set_start_method("spawn")

    ncpu = min(mp.cpu_count() * 2, 32)
    # event_list = sorted(list(glob(f"{catalog_path}/*.event.csv")))[::-1]
    # fs = fsspec.filesystem(output_protocol, token=output_token)
    fs = output_fs
    event_list = sorted(list(fs.glob(f"{result_path}/catalog/????/*.event.csv")), reverse=False)
    start_year = "1999"
    end_year = "2023"
    tmp = []
    for event_file in event_list:
        if (
            ## NCEDC
            # event_file.split("/")[-1].split(".")[0] >= start_year
            # and event_file.split("/")[-1].split(".")[0] <= end_year
            ## SCEDC
            event_file.split("/")[-2] >= start_year
            and event_file.split("/")[-2] <= end_year
        ):
            tmp.append(event_file)
    event_list = sorted(tmp, reverse=False)

    for event_file in event_list:
        print(event_file)
        # events = pd.read_csv(event_file, parse_dates=["time"])
        with fs.open(event_file) as f:
            events = pd.read_csv(f, parse_dates=["time"])
        # phases = pd.read_csv(
        #     f"{event_file.replace('event.csv', 'phase.csv')}",
        #     parse_dates=["phase_time"],
        #     keep_default_na=False,
        # )
        with fs.open(event_file.replace("event.csv", "phase.csv")) as f:
            phases = pd.read_csv(f, parse_dates=["phase_time"], keep_default_na=False)
        phases = phases.loc[
            phases.groupby(["event_id", "network", "station", "location", "instrument"]).phase_time.idxmin()
        ]
        phases.set_index("event_id", inplace=True)

        events = events[events.event_id.isin(phases.index)]

        # for _, event in events.iterrows():
        #     cut_data(event, phases.loc[event.event_id])
        # raise

        pbar = tqdm(events, total=len(events))
        # with mp.get_context("spawn").Pool(ncpu) as p:
        with mp.Pool(ncpu) as p:
            for _, event in events.iterrows():
                p.apply_async(cut_data, args=(event, phases.loc[event.event_id]), callback=lambda _: pbar.update(1))
            p.close()
            p.join()
        pbar.close()

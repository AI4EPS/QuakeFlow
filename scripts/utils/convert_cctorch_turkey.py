# %%
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import shutil
import obspy
from tqdm import tqdm
import multiprocessing as mp

# %%
catalog = pd.read_csv("../EikoLoc/eikoloc_catalog.csv", parse_dates=["time"])

# %%
catalog["index"] = catalog["event_index"]
catalog = catalog.set_index("index")

# %%
picks = pd.read_csv("../EikoLoc/gamma_picks.csv", parse_dates=["phase_time"])

# %%
picks["index"] = picks["event_index"]

# %%
picks = picks.set_index("event_index")

# %%
# picks[["network", "station", "location", "channel"]] = picks["station_id"].str.split(".", expand=True)
# picks["phase_time"] = picks["phase_time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

# %%
waveform_path = Path("../waveforms")
output_path = Path("waveforms")
if not output_path.exists():
    output_path.mkdir()

# %%
def save_mseed(f, year, jday):
    try:
        meta = obspy.read(f)
    except:
        return
    
    date = datetime.strptime(f"{year}-{jday}", "%Y-%j")
    month, day = date.strftime("%m"), date.strftime("%d")

    meta = meta.merge(fill_value="latest")
    min_time = min([tr.stats.starttime for tr in meta])
    max_time = max([tr.stats.endtime for tr in meta])
    meta = meta.slice(starttime=min_time, endtime=max_time)
    for trace in meta:
        station_id = trace.get_id()
        network, station, location, channel = station_id.split(".")
        for hour in range(24):
            starttime = obspy.UTCDateTime(f"{year}-{month}-{day}T{hour:02d}:00:00.000Z")
            endtime = obspy.UTCDateTime(f"{year}-{month}-{day}T{hour:02d}:00:00.000Z")+3600
            trace_hour = trace.slice(starttime=starttime, endtime=endtime)
            if len(trace_hour.data) > 0:
                trace_hour.write(output_path / f"{year}-{jday}" / f"{hour:02d}" / f"{station_id}.mseed", format="MSEED")
            # except Exception as e:
            #     print(f"{year}-{month}-{day}T{hour:02d}:00:00.000Z")
            #     print(obspy.UTCDateTime(f"{year}-{month}-{day}T{hour:02d}:00:00.000Z"))
            #     print(e)
            #     print(min_time, max_time, f"{year}-{month}-{day}T{hour:02d}")
            #     raise
    print("Finish: ", output_path / f"{year}-{jday}", f)
    

# %%
for day_dir in waveform_path.iterdir():

    year = datetime.fromisoformat(day_dir.name).strftime("%Y")
    jday = datetime.fromisoformat(day_dir.name).strftime("%j")
    
    if not (output_path / f"{year}-{jday}").exists():
        (output_path / f"{year}-{jday}").mkdir()
    for hour in range(24):
        if not (output_path / f"{year}-{jday}" / f"{hour:02d}").exists():
            (output_path / f"{year}-{jday}" / f"{hour:02d}").mkdir()

    file_list = set()
    mseeds = list(day_dir.rglob("*.mseed_[ENZ].mseed"))
    for x in mseeds:
        file_name = str(x)
        file_name = "_".join(file_name.split("_")[:-1] + ["?.mseed"])
        file_list.add(file_name)

    mseeds = list(day_dir.rglob("*_tdvms_?.mseed"))
    for x in mseeds:
        file_list.add(str(x))
    
    ncpu = mp.cpu_count()//2
    with mp.Pool(ncpu) as p:
        p.starmap(save_mseed, [(f, year, jday) for f in file_list])



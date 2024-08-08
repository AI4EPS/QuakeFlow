# %%
import json
import os
import warnings
from glob import glob

import obspy
import obspy.taup
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# %%
os.system(
    "curl -L -O https://github.com/AI4EPS/EPS207_Observational_Seismology/releases/download/fm_data/fm_data.zip && unzip -q -o fm_data.zip"
)

# %%
data_path = "fm_data"
os.system("mv fm_data/picks fm_data/picks_raw")

# %%
with open(f"{data_path}/stations.json", "r") as f:
    stations = json.load(f)

stations = pd.DataFrame.from_dict(stations, orient="index")
events = pd.read_csv(f"{data_path}/catalog.csv", parse_dates=["time"])
events["time"] = events["time"].dt.tz_localize(None)
events.set_index("event_id", inplace=True)

# %%
model = obspy.taup.TauPyModel("iasp91")
max_timediff = 2.0

plotting = False
if plotting:
    plt.figure(figsize=(10, 10))
for file in tqdm(list(glob(f"{data_path}/picks_raw/*.csv"))):
    picks = pd.read_csv(file, parse_dates=["phase_time"])
    event_id = file.split("/")[-1].replace(".csv", "")
    evot, mag, evla, evlo, evdp, x, y, z = events.loc[
        event_id, ["time", "magnitude", "latitude", "longitude", "depth_km", "x_km", "y_km", "z_km"]
    ].to_numpy()

    keep_idx = []
    for i, pick in picks.iterrows():
        stlo, stla = stations.loc[pick["station_id"], ["longitude", "latitude"]].to_numpy()
        epicdist = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo)[0] / 1000
        prac_phase_time = (pick["phase_time"] - evot).total_seconds()

        phase_type = pick["phase_type"]
        if phase_type == "P":
            arrivals = model.get_travel_times_geo(max(0, evdp), evla, evlo, stla, stlo, phase_list=["p", "P"])
            if plotting:
                plt.scatter(prac_phase_time, epicdist, color="b")
        else:
            arrivals = model.get_travel_times_geo(max(0, evdp), evla, evlo, stla, stlo, phase_list=["s", "S"])
            if plotting:
                plt.scatter(prac_phase_time, epicdist, color="r")

        theo_phase_time = arrivals[0].time
        if abs(theo_phase_time - prac_phase_time) < max_timediff:
            keep_idx.append(i)
        else:
            if plotting:
                plt.scatter(prac_phase_time, epicdist, color="g")

    picks_ = picks.iloc[keep_idx]
    picks_["event_index"] = event_id

    try:
        picks_.to_csv(f"{data_path}/picks/{event_id}.csv", index=None)
    except:
        os.mkdir(f"{data_path}/picks")
        picks_.to_csv(f"{data_path}/picks/{event_id}.csv", index=None)

    if plotting:
        plt.xlabel("Time (s)")
        plt.ylabel("Epicentral distance (km)")
        plt.title(event_id)
        plt.show()
        plt.close()

# %%
picks = []
for file in tqdm(list(glob(f"{data_path}/picks/*.csv"))):
    picks.append(pd.read_csv(file))
picks = pd.concat(picks, ignore_index=True)
picks.to_csv(f"{data_path}/picks.csv", index=None)


# %%

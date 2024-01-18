# %%
import redis
import json
import time
from random import randint
import fsspec
import pandas as pd
import obspy
import matplotlib.pyplot as plt
import numpy as np

# %%
PROTOCAL = "gs"
BUCKET = "quakeflow_share"
FOLDER = "demo/obspy"

stations = pd.read_csv(f"{PROTOCAL}://{BUCKET}/{FOLDER}/stations.csv")
fs = fsspec.filesystem(PROTOCAL)
mseeds = fs.glob(f"{BUCKET}/{FOLDER}/waveforms/*/*.mseed")

# %%
waveforms = {}
station_ids = []
min_timestamp = None
print("Reading waveforms: ", end="", flush=True)
for i, mseed in enumerate(mseeds):
    print(mseed.split("/")[-1], end=" ", flush=True)
    with fs.open(mseed, "rb") as f:
        st = obspy.read(f)
        st = st.merge(fill_value="latest")
        st = st.resample(100)
        tr = st[0]
        if min_timestamp is None:
            min_timestamp = tr.times("timestamp")[0]
        waveforms[tr.id] = {
            "data": tr.data.tolist(),
            # "timestamp": (tr.times("timestamp") - min_timestamp).tolist(),
            "timestamp": tr.times("timestamp").tolist(),
        }
        station_ids.append(tr.id)
    if i > 40:
        break
print("\nFinished reading waveforms.", flush=True)
with open("station_ids.json", "w") as f:
    json.dump(station_ids, f)


# %%
r = redis.Redis(host="localhost", port=6379, db=0)

index = {x: 0 for x in station_ids}
while True:
    for i, sid in enumerate(station_ids):
        print(sid, end=" ", flush=True)
        window_size = randint(80, 120)
        r.xadd(
            sid,
            {
                "data": json.dumps(waveforms[sid]["data"][index[sid] : index[sid] + window_size]),
                "timestamp": json.dumps(waveforms[sid]["timestamp"][index[sid] : index[sid] + window_size]),
            },
        )
        r.xtrim(sid, maxlen=60000)
        index[sid] += window_size
    print()
    time.sleep(1)

# %%

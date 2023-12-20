# %%
import json
import logging
import threading
import time
from datetime import datetime
from random import randint

import fsspec
import obspy
import pandas as pd
import redis
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)


app = FastAPI()

# %%
PROTOCAL = "gs"
BUCKET = "quakeflow_share"
FOLDER = "demo"
REDIS_HOST = "quakeflow-redis-master.default.svc.cluster.local"
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    redis_client.ping()
except:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)


def replay():
    fs = fsspec.filesystem(PROTOCAL)
    mseeds = fs.glob(f"{BUCKET}/{FOLDER}/waveforms/????-???/??/*.mseed")

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

    index = {x: 0 for x in station_ids}
    while True:
        for i, sid in enumerate(station_ids):
            # print(sid, end=" ", flush=True)
            window_size = randint(80, 120)
            data = waveforms[sid]["data"][index[sid] : index[sid] + window_size]
            timestamp = waveforms[sid]["timestamp"][index[sid] : index[sid] + window_size]
            if len(data) < window_size:  # wrap around
                index[sid] = 0
                data = waveforms[sid]["data"][:window_size]
                timestamp = waveforms[sid]["timestamp"][:window_size]
            redis_client.xadd(
                sid,
                {
                    "data": json.dumps(data),
                    "timestamp": json.dumps(timestamp),
                },
            )
            redis_client.xtrim(sid, maxlen=60000)
            index[sid] += window_size
        # print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(1)


threading.Thread(target=replay, daemon=True).start()


@app.get("/")
def read_root():
    return {"message": "Replaying waveforms."}

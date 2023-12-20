import json
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd
import redis
import requests
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)

app = FastAPI()

PROTOCAL = "gs"
BUCKET = "quakeflow_share"
FOLDER = "demo"
MAXQUEUE = 3000
MAXMSG = 35

REDIS_HOST = "quakeflow-redis-master.default.svc.cluster.local"
PHASE_PICKER_API = "picking-api.default.svc.cluster.local"
PHASE_ASSOCIATION_API = "association-api.default.svc.cluster.local"
EARTHQUAKE_RELOCATION_API = "location-api.default.svc.cluster.local"

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    redis_client.ping()
except:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)


stations = pd.read_csv(f"{PROTOCAL}://{BUCKET}/{FOLDER}/obspy/stations.csv")
stations["location"] = stations["location"].fillna("")
stations["id"] = (
    stations["network"] + "." + stations["station"] + "." + stations["location"] + "." + stations["channel"]
)
station_ids = stations["id"].tolist()
exist_ids = []
for sid in station_ids:
    if not redis_client.exists(sid):
        continue
    exist_ids.append(sid)
    if not redis_client.xinfo_groups(sid):
        redis_client.xgroup_create(sid, "quakeflow", id="$", mkstream=False)
station_ids = exist_ids

data = {id: deque(maxlen=MAXQUEUE) for id in station_ids}
timestamp = {id: deque(maxlen=MAXQUEUE) for id in station_ids}

normalize = lambda x: (x - np.mean(x)) / np.std(x) / 6.0


def preprocess(meta, sample_rate=100.0, padding=True):
    if padding:
        begin_time = min([min(meta[id]["timestamp"]) for id in meta])
        end_time = max([max(meta[id]["timestamp"]) for id in meta])
    else:
        begin_time = max([min(meta[id]["timestamp"]) for id in meta])
        end_time = min([max(meta[id]["timestamp"]) for id in meta])

    comp = ["3", "2", "1", "E", "N", "Z"]
    order = {key: i for i, key in enumerate(comp)}
    comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}  ## for cases less than 3 components

    station_ids = defaultdict(list)
    for id in meta:
        station_ids[id[:-1]].append(id[-1])

    nx = len(station_ids)
    nt = int(round((end_time - begin_time) * sample_rate)) + 1
    data = np.zeros([3, nx, nt], dtype=np.float32) * np.nan
    for i, s in enumerate(station_ids):
        for j, c in enumerate(sorted(station_ids[s], key=lambda x: order[x])):
            if len(station_ids[s]) != 3:  ## less than 3 component
                j = comp2idx[c]

            index = ((np.array(meta[f"{s}{c}"]["timestamp"]) - begin_time) * sample_rate).round().astype(int)
            data[j, i, index] = meta[f"{s}{c}"]["data"]

    data = np.where(np.isnan(data), np.nanmean(data, axis=2, keepdims=True), data)

    meta = {
        "id": list(station_ids.keys()),
        "vec": data.tolist(),  # [3, nx, nt]
        "timestamp": [datetime.fromtimestamp(begin_time).strftime("%Y-%m-%dT%H:%M:%S.%f")] * nx,
    }
    return meta


def monitor_data():
    print("Monitoring data...", flush=True)
    while True:
        print("Updating...", flush=True)
        waveforms = {}
        for i, sid in enumerate(station_ids):
            messages = redis_client.xreadgroup(
                groupname="quakeflow",
                consumername="quakeflow",
                streams={sid: ">"},
                count=2,
                block=0,
            )
            for stream, entry in messages:
                for msg_id, msg in entry:
                    data_ = json.loads(msg["data"])
                    timestamp_ = json.loads(msg["timestamp"])
                    data[sid].extend(data_)
                    timestamp[sid].extend(timestamp_)

            if len(data[sid]) > 0:
                waveforms[sid] = {
                    "data": list(data[sid]),
                    "timestamp": list(timestamp[sid]),
                }

        if len(waveforms) == 0:
            continue

        payload = preprocess(waveforms)
        print(f"Sending {len(payload['id'])} stations.", flush=True)

        timeout = 30  # specify timeout in seconds

        try:
            resp_picking = requests.post(f"http://{PHASE_PICKER_API}/predict", json=payload, timeout=timeout)
            print(resp_picking.json(), flush=True)
        except requests.exceptions.Timeout:
            print("Timeout occurred during phase picking request")
            continue

        try:
            resp_association = requests.post(
                f"http://{PHASE_ASSOCIATION_API}/predict", json=resp_picking.json(), timeout=timeout
            )
            print(resp_association.json(), flush=True)
        except requests.exceptions.Timeout:
            print("Timeout occurred during phase association request")
            continue

        try:
            resp_location = requests.post(
                f"http://{EARTHQUAKE_RELOCATION_API}/predict", json=resp_association.json()["picks"], timeout=timeout
            )
            print(resp_location.json(), flush=True)
        except requests.exceptions.Timeout:
            print("Timeout occurred during earthquake relocation request")
            continue

        # plt.figure()
        # tmp = np.array(payload["vec"])
        # for i in range(tmp.shape[1]):
        #     plt.plot(normalize(np.array(payload["vec"])[0, i, :]) + i)
        # plt.savefig("test.png")
        # print("updated")

        # # resp = requests.post(f"{PHASE_PICKER_API}/predict", json=payload)
        # # if resp.status_code == 200:
        # #     picks = resp.json()
        # #     print(picks)

        # # await send(websocket_endpoint_phase_picker, payload)
        # # picks = await receive(websocket_endpoint_phase_picker)
        # out_s = send(websocket_endpoint_phase_picker, payload)
        # out_r = receive(websocket_endpoint_phase_picker)
        # _, picks = await asyncio.gather(out_s, out_r)
        # print(picks)
        # ## events = requests.post(PHASE_ASSOCIATION_API, json={"picks": picks, "stations": stations})"})
        # ## mechanisms = requests.post(FOCAL_MECHANISM_API, json={"picks": picks, "events": events, "stations": stations})
        # ## relocations = requests.post(EARTHQUAKE_RELOCATION_API, json={"picks": picks, "events": events, "stations": stations})

        # # for sid in waveforms:
        # #     waveforms[sid]["timestamp"] = [datetime.fromtimestamp(x).isoformat() for x in waveforms[sid]["timestamp"]]
        # await websocket.send_json(waveforms)
        # logging.info("Updating...")
        # await asyncio.sleep(1)


# monitor_data()
threading.Thread(target=monitor_data, daemon=True).start()


@app.get("/")
def read_root():
    return {"message": "QuakeFlow Data Hub."}

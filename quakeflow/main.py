"""
I want to build a web api to display realtime waveforms from N sensors. 
1. Push replayed data from each sensor to Redis stream. Each sensor ID should have one stream. The data format follows {"id": f"{network}_{stattion}", "data": Array(), "timestamp": Array()}. Each array has a random length around 100 points. 
2. Write a backend for the following functions: Monitor only new data from Redis; Concatenate the waveforms based on "id"; Align them based on "timestamp"; Keep a queue of 6000 points for each "id"; 
3. Write a frontend api to visualize the waveforms in real-time. Plot each sensor at a unique vertical offset for clarity.
I want to use Python, FastAPI, Websockets, Plotly.js, and Redis Stream. 
Can you show me a complete example line by line. Only output code, no explanation.
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import redis
import requests
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import websockets

logging.basicConfig(level=logging.INFO)

## For fastapi
PHASE_PICKER_API = "ws://127.0.0.1:8001"
# PHASE_ASSOCIATOR_API = None
# FOCAL_MECHANISM_API = None
# EARTHQUAKE_LOCATION_API = None
# EARTHQUAKE_RELOCATION_API = None
## For gradio
# PHASE_PICKING_MODEL = None
# PHASE_ASSOCIATION_MODEL = None
# FOCAL_MECHANISM_MODEL = None
# EARTHQUAKE_LOCATION_MODEL = None
# EARTHQUAKE_RELOCATION_MODEL = None

app = FastAPI()
r = redis.Redis(host="localhost", port=6379, db=0)
station_ids = json.load(open("station_ids.json", "r"))
try:
    for sid in station_ids:
        r.xgroup_create(sid, "quakeflow", id="$", mkstream=False)
except:
    pass

MAXQUEUE = 3000
MAXMSG = 35
data = {id: deque(maxlen=MAXQUEUE) for id in station_ids}
timestamp = {id: deque(maxlen=MAXQUEUE) for id in station_ids}

normalize = lambda x: (x - np.mean(x)) / np.std(x) / 6.0


# websocket_endpoint_phase_picker = websockets.connect("ws://127.0.0.1:8001/ws")
# print("Connected to phase picker")


@app.on_event("startup")
async def startup():
    global websocket_endpoint_phase_picker
    websocket_endpoint_phase_picker = await websockets.connect("ws://127.0.0.1:8001/ws")
    print("Connected to phase picker")


async def send(websocket, payload):
    # await websocket.send(payload)
    await websocket.send(json.dumps(payload))
    print("Sent ")


async def receive(websocket):
    response = await websocket.recv()
    print("Received ")
    return response


@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())


def preprocess(meta, sample_rate=100.0, padding=True):
    if padding:
        begin_time = min([meta[id]["timestamp"][0] for id in meta])
        end_time = max([meta[id]["timestamp"][-1] for id in meta])
    else:
        begin_time = max([meta[id]["timestamp"][0] for id in meta])
        end_time = min([meta[id]["timestamp"][-1] for id in meta])

    comp = ["3", "2", "1", "E", "N", "Z"]
    order = {key: i for i, key in enumerate(comp)}
    comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}  ## for cases less than 3 components

    station_ids = defaultdict(list)
    for id in meta:
        station_ids[id[:-1]].append(id[-1])

    nx = len(station_ids)
    # nt = (end_time - begin_time).seconds * sample_rate + 1
    nt = int(round((end_time - begin_time) * sample_rate)) + 1
    data = np.zeros([3, nt, nx], dtype=np.float32)
    for i, s in enumerate(station_ids):
        for j, c in enumerate(sorted(station_ids[s], key=lambda x: order[x])):
            if len(station_ids[s]) != 3:  ## less than 3 component
                j = comp2idx[c]

            trace = meta[f"{s}{c}"]["data"]
            it0 = int(round((meta[f"{s}{c}"]["timestamp"][0] - begin_time) * sample_rate))
            # print(it0, it0 + len(trace), len(trace), nt, begin_time, end_time, meta[f"{s}{c}"]["timestamp"][0])
            data[j, it0 : it0 + len(trace), i] = trace

    meta = {
        "id": list(station_ids.keys()),
        # "vec": data.tolist(),
        "vec": np.transpose(data, [2, 1, 0]).tolist(),
        "timestamp": [datetime.fromtimestamp(begin_time).isoformat()] * nx,
    }
    return meta


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        waveforms = {}
        for i, sid in enumerate(station_ids):
            # messages = r.xrevrange(id, count=MAXMSG)
            # messages = messages[::-1]
            # for _, msg in messages:
            messages = r.xreadgroup(
                groupname="quakeflow",
                consumername="quakeflow",
                streams={sid: ">"},
                count=2,
                block=0,
            )
            for stream, entry in messages:
                for msg_id, msg in entry:
                    data_ = json.loads(msg[b"data"])
                    timestamp_ = json.loads(msg[b"timestamp"])
                    data[sid].extend(data_)
                    timestamp[sid].extend(timestamp_)
            waveforms[sid] = {
                "data": (normalize(np.array(data[sid])) + i).tolist(),
                # "timestamp": [datetime.fromtimestamp(x).isoformat() for x in timestamp[sid]],
                "timestamp": list(timestamp[sid]),
            }
            # await websocket.send_json({id: {"data": list(data[id]), "timestamp": list(timestamp[id])}})

        payload = preprocess(waveforms)
        # resp = requests.post(f"{PHASE_PICKER_API}/predict", json=payload)
        # if resp.status_code == 200:
        #     picks = resp.json()
        #     print(picks)

        # await send(websocket_endpoint_phase_picker, payload)
        # picks = await receive(websocket_endpoint_phase_picker)
        out_s = send(websocket_endpoint_phase_picker, payload)
        out_r = receive(websocket_endpoint_phase_picker)
        _, picks = await asyncio.gather(out_s, out_r)
        print(picks)
        ## events = requests.post(PHASE_ASSOCIATION_API, json={"picks": picks, "stations": stations})"})
        ## mechanisms = requests.post(FOCAL_MECHANISM_API, json={"picks": picks, "events": events, "stations": stations})
        ## relocations = requests.post(EARTHQUAKE_RELOCATION_API, json={"picks": picks, "events": events, "stations": stations})

        # for sid in waveforms:
        #     waveforms[sid]["timestamp"] = [datetime.fromtimestamp(x).isoformat() for x in waveforms[sid]["timestamp"]]
        await websocket.send_json(waveforms)
        logging.info("Updating...")
        await asyncio.sleep(1)

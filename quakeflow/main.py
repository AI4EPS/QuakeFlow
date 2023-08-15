"""
I want to build a web api to display realtime waveforms from N sensors. 
1. Push replayed data from each sensor to Redis stream. Each sensor ID should have one stream. The data format follows {"id": f"{network}_{stattion}", "data": Array(), "timestamp": Array()}. Each array has a random length around 100 points. 
2. Write a backend for the following functions: Monitor only new data from Redis; Concatenate the waveforms based on "id"; Align them based on "timestamp"; Keep a queue of 6000 points for each "id"; 
3. Write a frontend api to visualize the waveforms in real-time. Plot each sensor at a unique vertical offset for clarity.
I want to use Python, FastAPI, Websockets, Plotly.js, and Redis Stream. 
Can you show me a complete example line by line. Only output code, no explanation.
"""

from fastapi import FastAPI, WebSocket
import redis
import json
import asyncio
import numpy as np
import logging
from datetime import datetime
from collections import deque

logging.basicConfig(level=logging.INFO)

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
                "timestamp": [datetime.fromtimestamp(x).isoformat() for x in timestamp[sid]],
            }
            # await websocket.send_json({id: {"data": list(data[id]), "timestamp": list(timestamp[id])}})
        await websocket.send_json(waveforms)
        logging.info("Updating...")
        await asyncio.sleep(1)

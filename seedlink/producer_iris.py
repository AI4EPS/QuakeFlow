#!/usr/bin/env python
import logging
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime
from json import dumps

import matplotlib
import matplotlib.pyplot as plt
import obspy
import pandas as pd
import requests
from kafka import KafkaProducer
from obspy.clients.fdsn import Client
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient, create_client

logging.basicConfig(level=logging.WARN)
matplotlib.use("agg")

## realtime station information
# http://ds.iris.edu/gmap/#network=_REALTIME&starttime=2021-03-01&datacenter=IRISDMC&networktype=permanent&planet=earth
#
# http://ds.iris.edu/gmap/#network=_REALTIME&channel=HH*&starttime=2021-03-01&datacenter=IRISDMC&networktype=permanent&planet=earth

timestamp = lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

##################### Config #####################
pi = 3.1415926
degree2km = pi * 6371 / 180

## Location
# center = (-115.53, 32.98) #salton sea
# center = (-117.504, 35.705) #ridgecrest
center = (-155.32, 19.39)  # havaii
horizontal_degree = 2.0
vertical_degree = 2.0

## Time range
starttime = obspy.UTCDateTime("2021-01-01T00:00:00") 
endtime = obspy.UTCDateTime(datetime.utcnow()) 

## seismic stations
network_list = ["HV", "PT"]
# channel_list = "HH*,HN*,BH*,EH*"
channel_list = "HH*"

## data center
client = "IRIS"

config = {}
config["center"] = center
config["xlim_degree"] = [center[0] - horizontal_degree / 2, center[0] + horizontal_degree / 2]
config["ylim_degree"] = [center[1] - vertical_degree / 2, center[1] + vertical_degree / 2]
config["degree2km"] = degree2km
config["starttime"] = starttime.datetime
config["endtime"] = endtime.datetime
config["networks"] = network_list
config["channels"] = channel_list
config["client"] = client

with open("config.pkl", "wb") as fp:
    pickle.dump(config, fp)

##################### realtime streaming station list #####################
stations_total = pd.read_csv(
    "realtime-stations.txt",
    sep="|",
    header=None,
    skiprows=3,
    names=["network", "station", "latitude", "longitude", "elevation(m)", "location", "starttime", "endtime"],
)
stations_total = stations_total[stations_total["network"].isin(config["networks"])]

plt.figure()
plt.plot(stations_total["longitude"], stations_total["latitude"], '^')
# plt.axis("scaled")

stations_select = stations_total[
    (config["xlim_degree"][0] < stations_total["longitude"])
    & (stations_total["longitude"] < config["xlim_degree"][1])
    & (config["ylim_degree"][0] < stations_total["latitude"])
    & (stations_total["latitude"] < config["ylim_degree"][1])
]

stations_select = stations_select.reset_index()
print("Number of selected stations: ", len(stations_select))

##################### download station info #####################
stations = Client("IRIS").get_stations(
    network=",".join(config["networks"]),
    station=",".join(stations_select["station"]),
    starttime=config["starttime"],
    endtime=config["endtime"],
    minlongitude=config["xlim_degree"][0],
    maxlongitude=config["xlim_degree"][1],
    minlatitude=config["ylim_degree"][0],
    maxlatitude=config["ylim_degree"][1],
    channel=config["channels"],
    level="response",
)  # ,
# filename="stations.xml")

# stations = obspy.read_inventory("stations.xml")
print("Number of downloaded stations: {}".format(sum([len(x) for x in stations])))
# stations.plot('local', outfile="stations.png")
# stations.plot('local')

station_locs = defaultdict(dict)
station_resp = defaultdict(dict)
for network in stations:
    for station in network:
        for chn in station:
            sid = f"{network.code}.{station.code}.{chn.location_code}.{chn.code[:-1]}"
            station_resp[
                f"{network.code}.{station.code}.{chn.location_code}.{chn.code}"
            ] = chn.response.instrument_sensitivity.value
            if sid in station_locs:
                station_locs[sid]["component"] += f",{chn.code[-1]}"
                station_locs[sid]["response"] += f",{chn.response.instrument_sensitivity.value:.2f}"
            else:
                component = f"{chn.code[-1]}"
                response = f"{chn.response.instrument_sensitivity.value:.2f}"
                dtype = chn.response.instrument_sensitivity.input_units.lower()
                tmp_dict = {}
                tmp_dict["longitude"], tmp_dict["latitude"], tmp_dict["elevation(m)"] = (
                    chn.longitude,
                    chn.latitude,
                    chn.elevation,
                )
                tmp_dict["component"], tmp_dict["response"], tmp_dict["unit"] = component, response, dtype
                station_locs[sid] = tmp_dict

station_locs = pd.DataFrame.from_dict(station_locs, orient='index')
station_locs.to_csv(
    "stations.csv",
    sep="\t",
    float_format="%.3f",
    index_label="station",
    columns=["longitude", "latitude", "elevation(m)", "unit", "component", "response"],
)


class Client(EasySeedLinkClient):
    def __init__(self, server_url, producer, autoconnect=True):
        super().__init__(server_url, producer)
        self.producer = producer

    def on_data(self, trace):
        if time.time() % 10 < 0.5: ## print every 60s
            print(f'Received trace: {trace}')
        if trace.stats.sampling_rate != 100:
            trace = trace.interpolate(100, method="linear")
        if trace.stats.channel[1] == "N": ## acceleration
            trace = trace.integrate()
            trace = trace.filter("highpass", freq=1.0)
        value = {
            "timestamp": timestamp(trace.stats.starttime.datetime),
            "vec": (trace.data / station_resp[trace.id]).tolist(),
            "dt": trace.stats.delta,
        }
        self.producer.send('waveform_raw', key=trace.id, value=value)


if __name__ == '__main__':
    print('Connecting to Kafka cluster for producer...')

    # TODO Will need to clean up this with better env config
    try:
        BROKER_URL = 'quakeflow-kafka-headless:9092'
        # BROKER_URL = '34.83.137.139:9094'
        producer = KafkaProducer(
            bootstrap_servers=[BROKER_URL],
            key_serializer=lambda x: dumps(x).encode('utf-8'),
            value_serializer=lambda x: dumps(x).encode('utf-8'),
        )
    except Exception as error:
        print('k8s kafka not found or connection failed, fallback to local')
        # BROKER_URL = '127.0.0.1:9092'
        BROKER_URL = '127.0.0.1:9094'
        producer = KafkaProducer(
            bootstrap_servers=[BROKER_URL],
            key_serializer=lambda x: dumps(x).encode('utf-8'),
            value_serializer=lambda x: dumps(x).encode('utf-8'),
        )

    print(f'Starting producer ...')

    client = Client('rtserve.iris.washington.edu:18000', producer=producer)
    for x in station_locs.index:
        x = x.split(".")
        client.select_stream(x[0], x[1], x[-1] + "?")
    client.run()

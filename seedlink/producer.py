from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
import pickle
import datetime
import numpy as np
import time
import requests
import logging
# import matplotlib.pyplot as plt


# $ bin/kafka-topics.sh --create --topic test-topic --bootstrap-server localhost:9092


# def visualize(waveforms, ids, timestamps, picks, sampling_rate):
#     def normalize(x): return (x - np.mean(x)) / np.std(x)
#     def calc_dt(t1, t2): return (datetime.strptime(t1, "%Y-%m-%dT%H:%M:%S.%f") - datetime.strptime(t2, "%Y-%m-%dT%H:%M:%S.%f")).total_seconds()

#     plt.figure()
#     for i in range(len(waveforms)):
#         plt.plot(normalize(waveforms[i][:, 0]) / 6 + i, "k", linewidth=0.5)

#     idx_dict = {k: i for i, k in enumerate(ids)}
#     for i in range(len(picks)):
#         if picks[i]["type"] == "p":
#             color = "blue"
#         else:
#             color = "red"
#         idx = int(calc_dt(picks[i]["timestamp"], timestamps[idx_dict[picks[i]["id"]]]) * sampling_rate)
#         plt.plot([idx, idx], [idx_dict[picks[i]["id"]] - 0.5, idx_dict[picks[i]["id"]] + 0.5], color=color)
#     plt.show()


def timestamp(x): return x.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def replay_data(producer):
    with open('fakedata.pkl', 'rb') as f:
        fakedata = pickle.load(f)
    # Load data configs
    data = fakedata['data']
    start_time = fakedata['start_time']
    sampling_rate = fakedata['sampling_rate']
    n_station = len(fakedata['station_id'])

    # Specify widow_size
    # Each station produces 100 sample/per second in the realworld scenario
    window_size = 100

    # Replay the data according to the window_size
    idx = 0
    while idx < len(data):
        # Current timestamp
        delta = datetime.timedelta(seconds=idx / sampling_rate)
        ts = timestamp(start_time + delta)
        # logging.warning((idx, ts))
        print((idx, ts))

        # batch of data of window_size
        vecs = data[idx: idx + window_size].transpose([1, 0, 2])

        ########Send req to PhaseNet and GMMA API in bulk, for testing purpose##########
        # req = {
        #     'id': fakedata['station_id'],
        #     'timestamp': [ts] * n_station,
        #     "vec": vecs.tolist(),
        #     "dt": 1.0 / sampling_rate
        # }
        # resp = requests.get("http://localhost:8000/predict", json=req)
        # visualize(vecs, fakedata['station_id'], [ts] * n_station, resp.json(), sampling_rate)
        # catalog = requests.get("http://localhost:8001/predict", json={"picks": resp.json()})
        # print(catalog.json())
        ################################################################################

        # Send stream of station data to Kafka

        # for spark.py
        # for i, station_id in enumerate(fakedata['station_id']):
        #    producer.send('waveform_raw', key=fakedata["station_id"][i],
        #                  value=(ts, vecs[i].tolist()))

        # for spark_structure.py
        for i, station_id in enumerate(fakedata['station_id']):
            producer.send('waveform_raw', key=fakedata["station_id"][i],
                          value={"timestamp": ts, "vec": vecs[i].tolist()})

        # producer.send('waveform_raw', key=fakedata["station_id"][i],
        #         value=vecs[i].tolist())

        # Sleep for 1 second to stimulate real stations
        time.sleep(1.0)

        # Next iteration
        idx += window_size

        # if idx >= 3 * window_size:
        #     raise


if __name__ == '__main__':
    # logging.warning('Connecting to Kafka cluster for producer...')
    print('Connecting to Kafka cluster for producer...')

    # TODO Will need to clean up this with better env config
    try:
        BROKER_URL = 'quakeflow-kafka-headless:9092'
        producer = KafkaProducer(bootstrap_servers=[BROKER_URL],
                                 key_serializer=lambda x: dumps(x).encode('utf-8'),
                                 value_serializer=lambda x: dumps(x).encode('utf-8'))
    except Exception as error:
        print('k8s kafka not found or connection failed, fallback to local')
        BROKER_URL = 'localhost:9092'
        producer = KafkaProducer(bootstrap_servers=[BROKER_URL],
                                 key_serializer=lambda x: dumps(x).encode('utf-8'),
                                 value_serializer=lambda x: dumps(x).encode('utf-8'))

    logging.warning('Starting producer...')
    # Phasenet & GMMA API test
    replay_data(producer)

    # Uncomment this to test Kafka + Spark integration with dummy data
    # for ts in range(10000):
    #     print(ts)
    #     for sid in range(16):
    #         producer.send('testtopic', key=f'station_{sid}', value=(ts, np.repeat(ts * 100 + sid, 2).tolist()))
    #     sleep(1)

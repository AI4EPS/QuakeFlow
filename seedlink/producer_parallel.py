from time import sleep
from json import dumps
#from kafka import KafkaProducer
import numpy as np
import pickle
import datetime
import numpy as np
import time
import requests
import logging
import threading
import multiprocessing
# multiprocessing.set_start_method('spawn')
# forkserver


def timestamp(x): return x.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def call_api(req):
    resp = None
    while resp is None:
        try:
            resp = requests.post("http://phasenet.quakeflow.com/predict", json=req)
        except Exception as e:
            print(e)
    # return resp

def replay_data():
    processes = []

    with open('fakedata.pkl', 'rb') as f:
        fakedata = pickle.load(f)

    # Load data configs
    data = fakedata['data']
    start_time = fakedata['start_time']
    sampling_rate = fakedata['sampling_rate']
    n_station = len(fakedata['station_id'])

    # Specify widow_size
    # Each station produces 100 sample/per second in the realworld scenario
    window_size = 3000
    req_list = []
    # Replay the data according to the window_size
    idx = 0
    while idx < len(data):
        # Current timestamp
        delta = datetime.timedelta(seconds=idx / sampling_rate)
        ts = timestamp(start_time + delta)
        # print((idx, ts))

        # batch of data of window_size
        vecs = data[idx: idx + window_size].transpose([1, 0, 2])

        ########Send req to PhaseNet and GMMA API in bulk, for testing purpose##########
        req = {
            'id': fakedata['station_id'],
            'timestamp': [ts] * n_station,
            "vec": vecs.tolist(),
            "dt": 1.0 / sampling_rate
        }

        req_list.append(req)

        # def call_api(req):
        #     resp = None
        #     while resp is None:
        #         try:
        #             resp = requests.get("http://34.83.156.209:8000/predict2gmma", json=req)
        #         except Exception as e:
        #             print(e)
        
        # p = threading.Thread(target=call_api, args=(req,))
        # processes.append(p)
        # p.start()

        # p = multiprocessing.Process(target=call_api, args=(req,))
        # p.start()
        # processes.append(p)

        # Next iteration
        idx += window_size
        break

    # return processes
    return req_list

if __name__ == '__main__':

    
    processes_list = []

    req_list = replay_data()
    num_parallel = len(req_list)

    # for num_parallel in [1,2,4,8,16,32,64]:
    #for num_parallel in [256, 128,64,32,16,8,4,2,1]:
    for num_parallel in [64,32,16,8,4,2,1]:
        # num_parallel = 8
    
        pool = multiprocessing.Pool(processes=num_parallel)
    
        ## dummy run to start autoscaling

        if num_parallel == 64:
            repeat = 10
        else:
            repeat = 3
        #for i in range(repeat):
        for i in range(1000):
            prev_time = time.time()
            # processes = replay_data()
            #req_list = replay_data()
            #print(f"Dummy: Data generated: {time.time()-prev_time}s")
            pool.map(call_api, req_list*num_parallel)
            # for p in processes:
            #     p.join()
            # processes_list.extend(processes)
            print(f"Parallel = {num_parallel}; Dummy: Iter {i}: {time.time()-prev_time}s")
            time.sleep(5.0)
    
        start_time = time.time()
        repeat = 3
        for i in range(repeat):
            prev_time = time.time()
            # processes = replay_data()
            #req_list = replay_data()
            #print(f"Data generated: {time.time()-prev_time}s")
            pool.map(call_api, req_list*num_parallel)
            # for p in processes:
            #     p.join()
            # processes_list.extend(processes)
            print(f"Parallel = {num_parallel}; Iter {i}: {time.time()-prev_time}s")
    
        # for p in processes_list:
        #     p.join()
        
        print(f"Parallel = {num_parallel}; Processing time: {(time.time()-start_time)/repeat}s")
        pool.close()

import json
import requests
import sys
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.sql.functions import window
import numpy as np
from kafka import KafkaProducer

PHASENET_API_URL = "http://localhost:8000"


if __name__ == "__main__":
    WINDOW_DURATION = 30
    NUMBER_OF_BATCHES = 30
    SAMPLING_RATE = 100
    sc = SparkContext(appName="PythonStreamingRecieverKafkaWordCount")
    ssc = StreamingContext(sc, 1)  # 1 second window
    broker, topic = 'localhost:2181', 'waveform_raw'
    kvs = KafkaUtils.createStream(ssc,
                                  broker, "spark-streaming-consumer",
                                  {topic: 1})
    # lines = kvs.map(lambda x: (json.loads(x[1])['key'], json.loads(x[1])['data']))
    lines = kvs.map(lambda x: tuple(
        (
            x[0],
            tuple(json.loads(x[1]))
        )
    ))

    ssc.checkpoint("./checkpoint-quakeflow")

    def run_phasenet_predict(rdd):
        results = rdd.collect()
        # Corner case: empty RDD
        if not results:
            return
        station_ids, timestamps, vecs = results[0]
        req = {
            'id': station_ids,
            'timestamp': [x[0] for x in timestamps],  # workaround
            "vec": vecs,
            "dt": 1.0 / SAMPLING_RATE
        }
        try:
            resp = requests.get(f'{PHASENET_API_URL}/predict2gmma', json=req)
            print('Phasenet & GMMA resp', resp.json())
        except Exception as error:
            print('Phasenet & GMMA error', error)

    def send_waveform_grouped(rdd):
        # sub-optimal :(
        # Might be slow if we have to re-init the producer every 3 seconds
        # but this is due to Spark's constraint, unless we want to broadcast
        # producer to all nodes which I'm not sure how to do it
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                            key_serializer=lambda x: json.dumps(x).encode('utf-8'),
                            value_serializer=lambda x: json.dumps(x).encode('utf-8'))
        results = rdd.collect()
        if not results:
            return
        print('##large group##', results)
        producer.send('waveform_grouped',  value=results)
        

    # [groupByKeyAndWindow]
    # - windowDuration: width of the window, number of seconds
    # - slideDuration: sliding interval of the window

    # -> groupby: (station_id, (timestamp, feat_vecs))
    grouped_df = lines.groupByKeyAndWindow(windowDuration=WINDOW_DURATION+1, slideDuration=3)

    # -> map: (station_id, [(ts_0, vec_0), (ts_1, vec_1), ...]), sort data by timestamp
    df_feats = grouped_df.map(lambda x: (x[0][1:-1], sorted(x[1], key=lambda y: y[0])))

    # -> filter: discard rows that has less than NUMBER_OF_BATCHES of data batches
    df_feats = df_feats.filter(lambda x: len(x[1]) >= NUMBER_OF_BATCHES)

    # -> map: (station_id, [timestamps], [features (30, 100, 3)])
    df_feats = df_feats.map(lambda x: (x[0],
                                       [y[0] for y in x[1][:NUMBER_OF_BATCHES]],
                                       [y[1] for y in x[1][:NUMBER_OF_BATCHES]]))

    # -> map: ([station_id], [[timestamps]], [[flatten_features (3000, 3)]])
    df_feats = df_feats.map(lambda x: ([x[0]], [x[1]], [[row for batch in x[2] for row in batch]]))

    # Reduce everything into a single list [[station_id], [timestamps], [feats]]
    df_reduced = df_feats.reduce(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))

    # run_phasenet_predict
    df_reduced.foreachRDD(run_phasenet_predict)
    

    # Ugly as hell, but it works now (comment out for now, buggy)
    # grouped_large_df = lines.groupByKeyAndWindow(windowDuration=61, slideDuration=3)
    # df_large_feats = grouped_large_df.map(lambda x: (x[0][1:-1], sorted(x[1], key=lambda y: y[0])))
    # df_large_feats = df_large_feats.filter(lambda x: len(x[1]) >= 60)
    # df_large_feats = df_large_feats.map(lambda x: (x[0],
    #                                    [y[0] for y in x[1][:60]],
    #                                    [y[1] for y in x[1][:60]]))
    # df_large_feats = df_large_feats.map(lambda x: ([x[0]], [x[1]], [[row for batch in x[2] for row in batch]]))
    # df_large_feats = df_large_feats.reduce(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
    # df_large_feats.foreachRDD(send_waveform_grouped)

    # df_reduced.pprint()

    ssc.start()
    ssc.awaitTermination()


# Have to setup the Spark libraries first!
# ~/spark-2.3.3-bin-hadoop2.7/bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.3.3 test_spark.py > log.txt

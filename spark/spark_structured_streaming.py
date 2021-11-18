import os

import numpy as np
import pyspark.sql.functions as F
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    StringType,
    StructField,
    StructType,
)

# PHASENET_API_URL = "http://localhost:8000"
# BROKER_URL = 'localhost:9092'
PHASENET_API_URL = "http://phasenet-api:8000"
BROKER_URL = 'quakeflow-kafka-headless:9092'
# BROKER_URL = '34.83.137.139:9094'

## For seedlink dataformat
WATERMARK_DELAY = "60 seconds"
WINDOW_DURATION = "35 seconds"
SLIDE_DURATION = "10 seconds"
schema = StructType([StructField("timestamp", StringType()), StructField("vec", ArrayType(FloatType()))])

## For producer.py
# WATERMARK_DELAY = "1.5 seconds"
# WINDOW_DURATION = "30 seconds"
# SLIDE_DURATION = "3 seconds"
# NUMBER_SEGMENTS = "30 seconds"
# schema = StructType([StructField("timestamp", StringType()),
#                      StructField("vec", ArrayType(ArrayType(FloatType())))])

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1 pyspark-shell'

spark = SparkSession.builder.appName("spark").getOrCreate()
df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", f"{BROKER_URL}")
    .option("subscribe", "waveform_raw")
    .load()
)

df = (
    df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)", "timestamp")
    .withColumn("key", F.regexp_replace('key', '"', ''))
    .withColumn("value", F.from_json(col("value"), schema))
    .withColumn("vec", col('value.vec'))
    .withColumn("vec_timestamp", col('value.timestamp'))
    .withColumn("vec_timestamp_utc", F.from_utc_timestamp(col('value.timestamp'), "UTC"))
)

## For seedlink dataformat
df_window = (
    df.withWatermark("vec_timestamp_utc", WATERMARK_DELAY)
    .groupBy(df.key, F.window("vec_timestamp_utc", WINDOW_DURATION, SLIDE_DURATION))
    .agg(F.sort_array(F.collect_list(F.struct('vec_timestamp_utc', 'vec_timestamp', 'vec'))).alias("collected_list"))
    .withColumn("vec", F.flatten(col("collected_list.vec")))
    .withColumn("vec_timestamp", col("collected_list.vec_timestamp").getItem(0))
    .drop("collected_list")
)

## For producer.py
# df_window = df.withWatermark("vec_timestamp_utc", WATERMARK_DELAY) \
#     .groupBy(df.key, F.window("vec_timestamp_utc", WINDOW_DURATION, SLIDE_DURATION))\
#     .agg(F.sort_array(F.collect_list(F.struct('vec_timestamp_utc', 'vec_timestamp', 'vec'))).alias("collected_list"))\
#     .filter(F.size(col("collected_list")) == NUMBER_SEGMENTS)\
#     .withColumn("vec", F.flatten(col("collected_list.vec")))\
#     .withColumn("vec_timestamp", col("collected_list.vec_timestamp").getItem(0))\
#     .drop("collected_list")\


def foreach_batch_function(df_batch, batch_id):
    print(f'>>>>>>>>>>>>>>>> {batch_id} >>>>>>>>>>>>>>>>')

    df_batch = (
        df_batch.groupby(col('window'))
        .agg(
            F.collect_list('key').alias('key'),
            F.collect_list('vec_timestamp').alias('timestamp'),
            F.collect_list('vec').alias('vec'),
        )
        .sort(col("window"))
    )

    res = df_batch.collect()
    for x in res:
        # print(x.key, x.timestamp, len(x.vec), [len(x.vec[i]) for i in range(len(x.vec))])
        req = {'id': x.key, 'timestamp': x.timestamp, "vec": x.vec, "dt": 1.0 / 100}  # workaround

        try:
            # resp = requests.get('{}/predict2gmma'.format(PHASENET_API_URL), json=req)
            resp = requests.get('{}/predict_stream_phasenet2gamma'.format(PHASENET_API_URL), json=req)
            print('Phasenet & GaMMA catalog', resp.json()["catalog"])
        except Exception as error:
            print('Phasenet & GaMMA error', error)

    return None


query = df_window.writeStream.format("memory").outputMode("append").foreachBatch(foreach_batch_function).start()
query.awaitTermination()

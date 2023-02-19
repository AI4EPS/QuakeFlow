import os

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    collect_list,
    expr,
    from_json,
    from_utc_timestamp,
    regexp_replace,
    sort_array,
    struct,
    udf,
    window,
)
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

# Parameters
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1 pyspark-shell"
# BROKER_URL = "quakeflow-kafka-headless:9092"
BROKER_URL = "127.0.0.1:9094"
QUAKEFLOW_API_URL = "http://phasenet-api:8000"
KAFKA_TOPIC = "waveform_raw"
WATERMARK_DELAY = "60 seconds"
WINDOW_DURATION = "35 seconds"
SLIDE_DURATION = "10 seconds"

input_schema = StructType(
    [
        StructField("timestamp", StringType()),
        StructField("vec", ArrayType(FloatType())),
        StructField("dt", FloatType()),
    ]
)

# Define the connection to Kafka and the topic to read from
spark = SparkSession.builder.appName("spark").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", f"{BROKER_URL}")
    .option("subscribe", f"{KAFKA_TOPIC}")
    .option("startingOffset", "earliest")
    .load()
)

df = (
    df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
    .withColumn("key", regexp_replace("key", '"', ""))
    .withColumn("value", from_json(col("value"), input_schema))
    .select("key", "value.timestamp", "value.vec", "value.dt")
)

# Preformating
@udf(StringType())
def calculate_end_timestamp(begin_timestamp, vec, dt):
    return (pd.to_datetime(begin_timestamp) + pd.Timedelta(seconds=(len(vec) - 1) * dt)).isoformat(
        timespec="milliseconds"
    )


df = (
    df.withColumn("utc_timestamp", from_utc_timestamp(col("timestamp"), "UTC"))
    .withColumnRenamed("timestamp", "begin_timestamp")
    .withColumn("end_timestamp", calculate_end_timestamp("begin_timestamp", "vec", "dt"))
)

df = df.withColumn("station_id", expr("substr(key, 1, length(key)-1)")).withColumn("channel", col("key").substr(-1, 1))

# Sliding window
df = (
    df.withWatermark("utc_timestamp", f"{WATERMARK_DELAY}")
    .groupBy(
        window("utc_timestamp", f"{WINDOW_DURATION}", f"{SLIDE_DURATION}"),
        col("station_id"),
    )
    .agg(
        sort_array(
            collect_list(struct("key", "channel", "begin_timestamp", "end_timestamp", "dt", "vec")), asc=True
        ).alias("data_list"),
        F.min("begin_timestamp").alias("begin_timestamp"),
        F.max("end_timestamp").alias("end_timestamp"),
    )
)

# Merge waveform of the same station
@udf(ArrayType(ArrayType(FloatType())))
def concatenate_vec(vecs, channels, begin_timestamps, end_timestamps, dts):
    min_begin_timestamp = pd.to_datetime(min(begin_timestamps))
    max_end_timestamp = pd.to_datetime(max(end_timestamps))
    if len(set(dts)) > 1:
        print("Warning: dt is not the same")

    dt = dts[0]
    vec_length = int((max_end_timestamp - min_begin_timestamp).total_seconds() / dt)
    num_channels = 3
    vec = [[0.0] * vec_length] * num_channels
    comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}

    for c in channels:
        j = comp2idx[c]
        for i in range(len(vecs)):
            start_idx = int((pd.to_datetime(begin_timestamps[i]) - min_begin_timestamp).total_seconds() / dt)
            # end_idx = int((pd.to_datetime(end_timestamps[i]) - min_begin_timestamp).total_seconds() / dt)
            # vec[j][start_idx:end_idx] = vecs[i]
            vec[j][start_idx : start_idx + len(vecs[i])] = vecs[i]

    return vec


df = (
    df.withColumn(
        "vec",
        concatenate_vec(
            "data_list.vec", "data_list.channel", "data_list.begin_timestamp", "data_list.end_timestamp", "data_list.dt"
        ),
    )
    .withColumn("dt", col("data_list.dt").getItem(0))
    .drop("data_list")
)


# Call QuakeFlow API
def send_to_quakeflow_api(df_batch, batch_id):
    print(f">>>>>>>>>>>>>>>> {batch_id} >>>>>>>>>>>>>>>>")

    # df_bath = df_batch.agg(collect_list(struct("station_id", "begin_timestamp", "end_timestamp", "dt", "vec")).alias("data_list"))
    # station_id = df_bath.select("data_list.station_id").collect()[0][0]
    # begin_timestamp = df_bath.select("data_list.begin_timestamp").collect()[0][0]
    # end_timestamp = df_bath.select("data_list.end_timestamp").collect()[0][0]
    # dt = df_bath.select("data_list.dt").collect()[0][0]
    # vec = df_bath.select("data_list.vec").collect()[0][0]

    df_batch = df_batch.toPandas()
    station_id = df_batch["station_id"].tolist()
    begin_timestamp = df_batch["begin_timestamp"].tolist()
    end_timestamp = df_batch["end_timestamp"].tolist()
    dt = df_batch["dt"].tolist()
    vec = df_batch["vec"].tolist()

    payload = {
        "id": station_id,
        "vec": vec,
        "begin_timestamp": begin_timestamp,
        "end_timestamp": end_timestamp,
        "dt": dt,
    }

    print(station_id, begin_timestamp, end_timestamp, dt)
    try:
        resp = requests.get(f"{QUAKEFLOW_API_URL}/predict_streaming", json=payload)
        print("QuakeFlow catalog", resp.json()["catalog"])
    except Exception as error:
        print(error)


query = df.writeStream.format("memory").outputMode("append").foreachBatch(send_to_quakeflow_api).start()
# query = df.writeStream.format("console").outputMode("update").start()

query.awaitTermination()

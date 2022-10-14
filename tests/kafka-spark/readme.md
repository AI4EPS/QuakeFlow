# Kafka & Pyspark 

This folder will be deprecated as we split things into individual docker containers.

## Setup

1. Install Conda Env 
```
conda env create --name cs329s --file=env.yml
```

2. Run your Zookeeper and Kafka cluster

See https://kafka.apache.org/quickstart for the installation and detailed steps.

```
# Start the ZooKeeper service
$ bin/zookeeper-server-start.sh config/zookeeper.properties

# Start the Kafka broker service
$ bin/kafka-server-start.sh config/server.properties
```

3. Create a topic `testtopic` (just for test purpose)

```
$ bin/kafka-topics.sh --create --topic waveform_raw --bootstrap-server localhost:9092
```

4. Setup PhaseNet and GMMA

PhaseNet and GMMA are independent to this Quakeflow repo. You can clone and download 
both of them in a different folder.

PhaseNet: https://github.com/wayneweiqiang/PhaseNet

```
$ git clone -b quakeflow https://github.com/wayneweiqiang/PhaseNet
$ cd PhaseNet
$ uvicorn app:app --reload --port 8000
```

Open another terminal and run

GMMA: https://github.com/wayneweiqiang/GMMA

```
$ git clone -b quakeflow https://github.com/wayneweiqiang/GMMA
$ cd GMMA
$ uvicorn app:app --reload --port 8001
```

5. Run the `producer.py` script

```
$ python producer.py
```

and you should see the script print out some timestamps every second


<!-- 6. Run the `consumer.py` script

The consumer will read the messages from the Kafka cluster. -->

6. Run the `spark.py` script for testing the Spark features

- `spark-submit` is pre-installed in our environment

- Run the following command, and you will see the logs in `logs.txt`

```
$ spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.3.3 spark.py > logs.txt
```

7. Check the `GMMA` API service after 30 seconds, you should see [200 OK] and some outputs about the earthquakes

<img src="https://i.imgur.com/qPEzICR.png">

Go to the Spark UI portal (http://localhost:4040/) and you can see the jobs, stages and streaming statistics. 

<img src="https://i.imgur.com/Q7ndx2R.png">

Also some cool DAG Visualization about how the streaming ETL pipeline is done

<img src="https://i.imgur.com/TR1dUHA.png" height="900px">

<!-- https://stackoverflow.com/questions/40384458/spark-streaming-processing-time-vs-total-delay-vs-processing-delay -->

# Spark ETL Pipeline

Spark streaming ETL Pipeline

Build the docker image

```
docker build --tag quakeflow-spark:1.0 .
```

Run the Spark ETL Pipeline

```
docker run -it quakeflow-spark:1.0
```

Run it locally (make sure update the spark lib to 3.1.1)
```
python spark_structured_streaming.py
```

FROM ubuntu:18.04

# Install Java and Spark
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y openjdk-11-jdk git wget tzdata
RUN ln -fs /usr/share/zoneinfo/US/Pacific-New /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Setup env variables
ENV PYTHONUNBUFFERED=1


WORKDIR /app
COPY env.yml /app
RUN conda env create --name quakeflow --file=env.yml
SHELL ["conda", "run", "-n", "quakeflow", "/bin/bash", "-c"]


COPY . /app
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "quakeflow", "python", "/app/spark_streaming.py"]
# CMD /spark-2.4.7-bin-hadoop2.7/bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.4.0 /app/spark.py

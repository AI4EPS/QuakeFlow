# FROM continuumio/miniconda3

FROM ubuntu:18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

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
RUN conda env update -f=env.yml -n base
RUN pip install --no-cache-dir --upgrade -i https://pypi.anaconda.org/zhuwq0/simple gmma
RUN conda clean --all 
#SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

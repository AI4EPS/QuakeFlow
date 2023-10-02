FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y git wget
RUN apt-get install -y libgeos++-dev

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV MINICONDA_VERSION=Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
RUN wget "https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}" && \
    mkdir /root/.conda && \
    bash $MINICONDA_VERSION -b && \
    rm -f $MINICONDA_VERSION && \
    conda --version

ENV PYTHONUNBUFFERED=1
## plotting
RUN conda install -c conda-forge pygmt
RUN pip install --no-cache-dir cartopy plotly
## machine learning
RUN pip install --no-cache-dir numpy scipy matplotlib pandas scikit-learn
RUN pip install --no-cache-dir obspy pyproj
RUN pip install --no-cache-dir fsspec gcsfs
RUN pip install --no-cache-dir kfp
RUN pip install --no-cache-dir git+https://github.com/wayneweiqiang/GaMMA.git

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN conda clean --all --yes
RUN rm -rf /root/.cache/pip
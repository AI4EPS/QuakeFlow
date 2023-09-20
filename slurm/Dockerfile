FROM ubuntu:20.04


RUN apt-get update
RUN apt-get install -y git wget libgeos++-dev

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    conda --version

ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir numpy scipy matplotlib pandas scikit-learn
RUN pip install --no-cache-dir obspy cartopy pyproj
RUN pip install --no-cache-dir fsspec gcsfs
RUN pip install --no-cache-dir kfp
RUN pip install --no-cache-dir git+https://github.com/wayneweiqiang/GaMMA.git

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN conda clean --all --yes
RUN rm -rf /root/.cache/pip
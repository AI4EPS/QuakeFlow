FROM python:3.8

RUN apt-get update && apt-get install -y gfortran && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

ARG CACHEBUST=1 

ENV PATH="/opt/:${PATH}"
RUN wget -O HYPODD_1.3.tar.gz http://www.ldeo.columbia.edu/~felixw/HYPODD/HYPODD_1.3.tar.gz
RUN tar -xf HYPODD_1.3.tar.gz
COPY Makefile /opt/HYPODD/src/hypodd
COPY hypoDD.inc /opt/HYPODD/include
RUN ln -s $(which gfortran) f77
RUN ln -s $(which gfortran) g77
RUN make -C HYPODD/src

RUN mkdir hypodd
RUN cp HYPODD/src/ph2dt/ph2dt hypodd/ph2dt
RUN cp HYPODD/src/hypoDD/hypoDD hypodd/hypoDD

# RUN ls -l /opt/HYPODD/src/hypoDD/hypoDD
# COPY gamma2hypodd.py convert_stations.py /opt/
# COPY hypoDD_ct.inp hypoDD_cc.inp ph2dt.inp /opt/
RUN python -m pip install --upgrade numpy pandas tqdm minio && rm -rf /var/cache/apk/*
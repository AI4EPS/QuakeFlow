FROM python:3.8

RUN apt-get update && apt-get install -y vim ssh rsync && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade obspy && rm -rf /var/cache/apk/*

WORKDIR /opt

# RUN ssh-keygen -q -t rsa -N "" <<< $"\ny" >/dev/null 2>&1
# RUN ssh-copy-id zhuwq@wintermute.gps.caltech.edu

# ARG CACHEBUST=1
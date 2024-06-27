# %% [markdown]
##  Important: Before downloading
# - Make sure on Hinet website you select which stations and networks you want to download continuous waveform data for and then run this notebook (ie Hinet vs Fnet data and which province(s))
import os
import threading
import time
from datetime import datetime

import pandas as pd
from HinetPy import Client

# %%
codes = ["0101", "0103"]
codes = ["0101"]
# codes = ["0103"]
USERNAME = ""
PASSWORD = ""
TIMEOUT = 6000  # seconds
client = Client(user=USERNAME, password=PASSWORD, timeout=TIMEOUT * 10, retries=3, max_sleep_count=300)

# %%
root_path = "local"
region = "hinet"
result_path = f"{root_path}/{region}/win32"
backup_path = f"{root_path}/{region}/backup"
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(backup_path):
    os.makedirs(backup_path)
for code in codes:
    if not os.path.exists(f"{backup_path}/{code}"):
        os.makedirs(f"{backup_path}/{code}")

# %%
client.select_stations("0101", ["N.SHKH", "N.TGIH", "N.AMZH", "N.WMZH", "N.UCUH", "N.YGDH", "N.SUZH"])
client.select_stations("0103", ["N.WJMF"])
stations = []
for code in codes:
    stations.extend(client.get_selected_stations(code))
print(f"Number of stations: {len(stations)}")

network = ""
location = ""
instrument = ""
component = "E,N,U"
stations = [
    {
        "station_id": f"{network}.{x.name}.{location}.{instrument}",
        "code": x.code,
        "elevation_m": x.elevation,
        "latitude": x.latitude,
        "longitude": x.longitude,
        "network": network,
        "location": location,
        "instrument": instrument,
        "component": component,
    }
    for x in stations
]
stations = pd.DataFrame(stations)
print(stations)
if not os.path.exists(f"{root_path}/{region}/results/data"):
    os.makedirs(f"{root_path}/{region}/results/data")
stations.to_csv(f"{root_path}/{region}/results/data/stations.csv", index=False)
stations.set_index("station_id", inplace=True)
stations.to_json(f"{root_path}/{region}/results/data/stations.json", orient="index", indent=2)

# %%
start_date = pd.to_datetime("2000-01-01")
end_date = pd.to_datetime("2024-06-25")
# end_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
span = 60  # minutes
dates = pd.date_range(start_date, end_date, freq=pd.Timedelta(minutes=span), inclusive="left")[-3::-3]

# %%
fails = 0
for date in dates:
    print(f"Downloading data for {date}")
    outdir = f"{result_path}/{date.strftime('%Y-%j/%H')}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for code in codes:

        if os.path.exists(f"{outdir}/{code}_{date.strftime('%Y%m%d%H%M')}_{span}.cnt") and os.path.exists(
            f"{outdir}/{code}_{date.strftime('%Y%m%d')}.ch"
        ):
            print(f"Data already exists for {date}")
            continue

        # client.get_continuous_waveform(
        #     code=code, starttime=date.strftime("%Y-%m-%dT%H:%M:%S.%f"), span=span, outdir=outdir, threads=3
        # )

        thread = threading.Thread(
            target=client.get_continuous_waveform,
            kwargs={
                "code": code,
                "starttime": date.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "span": span,
                "outdir": outdir,
                "threads": 3,
                "cleanup": False,
            },
        )
        thread.start()
        thread.join(TIMEOUT)
        if thread.is_alive():
            print(f"Timeout for {code} {date}")

        # check if data is downloaded
        if not os.path.exists(f"{outdir}/{code}_{date.strftime('%Y%m%d%H%M')}_{span}.cnt"):
            fails += 1
            print(f"Failed to download data: {code}_{date.strftime('%Y%m%d%H%M')}_{span}.cnt")
            print(f"Wait for {fails * 60} seconds...")
            time.sleep(fails * 60)
        else:
            fails = 0

        if os.path.exists(f"{code[:2]}_{code[2:]}_{date.strftime('%Y%m%d')}.euc.ch"):
            os.system(
                f"mv {code[:2]}_{code[2:]}_{date.strftime('%Y%m%d')}.euc.ch {outdir}/{code}_{date.strftime('%Y%m%d')}.ch"
            )

        os.system(f"mv ./*.cnt {backup_path}/{code}/")

# %%

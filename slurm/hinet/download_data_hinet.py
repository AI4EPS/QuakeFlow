# %% [markdown]
##  Important: Before downloading
# - Make sure on Hinet website you select which stations and networks you want to download continuous waveform data for and then run this notebook (ie Hinet vs Fnet data and which province(s))
import os

import pandas as pd
from HinetPy import Client

# %%
client = Client(timeout=120, retries=6)
USERNAME = ""
PASSWORD = ""
client.login(USERNAME, PASSWORD)

# %%
start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2024-01-04")
span = 60  # minutes
dates = pd.date_range(start_date, end_date, freq=pd.Timedelta(minutes=span))

# %%
root_path = "local"
region = "hinet"
result_path = f"{root_path}/{region}/win32"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# %%
code = "0101"
for date in dates:
    print(f"Downloading data for {date}")
    outdir = f"{result_path}/{date.strftime('%Y-%j/%H')}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if os.path.exists(f"{outdir}/{code}_{date.strftime('%Y%m%d%H%M')}_{span}.cnt"):
        print(f"Data already exists for {date}")
        continue

    client.get_continuous_waveform(
        code=code, starttime=date.strftime("%Y-%m-%dT%H:%M:%S.%f"), span=span, outdir=outdir, threads=3
    )
# %%

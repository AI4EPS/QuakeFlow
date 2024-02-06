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
start_date = pd.to_datetime("2024-02-01")
end_date = pd.to_datetime("2024-02-04")
dates = pd.date_range(start_date, end_date, freq="D")

# %%
root_path = "local"
region = "hinet"
result_path = f"{root_path}/{region}/win32"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# %%
for date in dates:
    print(f"Downloading data for {date}")
    outdir = f"{result_path}/{date.strftime('%Y-%j')}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    client.get_continuous_waveform(
        code="0101", starttime=date.strftime("%Y-%m-%dT%H:%M:%S.%f"), span=1440, outdir=outdir, threads=3
    )
# %%

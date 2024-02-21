# %%
import json
import os

import fsspec
import pandas as pd

# %%
protocol = "gs"
token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
with open(token_json, "r") as fp:
    token = json.load(fp)

# %%
bucket = "quakeflow_catalog"
folder = "NC/phasenet"

fs = fsspec.filesystem(protocol, token=token)

# %%
# year = 2023
# csv_list = fs.glob(f"{bucket}/{folder}/??/{year}/{year}.???/*.csv")  ## choose year and jday
# csv_list = [f"{protocol}://{csv}" for csv in csv_list]
# print(f"Save {len(csv_list)} CSV files")
# with open(f"{year}.txt", "w") as fp:
#     fp.write("\n".join(csv_list))

# with open(f"{year}.txt", "r") as fp:
#     csv_list = fp.read().splitlines()

# %%
year = 2023
with fs.open(f"{bucket}/{folder}/csv_list/{year}.txt", "r") as fp:
    csv_list = fp.read().splitlines()

for csv in csv_list:
    df = pd.read_csv(fs.open(csv, token=token))
    print(df)
    raise

# %%

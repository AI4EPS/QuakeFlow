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
csv_list = fs.glob(f"{bucket}/{folder}/ncedc-pds/continuous_waveforms/??/2023/2023.???/*.csv")
with open("csv_list.txt", "w") as fp:
    fp.write("\n".join(csv_list))

# %%
with open("csv_list.txt", "r") as fp:
    csv_list = fp.read().split("\n")

for csv in csv_list:
    df = pd.read_csv(fs.open(csv, "r"))
    print(df)
    raise

# %%

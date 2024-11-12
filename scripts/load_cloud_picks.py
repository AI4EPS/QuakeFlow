# %%
import json
import os
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pandas as pd
from tqdm import tqdm

# %%
if __name__ == "__main__":

    # %%
    protocol = "gs"
    token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)

    bucket = "quakeflow_catalog"
    folder = "NC/phasenet_merged"  ## NCEDC
    # folder = "SC/phasenet_merged"  ## SCEDC

    fs = fsspec.filesystem(protocol, token=token)

    def load_csv(jday):
        with fs.open(jday, "r") as fp:
            return pd.read_csv(fp, dtype=str)

    # %%
    years = range(2023, 2024)

    for year in years:
        jdays = fs.glob(f"{bucket}/{folder}/{year}/????.???.csv")

        with ThreadPoolExecutor(max_workers=32) as executor:
            picks = list(
                tqdm(executor.map(load_csv, jdays), total=len(jdays), desc=f"Loading {bucket}/{folder}/{year}")
            )

    # %%
    picks = pd.concat(picks)
    picks.to_csv("phasenet_picks.csv", index=False)

    # %%
    picks = pd.read_csv("phasenet_picks.csv")
    print(f"Loaded {len(picks):,} picks")

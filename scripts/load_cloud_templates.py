# %%
import json
import os
from concurrent.futures import ThreadPoolExecutor

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
if __name__ == "__main__":

    # %%
    result_path = "results/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # %%
    protocol = "gs"
    token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
    with open(token_json, "r") as fp:
        token = json.load(fp)

    bucket = "quakeflow_catalog"
    folder = "Cal/cctorch"

    fs = fsspec.filesystem(protocol, token=token)

    # %%
    def plot_templates(templates, events, picks):
        templates = templates - np.nanmean(templates, axis=(-1), keepdims=True)
        std = np.std(templates, axis=(-1), keepdims=True)
        std[std == 0] = 1.0
        templates = templates / std

        plt.figure(figsize=(10, 10))
        plt.imshow(templates[:, -1, 0, :], origin="lower", aspect="auto", vmin=-0.3, vmax=0.3, cmap="RdBu_r")
        plt.colorbar()
        plt.show()

    # %%
    years = [2023]

    for year in years:
        num_jday = 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365

        for jday in range(1, num_jday + 1):

            if not fs.exists(f"{bucket}/{folder}/{year}/template_{jday:03d}.dat"):
                continue

            with fs.open(f"{bucket}/{folder}/{year}/cctorch_picks_{jday:03d}.csv", "r") as fp:
                picks = pd.read_csv(fp, dtype=str)
            with fs.open(f"{bucket}/{folder}/{year}/cctorch_events_{jday:03d}.csv", "r") as fp:
                events = pd.read_csv(fp, dtype=str)
            with fs.open(f"{bucket}/{folder}/{year}/config_{jday:03d}.json", "r") as fp:
                config = json.load(fp)
            template_file = fs.open(f"{bucket}/{folder}/{year}/template_{jday:03d}.dat", "rb")
            templates = np.frombuffer(template_file.read(), dtype=np.float32).reshape(tuple(config["template_shape"]))
            template_file.close()

            print(f"events: {len(events):,} ")
            print(f"picks: {len(picks):,} ")
            print(f"templates: {templates.shape}")

            picks.to_csv(f"{result_path}/picks_{year:04d}_{jday:03d}.csv", index=False)
            events.to_csv(f"{result_path}/events_{year:04d}_{jday:03d}.csv", index=False)
            np.save(f"{result_path}/templates_{year:04d}_{jday:03d}.npy", templates)

            plot_templates(templates, events, picks)

            # break

# %%

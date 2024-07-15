# %%
import multiprocessing as mp

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# %%
# for folder in ["/nfs/quakeflow_dataset/NC/quakeflow_nc", "/nfs/quakeflow_dataset/NC"]:
# for mode in ["", "_train", "_test"]:
def process(i, folder, mode):
    h5_file = f"{folder}/waveform{mode}.h5"
    print(f"Processing {h5_file}")

    events_df = []
    picks_df = []
    with h5py.File(h5_file, "r") as f:
        event_ids = list(f.keys())
        for event_id in tqdm(event_ids, desc=f"{h5_file}", position=i):
            event_attrs = dict(f[event_id].attrs)
            events_df.append(event_attrs)

            station_ids = list(f[event_id].keys())
            for station_id in station_ids:
                station_attrs = dict(f[event_id][station_id].attrs)
                station_attrs["event_id"] = event_id
                station_attrs["station_id"] = station_id
                picks_df.append(station_attrs)

    events_df = pd.DataFrame(events_df)
    picks_df = pd.DataFrame(picks_df)
    events_df.to_csv(f"{folder}/events{mode}.csv", index=False)
    picks_df.to_csv(f"{folder}/picks{mode}.csv", index=False)


# %%
folders = [
    "/nfs/quakeflow_dataset/NC/quakeflow_nc",
    "/nfs/quakeflow_dataset/SC/quakeflow_sc",
    "/nfs/quakeflow_dataset/NC",
    "/nfs/quakeflow_dataset/SC",
]
mode = ["", "_train", "_test"]
inputs = [(folder, m) for folder in folders for m in mode]

pbar = tqdm(total=len(inputs))
callback = lambda *args: pbar.update()

with mp.Pool(len(inputs)) as pool:
    jobs = []
    for i, (folder, mode) in enumerate(inputs):
        job = pool.apply_async(
            process,
            args=(
                i,
                folder,
                mode,
            ),
            callback=callback,
        )
        jobs.append(job)
    pool.close()
    pool.join()

    results = [job.get() for job in jobs]

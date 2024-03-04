# %%
import multiprocessing as mp
import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
data_path = "waveform_ps_h5"
result_path = "data"
file_list = sorted(glob(f"{data_path}/*.h5"))
# %%
file_size = {file: os.path.getsize(file)/1e9 for file in file_list}

# %%
MAX_SIZE = 45 # GB
for file, size in file_size.items():
    if size > MAX_SIZE:
        # split into smaller files
        NUM_FILES = int(np.ceil(size / MAX_SIZE))
        with h5py.File(file, "r") as f:
            event_ids = list(f.keys())
            for event_id in tqdm(event_ids, desc=f"Processing {file}"):
                index = int(event_id[-1]) % NUM_FILES
                # with h5py.File(f"{result_path}/{file.split('/')[-1].replace('.h5', '')}_{index}.h5", "a") as g:
                with h5py.File(f"{data_path}/{file.split('/')[-1].replace('.h5', '')}_{index}.h5", "a") as g:
                    if event_id in g:
                        print(f"Event {event_id} already exists in {file.split('/')[-1].replace('.h5', '')}_{index}.h5")
                        continue
                    # copy 
                    f.copy(event_id, g)
    # else:
    #     print(f"Copying {file} to {result_path}")
    #     os.system(f"cp {file} {result_path}")
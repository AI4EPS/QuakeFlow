# %%
import os

import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
h5_dir = "waveform_ps"
h5_out = "waveform_ps.h5"
h5_train = "waveform_ps_train.h5"
h5_test = "waveform_ps_test.h5"

# %%
h5_dir = "waveform_h5"
h5_out = "waveform.h5"
h5_train = "waveform_train.h5"
h5_test = "waveform_test.h5"

h5_files = sorted(os.listdir(h5_dir))
train_files = h5_files[:-1]
test_files = h5_files[-1:]
print(f"train files: {train_files}")
print(f"test files: {test_files}")

# %%
# with h5py.File(h5_out, "w") as fp:
#     # external linked file
#     for h5_file in h5_files:
#         with h5py.File(os.path.join(h5_dir, h5_file), "r") as f:
#             for event in tqdm(f.keys(), desc=h5_file, total=len(f.keys())):
#                 if event not in fp:
#                     fp[event] = h5py.ExternalLink(os.path.join(h5_dir, h5_file), event)
#                 else:
#                     print(f"{event} already exists")
#                     continue

# %%
with h5py.File(h5_train, "w") as fp:
    # external linked file
    for h5_file in train_files:
        with h5py.File(os.path.join(h5_dir, h5_file), "r") as f:
            for event in tqdm(f.keys(), desc=h5_file, total=len(f.keys())):
                if event not in fp:
                    fp[event] = h5py.ExternalLink(os.path.join(h5_dir, h5_file), event)
                else:
                    print(f"{event} already exists")
                    continue

with h5py.File(h5_test, "w") as fp:
    # external linked file
    for h5_file in test_files:
        with h5py.File(os.path.join(h5_dir, h5_file), "r") as f:
            for event in tqdm(f.keys(), desc=h5_file, total=len(f.keys())):
                if event not in fp:
                    fp[event] = h5py.ExternalLink(os.path.join(h5_dir, h5_file), event)
                else:
                    print(f"{event} already exists")
                    continue

# %%
# # check h5 file
# with h5py.File("waveform_ps.h5", "r") as fp:
#     for event in tqdm(fp.keys(), total=len(fp.keys())):
#         for station in fp[event].keys():
#             print(fp[event][station].shape)
#             raise
#         raise
#             plt.figure()
#             plt.plot(fp[event][station][0, :])
#             plt.show()
#             raise
#         raise

# %%

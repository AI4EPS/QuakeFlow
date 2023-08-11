# %%
from pathlib import Path
import os
import torch

# %%
region = "BayArea"
root_path = Path(region)
data_path = root_path / "obspy"
result_path = root_path / "phasenet"
if not result_path.exists():
    result_path.mkdir()

# %%
waveform_path = data_path / "waveforms"
mseed_list = sorted(list(waveform_path.rglob("*.mseed")))
file_list = []
for f in mseed_list:
    file_list.append(str(f))

file_list = sorted(list(set(file_list)))
with open(result_path / "data_list.txt", "w") as fp:
    fp.write("\n".join(file_list))


# %%
num_gpu = torch.cuda.device_count()
if num_gpu == 0:
    cmd = f"python ../EQNet/predict.py --model phasenet --add_polarity --add_event --format mseed --data_list {result_path/'data_list.txt'} --batch_size 1 --result_path {result_path} --device=cpu"
elif num_gpu == 1:
    cmd = f"python ../EQNet/predict.py --model phasenet --add_polarity --add_event --format mseed --data_list {result_path/'data_list.txt'} --batch_size 1 --result_path {result_path}"
else:
    cmd = f"torchrun --standalone --nproc_per_node {num_gpu}  ../EQNet/predict.py --model phasenet --add_polarity --add_event --format mseed --data_list {result_path/'data_list.txt'} --batch_size 1 --result_path {result_path}"
print(cmd)
os.system(cmd)

# # %%
# cmd = f"gsutil -m cp -r {result_path}/picks_phasenet {protocol}{bucket}/{folder}/phasenet/picks"
# print(cmd)
# os.system(cmd)

# %%

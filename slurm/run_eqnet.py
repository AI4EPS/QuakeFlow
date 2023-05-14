# %%
from pathlib import Path
import os
import torch

# %%
# region = "Hawaii_Loa"
region = "South_Pole"
region = "South_Pole2"
root_path = Path(region)

# %%
mseed_path = root_path / "waveforms"
mseeds = sorted(list(mseed_path.rglob("*.mseed")))
file_list = []
for f in mseeds:
    file_list.append(str(f).split(".mseed")[0][:-1]+"*.mseed")

file_list = sorted(list(set(file_list)))

# %%
with open("mseed_list.txt", "w") as fp:
    fp.write("\n".join(file_list))

# %%
num_gpu = torch.cuda.device_count()

# %%
# os.system(f"torchrun --standalone --nproc_per_node 4 ../EQNet/predict.py --model phasenet --add_polarity --add_event --data_path ./ --data_list mseed_list.txt  --response_xml '{root_path}/stations/*xml'  --result_path ./eqnet_picks --batch_size=1 --format mseed")
os.system(f"torchrun --standalone --nproc_per_node {num_gpu} ../EQNet/predict.py --model phasenet --add_polarity --add_event --data_path ./ --data_list mseed_list.txt  --response_xml '{root_path}/response.xml'  --result_path {root_path}/eqnet_picks --batch_size=1 --format mseed")

os.system(f"cp {root_path}/eqnet_picks/picks_phasenet_raw.csv {root_path}/results/picks_eqnet.csv")

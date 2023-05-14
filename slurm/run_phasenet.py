# %%
from pathlib import Path
import os

# %%
# region = "Hawaii_Loa"
region = "South_Pole"
root_path = Path(region)

# %%
waveform_path = root_path / "waveforms"
mseed_list = sorted(list(waveform_path.rglob("*.mseed")))
file_list = []
for f in mseed_list:
    file_list.append(str(f).split(".mseed")[0][:-1]+"*.mseed")

file_list = sorted(list(set(file_list)))

# %%
with open("mseed_list.csv", "w") as fp:
    fp.write("fname\n")
    fp.write("\n".join(file_list))

# %%
# os.system(f"python ../PhaseNet/phasenet/predict.py --model=../PhaseNet/model/190703-214543 --data_list=mseed.csv --data_dir=./ --format=mseed --amplitude --response_xml={root_path}/stations/*xml --batch_size=1")
os.system(f"python ../PhaseNet/phasenet/predict.py --model=../PhaseNet/model/190703-214543 --data_list=mseed_list.csv --data_dir=./ --format=mseed --amplitude --response_xml={root_path}/response.xml --result_dir={root_path}/phasenet_picks/  --batch_size=1")

os.system(f"cp {root_path}/phasenet_picks/picks.csv {root_path}/results/picks_phasenet.csv")

# %%
from pathlib import Path
import os

# %%
# region = "Hawaii_Loa"
# region = "South_Pole"
# region = "Kilauea"
region = "Kilauea_debug"
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
    file_list.append(str(f).split(".mseed")[0][:-1]+"*.mseed")

file_list = sorted(list(set(file_list)))

# %%
with open(result_path / "mseed_list.csv", "w") as fp:
    fp.write("fname\n")
    fp.write("\n".join(file_list))

# %%
# os.system(f"python ../PhaseNet/phasenet/predict.py --model=../PhaseNet/model/190703-214543 --data_list=mseed.csv --data_dir=./ --format=mseed --amplitude --response_xml={root_path}/stations/*xml --batch_size=1")
os.system(f"python ../PhaseNet/phasenet/predict.py --model=../PhaseNet/model/190703-214543 --data_dir=./ --data_list={result_path}/mseed_list.csv --format=mseed --amplitude --response_xml={data_path}/inventory.xml --result_dir={result_path} --batch_size=1")

# os.system(f"cp {result_path}/picks.csv {result_path}/picks.csv")

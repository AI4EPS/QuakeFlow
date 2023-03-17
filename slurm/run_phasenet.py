# %%
from pathlib import Path
import os

# %%
mseed_path = Path("waveforms")
mseeds = sorted(list(mseed_path.rglob("*.mseed")))
file_list = []
for f in mseeds:
    file_list.append(str(f).split(".mseed")[0][:-1]+"*.mseed")

file_list = sorted(list(set(file_list)))

# %%
with open("mseed.csv", "w") as fp:
    fp.write("fname\n")
    fp.write("\n".join(file_list))


# %%
os.system("python ../PhaseNet/phasenet/predict.py --model=../PhaseNet/model/190703-214543 --data_list=mseed.csv --data_dir=./ --format=mseed --amplitude --response_xml=stations/*xml --batch_size=1")
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
with open("mseed_list.txt", "w") as fp:
    fp.write("\n".join(file_list))


# %%
os.system("torchrun --standalone --nproc_per_node 4 ../EQNet/predict.py --model phasenet --add_polarity --add_event --data_path ./ --data_list mseed_list.txt  --response_xml "stations/*xml"  --result_path ./eqnet_picks --batch_size=1 --format mseed")

os.system("cp eqnet_picks/picks_phasenet_raw.csv results/picks.csv")
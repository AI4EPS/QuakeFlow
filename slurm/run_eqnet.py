# %%
from pathlib import Path
import os

# %%
waveform_path = Path("waveforms")
mseed_list = sorted(list(waveform_path.rglob("*.mseed")))
mseeds = []
# mseeds = defaultdict(list)
for f in mseed_list:
    mseeds.append(str(f).split(".mseed")[0][:-1]+"*.mseed")
    # key = str(f).split(".mseed")[0]
    # mseeds[key[:-1]].append(key[-1])

# %%
with open("mseed_list.txt", "w") as fp:
    fp.write("\n".join(mseeds))


# %%
os.system("torchrun --standalone --nproc_per_node 4 ../EQNet/predict.py --model phasenet --add_polarity --add_event --data_path ./ --data_list mseed_list.txt  --response_xml "stations/*xml"  --result_path ./eqnet_picks --batch_size=1 --format mseed")

os.system("cp eqnet_picks/picks_phasenet_raw.csv results/picks.csv")
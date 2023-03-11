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
with open("mseed.csv", "w") as fp:
    fp.write("fname\n")
    fp.write("\n".join(mseeds))


# %%
os.system("python ../PhaseNet/phasenet/predict.py --model=../PhaseNet/model/190703-214543 --data_list=mseed.csv --data_dir=./ --format=mseed --amplitude --response_xml=stations/*xml --batch_size=1")
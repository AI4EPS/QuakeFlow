# %%
import os
from pathlib import Path

output_path = Path("templates")
dt_ct = Path("relocation/hypodd/dt.ct")
lines = []
with open(dt_ct, "r") as fp:
    for line in fp:
        if line.startswith("#"):
            ev1, ev2 = line.split()[1:3]
            lines.append(f"{ev1},{ev2}\n")

print(f"Number of pairs: {len(lines)}")
with open(output_path / "event_pair.txt", "w") as fp:
    fp.writelines(lines)

# %%
os.system("python ../CCTorch/run.py --pair-list=templates/event_pair.txt  --data-path=templates/template.dat --data-format=memmap --config=templates/config.json  --batch-size=4096  --result-path=templates/ccpairs")
# os.system("python ../CCTorch/run.py --data-list1=templates/event_index.txt  --data-path=templates/template.dat --data-format=memmap --config=templates/config.json  --batch-size=1024  --result-path=templates/ccpairs")
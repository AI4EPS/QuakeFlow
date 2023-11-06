# %%
import argparse
import os
from glob import glob
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    ## true
    parser.add_argument("--dtct_pair", action="store_true", help="run convert_dtcc.py")
    return parser.parse_args()


args = parse_args()

# %%
root_path = "local"
region = "demo"

output_path = f"{region}/templates"
if not os.path.exists(f"{root_path}/{output_path}"):
    os.makedirs(f"{root_path}/{output_path}")

mseed_list = glob(f"{root_path}/{region}/waveforms/????-???/??/")
mseed_list = [x + "*.mseed" for x in mseed_list]
with open(f"{root_path}/{region}/cctorch/mseed_list.txt", "w") as fp:
    fp.write("\n".join(mseed_list))

batch = 2048
if args.dtct_pair:
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
    os.system(
        "python ../CCTorch/run.py --pair_list=templates/event_pair.txt  --data_path=templates/template.dat --data_format=memmap --config=templates/config.json  --batch_size=512  --result_path=templates/ccpairs"
    )

else:
    num_gpu = torch.cuda.device_count()
    base_cmd = f"../CCTorch/run.py --data_list1={root_path}/{region}/cctorch/event_index.txt --data_path1={root_path}/{region}/cctorch/template.dat --data_format1=memmap --config={root_path}/{region}/cctorch/config.json  --batch_size={batch}  --result_path={root_path}/{region}/cctorch/ccpairs"
    if num_gpu == 0:
        if os.uname().sysname == "Darwin":
            os.system(f"python {base_cmd} --device=mps")
        else:
            os.system(f"python {base_cmd} --device=cpu")
    else:
        os.system(f"torchrun --standalone --nproc_per_node {num_gpu} {base_cmd}")

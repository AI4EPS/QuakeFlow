# %%
import argparse
import os
import sys
from glob import glob
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", nargs="?", type=str, default="local", help="root path")
    parser.add_argument("region", nargs="?", type=str, default="demo", help="region")
    parser.add_argument("--dtct_pair", action="store_true", help="run convert_dtcc.py")
    return parser.parse_args()


args = parse_args()

# %%
root_path = args.root_path
region = args.region

result_path = f"{region}/templates"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")


## based on GPU memory

batch = 1_024
block_size1 = 10_000
block_size2 = 10_000  # ~7GB

if args.dtct_pair:
    dt_ct = f"{root_path}/{region}/hypodd/dt.ct"
    pair_list = f"{root_path}/{region}/hypodd/event_pairs.txt"
    lines = []
    with open(dt_ct, "r") as fp:
        for line in fp:
            if line.startswith("#"):
                ev1, ev2 = line.split()[1:3]
                if ev1 > ev2:
                    ev1, ev2 = ev2, ev1
                lines.append(f"{ev1},{ev2}\n")

    print(f"Number of pairs from hypodd dt.ct: {len(lines)}")
    with open(f"{root_path}/{region}/hypodd/event_pairs.txt", "w") as fp:
        fp.writelines(lines)
    base_cmd = f"../CCTorch/run.py --pair_list={root_path}/{region}/hypodd/event_pairs.txt --data_path1={root_path}/{region}/cctorch/template.dat --data_format1=memmap --config={root_path}/{region}/cctorch/config.json  --batch_size={batch} --block_size1={block_size1} --block_size2={block_size2} --result_path={root_path}/{region}/cctorch/ccpairs"

else:
    base_cmd = f"../CCTorch/run.py --pair_list={root_path}/{region}/cctorch/event_pairs.txt --data_path1={root_path}/{region}/cctorch/template.dat --data_format1=memmap --config={root_path}/{region}/cctorch/config.json  --batch_size={batch} --block_size1={block_size1} --block_size2={block_size2} --result_path={root_path}/{region}/cctorch/ccpairs"
num_gpu = torch.cuda.device_count()
if num_gpu == 0:
    if os.uname().sysname == "Darwin":
        os.system(f"python {base_cmd} --device=mps")
    else:
        os.system(f"python {base_cmd} --device=cpu")
else:
    os.system(f"torchrun --standalone --nproc_per_node {num_gpu} {base_cmd}")

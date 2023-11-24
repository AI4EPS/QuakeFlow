import argparse
import os
import sys
from glob import glob
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="local", help="root path")
    parser.add_argument("--region", type=str, default="demo", help="region")
    return parser.parse_args()


args = parse_args()

# %%
root_path = args.root_path
region = args.region

result_path = f"{region}/adloc"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")

batch = 100

base_cmd = f"../ADLoc/run.py --config {root_path}/{region}/config.json --stations {root_path}/{region}/obspy/stations.json --events {root_path}/{region}/gamma/gamma_events.csv --picks {root_path}/{region}/gamma/gamma_picks.csv --result_path {root_path}/{region}/adloc --batch_size {batch}"
os.system(f"python {base_cmd} --device=cpu --epochs=1")

# num_gpu = torch.cuda.device_count()
# if num_gpu == 0:
#     if os.uname().sysname == "Darwin":
#         os.system(f"python {base_cmd} --device=mps")
#     else:
#         os.system(f"python {base_cmd} --device=cpu")
# else:
#     os.system(f"torchrun --standalone --nproc_per_node {num_gpu} {base_cmd}")

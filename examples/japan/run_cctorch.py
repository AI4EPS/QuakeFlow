# %%
import os

import torch

# %%
# root_path = args.root_path
# region = args.region
root_path = "local"
region = "hinet"

data_path = f"{region}/cctorch"
result_path = f"{region}/cctorch/ccpairs"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")


## based on GPU memory

batch = 1_024
block_size1 = 1000_000
block_size2 = 1000_000


base_cmd = (
    f"../../CCTorch/run.py --pair_list={root_path}/{region}/cctorch/pairs.txt --data_path1={root_path}/{region}/cctorch/template.dat --data_format1=memmap "
    f"--data_list1={root_path}/{region}/cctorch/cctorch_picks.csv "
    f"--events_csv={root_path}/{region}/cctorch/cctorch_events.csv --picks_csv={root_path}/{region}/cctorch/cctorch_picks.csv --stations_csv={root_path}/{region}/cctorch/cctorch_stations.csv "
    f"--config={root_path}/{region}/cctorch/config.json  --batch_size={batch} --block_size1={block_size1} --block_size2={block_size2} --result_path={root_path}/{result_path}"
)

num_gpu = torch.cuda.device_count()
if num_gpu == 0:
    if os.uname().sysname == "Darwin":
        cmd = f"python {base_cmd} --device=cpu"
    else:
        cmd = f"python {base_cmd} --device=cpu"
else:
    cmd = f"torchrun --standalone --nproc_per_node {num_gpu} {base_cmd}"
print(cmd)
os.system(cmd)

# %%
os.chdir(f"{root_path}/{region}/cctorch")
source_file = f"ccpairs/CC_{num_gpu:03d}_dt.cc"
target_file = f"dt.cc"
print(f"{source_file} -> {target_file}")
if os.path.lexists(target_file):
    os.remove(target_file)
os.symlink(source_file, target_file)

source_file = f"ccpairs/CC_{num_gpu:03d}.csv"
target_file = f"dtcc.csv"
print(f"{source_file} -> {target_file}")
if os.path.lexists(target_file):
    os.remove(target_file)
os.symlink(source_file, target_file)

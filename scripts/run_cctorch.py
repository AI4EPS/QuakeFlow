# %%
import os

import torch
from args import parse_args

args = parse_args()
# %%
root_path = args.root_path
region = args.region

data_path = f"{region}/cctorch"
result_path = f"{region}/cctorch/ccpairs"

# data_path = f"{region}/cctorch_ca"
# result_path = f"{region}/cctorch_ca/ccpairs"

# data_path = f"{region}/cctorch_gamma"
# result_path = f"{region}/cctorch_gamma/ccpairs"

if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")


## based on GPU memory

batch = 1_024
block_size1 = 1000_000
block_size2 = 1000_000

if args.dtct_pair:
    dt_ct = f"{root_path}/{region}/hypodd/dt.ct"
    pair_list = f"{root_path}/{region}/hypodd/pairs.txt"
    lines = []
    with open(dt_ct, "r") as fp:
        for line in fp:
            if line.startswith("#"):
                ev1, ev2 = line.split()[1:3]
                if ev1 > ev2:
                    ev1, ev2 = ev2, ev1
                lines.append(f"{ev1},{ev2}\n")

    print(f"Number of pairs from hypodd dt.ct: {len(lines)}")
    with open(f"{root_path}/{region}/hypodd/pairs.txt", "w") as fp:
        fp.writelines(lines)
    base_cmd = f"../CCTorch/run.py --pair_list={root_path}/{region}/hypodd/pairs.txt --data_path1={root_path}/{region}/cctorch/template.dat --data_format1=memmap --config={root_path}/{region}/cctorch/config.json  --batch_size={batch} --block_size1={block_size1} --block_size2={block_size2} --result_path={root_path}/{result_path}"

else:
    base_cmd = (
        f"../CCTorch/run.py --pair_list={root_path}/{data_path}/pairs.txt --data_path1={root_path}/{data_path}/template.dat --data_format1=memmap "
        f"--data_list1={root_path}/{data_path}/cctorch_picks.csv "
        f"--events_csv={root_path}/{data_path}/cctorch_events.csv --picks_csv={root_path}/{data_path}/cctorch_picks.csv --stations_csv={root_path}/{data_path}/cctorch_stations.csv "
        f"--config={root_path}/{data_path}/config.json  --batch_size={batch} --block_size1={block_size1} --block_size2={block_size2} "
        f"--result_path={root_path}/{result_path}"
    )


if torch.cuda.is_available():
    device = "cuda"
    num_gpu = torch.cuda.device_count()
elif torch.backends.mps.is_available():
    device = "mps"
    num_gpu = 0
else:
    device = "cpu"
    num_gpu = 0

if num_gpu > 0:
    cmd = f"torchrun --standalone --nproc_per_node {num_gpu} {base_cmd} --device={device}"
else:
    cmd = f"python {base_cmd} --device={device}"
print(cmd)
os.system(cmd)

# %%
for rank in range(num_gpu):
    if not os.path.exists(f"{root_path}/{result_path}/CC_{rank:03d}_{num_gpu:03d}.csv"):
        continue
    if rank == 0:
        cmd = f"cat {root_path}/{result_path}/CC_{rank:03d}_{num_gpu:03d}.csv > {root_path}/{data_path}/dtcc.csv"
    else:
        cmd = (
            f"tail -n +2 {root_path}/{result_path}/CC_{rank:03d}_{num_gpu:03d}.csv >> {root_path}/{data_path}/dtcc.csv"
        )
    print(cmd)
    os.system(cmd)


cmd = f"cat {root_path}/{result_path}/CC_*_{num_gpu:03d}_dt.cc > {root_path}/{data_path}/dt.cc"
print(cmd)
os.system(cmd)

# # %%
# os.chdir(f"{root_path}/{region}/cctorch")
# source_file = f"ccpairs/CC_{num_gpu:03d}_dt.cc"
# target_file = f"dt.cc"
# print(f"{source_file} -> {target_file}")
# if os.path.lexists(target_file):
#     os.remove(target_file)
# os.symlink(source_file, target_file)

# source_file = f"ccpairs/CC_{num_gpu:03d}.csv"
# target_file = f"dtcc.csv"
# print(f"{source_file} -> {target_file}")
# if os.path.lexists(target_file):
#     os.remove(target_file)
# os.symlink(source_file, target_file)

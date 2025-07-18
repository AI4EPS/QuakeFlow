# %%
import argparse
import os

import torch
import json
import fsspec


##
def parse_args():
    parser = argparse.ArgumentParser(description="Run Gamma on NCEDC/SCEDC data")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--root_path", type=str, default="local")
    parser.add_argument("--region", type=str, default="Cal")
    parser.add_argument("--bucket", type=str, default="quakeflow_catalog")
    return parser.parse_args()


args = parse_args()

# ########## DEBUG by day ##############
# year = 2019
# jday = 185
# jday = 186
# jday = 187
# data_path = f"{region}/cctorch/{year}"
# #######################################
protocol = "gs"
token_json = f"application_default_credentials.json"
with open(token_json, "r") as fp:
    token = json.load(fp)

fs = fsspec.filesystem(protocol, token=token)

# %%
root_path = args.root_path
region = args.region
bucket = args.bucket
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
block_size1 = 2_000_000
block_size2 = 2_000_000

########## DEBUG by day ##############
# base_cmd = (
#     f"../../CCTorch/run.py --pair_list={root_path}/{data_path}/pairs_{jday:03d}.txt --data_path1={root_path}/{data_path}/template_{jday:03d}.dat --data_format1=memmap "
#     f"--data_list1={root_path}/{data_path}/cctorch_picks_{jday:03d}.csv "
#     f"--events_csv={root_path}/{data_path}/cctorch_events_{jday:03d}.csv --picks_csv={root_path}/{data_path}/cctorch_picks_{jday:03d}.csv --stations_csv={root_path}/{data_path}/cctorch_stations_{jday:03d}.csv "
#     f"--config={root_path}/{data_path}/config_{jday:03d}.json  --batch_size={batch} --block_size1={block_size1} --block_size2={block_size2} "
#     f"--result_path={root_path}/{result_path}"
# )
#######################################


base_cmd = (
    f"../../CCTorch/run.py --pair_list={root_path}/{data_path}/pairs.txt --data_path1={root_path}/{data_path}/template.dat --data_format1=memmap "
    # f"/opt/CCTorch/run.py --pair_list={root_path}/{data_path}/pairs.txt --data_path1={root_path}/{data_path}/template.dat --data_format1=memmap "
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
# ########## DEBUG by day ##############
# origin = data_path
# data_path = f"{data_path}/../"
# #######################################

if num_gpu == 0:
    cmd = f"cat {root_path}/{result_path}/CC_000_001.csv > {root_path}/{data_path}/dtcc.csv"
    print(cmd)
    os.system(cmd)
    cmd = f"cat {root_path}/{result_path}/CC_000_001_dt.cc > {root_path}/{data_path}/dt.cc"
    print(cmd)
    os.system(cmd)

    cmd = f"cat {root_path}/{result_path}/CC_000_001_dt.cc > {root_path}/{data_path}/dt.cc"
    print(cmd)
    os.system(cmd)

else:
    for rank in range(num_gpu):
        if not os.path.exists(f"{root_path}/{result_path}/CC_{rank:03d}_{num_gpu:03d}.csv"):
            continue
        if rank == 0:
            cmd = f"cat {root_path}/{result_path}/CC_{rank:03d}_{num_gpu:03d}.csv > {root_path}/{data_path}/dtcc.csv"
        else:
            cmd = f"tail -n +2 {root_path}/{result_path}/CC_{rank:03d}_{num_gpu:03d}.csv >> {root_path}/{data_path}/dtcc.csv"
        print(cmd)
        os.system(cmd)

    cmd = f"cat {root_path}/{result_path}/CC_*_{num_gpu:03d}_dt.cc > {root_path}/{data_path}/dt.cc"
    print(cmd)
    os.system(cmd)

# ########## DEBUG by day ##############
# data_path = origin
# #######################################

########## DEBUG by day ##############
# ##
# cmd = f"cp {root_path}/{data_path}/cctorch_stations_{jday:03d}.csv {root_path}/{data_path}/../cctorch_stations.csv"
# print(cmd)
# os.system(cmd)

# cmd = f"cp {root_path}/{data_path}/cctorch_events_{jday:03d}.csv {root_path}/{data_path}/../cctorch_events.csv"
# print(cmd)
# os.system(cmd)

# cmd = f"cp {root_path}/{data_path}/cctorch_picks_{jday:03d}.csv {root_path}/{data_path}/../cctorch_picks.csv"
# print(cmd)
# os.system(cmd)
#######################################


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

if protocol == "gs":
    print(f"{root_path}/{data_path}/dt.cc -> {bucket}/{data_path}/dt.cc")
    fs.put(f"{root_path}/{data_path}/dt.cc", f"{bucket}/{data_path}/dt.cc")
    print(f"{root_path}/{data_path}/dtcc.csv -> {bucket}/{data_path}/dtcc.csv")
    fs.put(f"{root_path}/{data_path}/dtcc.csv", f"{bucket}/{data_path}/dtcc.csv")
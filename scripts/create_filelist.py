# %%
import os
from glob import glob

# %%
protocol = "file"
token = None

## get from command line
root_path = "local"
region = "demo"
if len(os.sys.argv) > 1:
    root_path = os.sys.argv[1]
    region = os.sys.argv[2]
print(f"root_path: {root_path}")
print(f"region: {region}")

# %%
result_path = f"{region}/phasenet_das"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}", exist_ok=True)

# %%
folder_depth = 2
csv_list = sorted(glob(f"{root_path}/{result_path}/picks_phasenet_das/????-??-??/*.csv"))
csv_list = ["/".join(x.split("/")[-folder_depth:]) for x in csv_list]

# %%
hdf5_list = sorted(glob(f"{root_path}/{region}/????-??-??/*.h5"))
num_to_process = 0
with open(f"{root_path}/{result_path}/filelist.csv", "w") as fp:
    # fp.write("\n".join(hdf5_list))
    for line in hdf5_list:
        csv_name = "/".join(line.split("/")[-folder_depth:]).replace(".h5", ".csv")
        if csv_name not in csv_list:
            fp.write(f"{line}\n")
            num_to_process += 1

print(f"filelist.csv created in {root_path}/{result_path}: {num_to_process} / {len(hdf5_list)} to process")

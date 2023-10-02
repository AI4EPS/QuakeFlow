# %%
import os
from pathlib import Path
import argparse
import torch
from glob import glob


# %%
root_path = "local"
region = "demo"

output_path = f"{region}/cctorch"
if not os.path.exists(f"{root_path}/{output_path}"):
    os.makedirs(f"{root_path}/{output_path}")

# mseed_list = glob(f"{root_path}/{region}/waveforms/????-???/??/")
# mseed_list = [x + "*.mseed" for x in mseed_list]
# with open(f"{root_path}/{region}/cctorch/mseed_list.txt", "w") as fp:
#     fp.write("\n".join(mseed_list))


num_gpu = torch.cuda.device_count()
base_cmd = f"python ../CCTorch/run.py --mode=TM --data_list1={root_path}/{region}/cctorch/mseed_list.txt --data_list2={root_path}/{region}/cctorch/event_index.txt --data_path2={root_path}/{region}/cctorch/template.dat --data_format1=mseed --data_format2=memmap --config={root_path}/{region}/cctorch/config.json  --batch_size=1 --block_size1=1 --block_size2=32 --reduce_x --shift_t --normalize --maxlag=0 --result_path={root_path}/{output_path}"
if num_gpu == 0:
    os.system(f"{base_cmd} --device=cpu")
    # os.system(f"{base_cmd} --device=mps")
else:
    os.system(base_cmd)

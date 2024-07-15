# %%
import argparse
import json
import logging
import os
import time
from tqdm import tqdm
import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# %%
NUM_NODES = 32  # < 2004
# NUM_NODES = 128 # >= 2004

###### Hardcoded #######
token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
with open(token_json, "r") as fp:
    token = json.load(fp)
fs = fsspec.filesystem("gs", token=token)

# for year in tqdm(range(1986, 1999)[::-1]):
#     cmd = f"python run_phasenet.py --region NC --year {year} --num_nodes 1"
#     os.system(cmd)

# raise

for year in range(1999, 2005)[::-1]:

    cmds = [
        f"python submit_phasenet.py --region NC --branch ncedc --year {year} --num_nodes {NUM_NODES}",
        f"python submit_phasenet.py --region SC --branch scedc --year {year} --num_nodes {NUM_NODES}",
    ]

    # while True:
    if True:
        for cmd in cmds:
            logging.info(f"Running: {cmd}")
            os.system(cmd)
            logging.info("Sleeping for 1 minutes...")
            time.sleep(60)

        finish = True
        for REGION in ["NC", "SC"]:
            for NODE_RANK in range(NUM_NODES):
                mseed_file = (
                    f"gs://quakeflow_catalog/{REGION}/phasenet/mseed_list/{year}_{NODE_RANK:03d}_{NUM_NODES:03d}.txt"
                )
                if fs.exists(mseed_file):
                    with fs.open(mseed_file, "r") as fp:
                        mseed_list = fp.readlines()
                    if len(mseed_list) > 0:
                        print(f"{mseed_file}, {len(mseed_list) = }")
                        finish = False
                        break
        if finish:
            break

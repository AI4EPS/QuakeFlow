import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import sky
from tqdm import tqdm
import pandas as pd

###### Hardcoded #######
token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
with open(token_json, "r") as fp:
    token = json.load(fp)
fs = fsspec.filesystem("gs", token=token)
###### Hardcoded #######


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=16)
    return parser.parse_args()


args = parse_args()
NUM_NODES = args.num_nodes

task = sky.Task(
    name=f"run-download",
    setup="""
echo "Begin setup."                                                           
echo export WANDB_API_KEY=$WANDB_API_KEY >> ~/.bashrc
pip install h5py tqdm wandb pandas scipy numpy==1.26.4
pip install fsspec gcsfs s3fs                                                   
pip install obspy pyproj
""",
    run="""
num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    ls -al /opt
    ls -al /data
    ls -al ./
fi
echo "Downloading waveforms on (node_rank, num_node) = ($NODE_RANK, $NUM_NODE)"
python download_waveform.py --num_node $NUM_NODE --node_rank $NODE_RANK
""",
    workdir=".",
    num_nodes=1,
    envs={"NUM_NODE": NUM_NODES, "NODE_RANK": 0},
)
task.set_resources(
    sky.Resources(
        cloud=sky.GCP(),
        region="us-west1",  # GCP
        # region="us-west-2",  # AWS
        accelerators=None,
        cpus=2,
        disk_tier="low",
        disk_size=10,  # GB
        memory=None,
        use_spot=True,
    ),
)

jobs = []
try:
    sky.status(refresh="AUTO")
except Exception as e:
    print(e)

job_idx = 1
requests_ids = []
for NODE_RANK in range(NUM_NODES):

    task.update_envs({"NODE_RANK": NODE_RANK})
    cluster_name = f"download-{NODE_RANK:03d}"

    requests_ids.append(sky.jobs.launch(task, name=f"{cluster_name}"))

    print(f"Running download waveform on (rank={NODE_RANK}, num_node={NUM_NODES}) of {cluster_name}")

    job_idx += 1

for request_id in requests_ids:
    print(sky.get(request_id))

# with ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
#     for NODE_RANK in range(NUM_NODES):

#         task.update_envs({"NODE_RANK": NODE_RANK})
#         cluster_name = f"download-{NODE_RANK:03d}"

# status = sky.status(cluster_names=[f"{cluster_name}"], refresh="AUTO")
# print(f"{status = }")
# if len(status) > 0:
#     if status[0]["status"].value == "INIT":
#         sky.down(f"{cluster_name}")
#     if (not status[0]["to_down"]) and (not status[0]["status"].value == "INIT"):
#         sky.autostop(f"{cluster_name}", idle_minutes=10, down=True)
#     print(f"Cluster {cluster_name}/{NUM_NODES} already exists.")
#     continue

# status = sky.status(cluster_names=[f"{cluster_name}"])
# if len(status) == 0:
#     print(f"Launching cluster {cluster_name}/{NUM_NODES}...")
#     jobs.append(
#         executor.submit(
#             sky.launch,
#             task,
#             cluster_name=f"{cluster_name}",
#             idle_minutes_to_autostop=10,
#             down=True,
#             detach_setup=True,
#             detach_run=True,
#         )
#     )
#     time.sleep(10)


# for job in jobs:
#     print(job.result())

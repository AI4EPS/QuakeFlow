import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import sky
from tqdm import tqdm

# NUM_NODES = 128
# YEAR = 2022
# REGION = "NC"
# BRANCH = "ncedc"
# REGION = "SC"
# BRANCH = "scedc"

###### Hardcoded #######
token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
with open(token_json, "r") as fp:
    token = json.load(fp)
fs = fsspec.filesystem("gs", token=token)
###### Hardcoded #######


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=128)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--region", type=str, default="NC")
    parser.add_argument("--branch", type=str, default="ncedc")
    return parser.parse_args()


args = parse_args()
NUM_NODES = args.num_nodes
YEAR = args.year
REGION = args.region
BRANCH = args.branch


task = sky.Task(
    name=f"run-phasenet-{REGION}-{YEAR}",
    setup="""
echo "Begin setup."                                                           
echo export WANDB_API_KEY=$WANDB_API_KEY >> ~/.bashrc
pip install h5py tqdm wandb pandas numpy scipy
pip install fsspec gcsfs s3fs                                                   
pip install obspy pyproj
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
### PhaseNet
pip install tensorflow==2.14.0
[ ! -d "PhaseNet" ] && git clone https://github.com/AI4EPS/PhaseNet.git
cd PhaseNet && git checkout $BRANCH && git pull origin $BRANCH && cd ..
### GaMMA
# pip install -e /opt/GaMMA
""",
    run="""
num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    ls -al /opt
    ls -al /data
    ls -al ./
fi
echo "Running PhaseNet on (node_rank, num_node) = ($NODE_RANK, $NUM_NODE)"
python run_phasenet.py --model_path PhaseNet --num_node $NUM_NODE --node_rank $NODE_RANK --region $REGION --year $YEAR
""",
    workdir=".",
    num_nodes=1,
    envs={"NUM_NODE": NUM_NODES, "NODE_RANK": 0, "BRANCH": BRANCH, "REGION": REGION, "YEAR": YEAR},
)
task.set_resources(
    sky.Resources(
        cloud=sky.GCP(),
        region="us-west1",  # GCP
        # region="us-west-2",  # AWS
        accelerators=None,
        cpus=8,
        disk_tier="low",
        disk_size=50,  # GB
        memory=None,
        use_spot=True,
    ),
)
# task.set_file_mounts(
#     {
#         "run_phasenet.py": "run_phasenet.py",
#     },
# )

jobs = []
try:
    sky.status(refresh=True)
except Exception as e:
    print(e)

with ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
    for NODE_RANK in range(NUM_NODES):

        task.update_envs({"NODE_RANK": NODE_RANK})
        cluster_name = f"phasenet-{REGION}-{YEAR}-{NODE_RANK:03d}"

        status = sky.status(cluster_names=[f"{cluster_name}"], refresh=True)
        if len(status) > 0:
            if status[0]["status"].value == "INIT":
                sky.down(f"{cluster_name}")
            if (not status[0]["to_down"]) and (not status[0]["status"].value == "INIT"):
                sky.autostop(f"{cluster_name}", idle_minutes=10, down=True)
            print(f"Cluster {cluster_name}/{NUM_NODES} already exists.")
            continue

        ###### Hardcoded #######
        mseed_file = f"gs://quakeflow_catalog/{REGION}/phasenet/mseed_list/{YEAR}_{NODE_RANK:03d}_{NUM_NODES:03d}.txt"
        if fs.exists(mseed_file):
            with fs.open(mseed_file, "r") as fp:
                mseed_list = fp.readlines()
            print(f"{mseed_file}, {len(mseed_list) = }")
            if len(mseed_list) == 0:
                print(f"Skipping {mseed_file}...")
                continue
        ###### Hardcoded #######

        status = sky.status(cluster_names=[f"{cluster_name}"])
        if len(status) == 0:
            print(f"Launching cluster {cluster_name}/{NUM_NODES}...")
            jobs.append(
                executor.submit(
                    sky.launch,
                    task,
                    cluster_name=f"{cluster_name}",
                    idle_minutes_to_autostop=10,
                    down=True,
                    detach_setup=True,
                    detach_run=True,
                )
            )
            time.sleep(10)

for job in jobs:
    print(job.result())

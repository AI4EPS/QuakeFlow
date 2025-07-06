import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import sky


# NUM_NODES = 8
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=16)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--region", type=str, default="CA")
    return parser.parse_args()


args = parse_args()
NUM_NODES = args.num_nodes
YEAR = args.year
REGION = args.region

task = sky.Task(
    name="cut_template",
    setup="""
echo "Begin setup."                                                           
echo export WANDB_API_KEY=$WANDB_API_KEY >> ~/.bashrc
pip install -U h5py tqdm wandb pandas scipy scikit-learn numpy==1.26.4
pip install -U fsspec gcsfs s3fs                                                   
pip install -U obspy pyproj
# pip install -e /opt/ADLoc
pip install ADLoc
""",
    run="""
num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    ls -al /opt
    ls -al /data
    ls -al ./
    cat config.json
fi
python cut_templates_cc.py --num_node $NUM_NODES --node_rank $NODE_RANK --year $YEAR --region $REGION
""",
    workdir=".",
    num_nodes=1,
    envs={"NUM_NODES": NUM_NODES, "NODE_RANK": 0, "YEAR": YEAR, "REGION": REGION},
)

task.set_file_mounts(
    {
        # "/opt/ADLoc": "../../ADLoc",
        # "config.json": "local/Mendocino/config.json",
        # "config.json": "local/Cal/config.json",
    },
)
# task.set_storage_mounts({
#     '/remote/imagenet/': sky.Storage(name='my-bucket',
#                                      source='/local/imagenet'),
# })
task.set_resources(
    sky.Resources(
        cloud=sky.GCP(),
        region="us-west1",  # GCP
        # region="us-west-2",  # AWS
        accelerators=None,
        cpus=16,
        disk_tier="low",
        disk_size=50,  # GB
        memory="64+",
        use_spot=True,
    ),
)

# for NODE_RANK in range(NUM_NODES):
#     task.update_envs({"NODE_RANK": NODE_RANK})
#     cluster_name = f"cctorch-{NODE_RANK:02d}"
#     print(f"Launching cluster {cluster_name}-{NUM_NODES}...")
#     sky.jobs.launch(
#         task,
#         name=f"{cluster_name}",
#     )

jobs = []
try:
    sky.status(refresh="AUTO")
except Exception as e:
    print(e)

# task.update_envs({"NODE_RANK": 0})
# job_id = sky.launch(task, cluster_name="template", fast=True)
# # job_id = sky.exec(task, cluster_name="template")
# status = sky.stream_and_get(job_id)
# # sky.tail_logs(cluster_name="cctorch8", job_id=job_id, follow=True)
# print(f"Job ID: {job_id}, status: {status}")

# raise

job_idx = 1
requests_ids = []
for NODE_RANK in range(NUM_NODES):
    # for NODE_RANK in range(30):

    task.update_envs({"NODE_RANK": NODE_RANK})
    cluster_name = f"template-{NODE_RANK:03d}"

    requests_ids.append(sky.jobs.launch(task, name=f"{cluster_name}"))

    print(f"Running cut_template on (rank={NODE_RANK}, num_node={NUM_NODES}) of {cluster_name}")

    job_idx += 1

for request_id in requests_ids:
    print(sky.get(request_id))

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
    name="run_gamma",
    setup="""
echo "Begin setup."                                                           
echo export WANDB_API_KEY=$WANDB_API_KEY >> ~/.bashrc
pip install h5py tqdm wandb pandas numpy scipy
pip install fsspec gcsfs                                                    
pip install obspy pyproj
pip install -e /opt/GaMMA
""",
    run="""
num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    ls -al /opt
    ls -al /data
    ls -al ./
fi
python run_gamma.py --num_node $NUM_NODES --node_rank $NODE_RANK
""",
    workdir=".",
    num_nodes=1,
    envs={"NUM_NODES": NUM_NODES, "NODE_RANK": 0},
)

task.set_file_mounts(
    {
        "/opt/GaMMA": "../../GaMMA",
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
        memory=None,
        use_spot=False,
    ),
)

# for NODE_RANK in range(NUM_NODES):
#     task.update_envs({"NODE_RANK": NODE_RANK})
#     cluster_name = f"gamma-{NODE_RANK:02d}"
#     print(f"Launching cluster {cluster_name}-{NUM_NODES}...")
#     sky.jobs.launch(
#         task,
#         name=f"{cluster_name}",
#     )

jobs = []
sky.status(refresh=True)
with ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
    for NODE_RANK in range(NUM_NODES):
        task.update_envs({"NODE_RANK": NODE_RANK})
        cluster_name = f"gamma-{NODE_RANK:02d}"
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
                    detach_run=True,
                )
            )
            # jobs.append(
            #     executor.submit(
            #         sky.jobs.launch,
            #         task,
            #         name=f"{cluster_name}",
            #     )
            # )
            time.sleep(5)
        else:
            if not status[0]["to_down"]:
                sky.autostop(f"{cluster_name}", idle_minutes=10, down=True)
            print(f"Cluster {cluster_name}/{NUM_NODES} already exists.")

for job in jobs:
    print(job.result())

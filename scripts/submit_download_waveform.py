import time
from concurrent.futures import ThreadPoolExecutor

import sky
from args import parse_args

args = parse_args()
ROOT_PATH = args.root_path
REGION = args.region
PROTOCOL = args.protocol
BUCKET = args.bucket
TOKEN = args.token
NUM_NODES = args.num_nodes

task = sky.Task(
    name="download_waveform",
    setup="""
echo "Begin setup."
pip install obspy
pip install pandas numpy
pip install -U fsspec gcsfs s3fs
echo "Setup complete."
""",
    run="""
num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    ls -al /opt
    ls -al /data
    ls -al ./
fi
python download_waveform_v3.py --region $REGION --bucket $BUCKET --protocol $PROTOCOL --token $TOKEN  --num_nodes $NUM_NODES --node_rank $NODE_RANK
""",
    workdir=".",
    num_nodes=1,
    envs={
        "ROOT_PATH": ROOT_PATH,
        "REGION": REGION,
        "PROTOCOL": PROTOCOL,
        "BUCKET": BUCKET,
        "TOKEN": TOKEN,
        "NUM_NODES": NUM_NODES,
        "NODE_RANK": 0,
    },
)

task.set_file_mounts(
    {},
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
        cpus=2,
        disk_tier="low",
        disk_size=50,  # GB
        memory=None,
        use_spot=True,
    ),
)

jobs = []
try:
    sky.status(refresh=True)
except Exception as e:
    print(e)

with ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
    for NODE_RANK in range(NUM_NODES):

        task.update_envs({"NODE_RANK": NODE_RANK})
        cluster_name = f"obspy-{NODE_RANK:02d}-{NUM_NODES:02d}-{REGION}"

        status = sky.status(cluster_names=[f"{cluster_name}"], refresh=True)
        if len(status) > 0:
            if status[0]["status"].value == "INIT":
                sky.down(f"{cluster_name}")
            if (not status[0]["to_down"]) and (not status[0]["status"].value == "INIT"):
                sky.autostop(f"{cluster_name}", idle_minutes=10, down=True)
            print(f"Cluster {cluster_name}/{NUM_NODES} already exists.")
            continue

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
                    detach_setup=False,
                    detach_run=False,
                )
            )
            time.sleep(5)

for job in jobs:
    print(job.result())

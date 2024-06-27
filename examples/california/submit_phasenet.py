import sky

NUM_NODE = 3
REGION = "NC"
YEAR = 2022
BRANCH = "ncedc"

task = sky.Task(
    name="run_phasenet",
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
python run_phasenet.py --model_path PhaseNet --num_node $NUM_NODE --node_rank $NODE_RANK --region $REGION --year $YEAR
""",
    workdir=".",
    num_nodes=1,
    envs={"NUM_NODE": NUM_NODE, "NODE_RANK": 0, "BRANCH": BRANCH, "REGION": REGION, "YEAR": YEAR},
)
task.set_resources(
    sky.Resources(
        cloud=sky.GCP(),
        region="us-west1",  # GCP
        # region="us-west-2",  # AWS
        accelerators=None,
        cpus=16,
        memory=None,
        use_spot=True,
    ),
)
# task.set_file_mounts(
#     {
#         "run_phasenet.py": "run_phasenet.py",
#     },
# )

for NODE_RANK in range(NUM_NODE):
    task.update_envs({"NODE_RANK": NODE_RANK})
    if NODE_RANK < NUM_NODE - 1:
        # job_id, handle = sky.launch(task, cluster_name="mycluster")
        # job_id, handle = sky.launch(task)
        sky.jobs.launch(task)
    else:
        # job_id, handle = sky.launch(task, cluster_name="mycluster", detach_run=True)
        # job_id, handle = sky.launch(task, detach_run=True)
        sky.jobs.launch(task, detach_run=True)
    # print(f"Rank {NODE_RANK} running on job_id={job_id}: {handle}")

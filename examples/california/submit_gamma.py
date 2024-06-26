import sky

NUM_NODES = 64
NODE_RANK = 0

# task = sky.Task(run="echo hello SkyPilot")
task = sky.Task(
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
fi
python run_gamma.py --num_node {0} --node_rank {1}
""".format(
        NUM_NODES, NODE_RANK
    ),
    workdir=".",
    num_nodes=1,
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
        cpus=2,
        memory=None,
    ),
)
sky.launch(task, cluster_name="mycluster")
# sky.exec(task, cluster_name="mycluster")

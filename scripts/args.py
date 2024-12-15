import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="demo", help="region")
    parser.add_argument("--root_path", type=str, default="local", help="root path")

    ## Cloud
    parser.add_argument("--protocol", type=str, default="file", help="protocol (file, gs, s3)")
    parser.add_argument("--bucket", type=str, default=None, help="bucket name")
    parser.add_argument("--token", type=str, default=None, help="token")

    # parser.add_argument("--bucket", type=str, default="quakeflow_catalog", help="bucket name")
    # parser.add_argument("--protocol", type=str, default="gs", help="protocol (file, gs, s3)")
    # parser.add_argument("--token", type=str, default="application_default_credentials.json", help="token")

    ## Parallel
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="node rank")

    ## Model
    parser.add_argument("--model", type=str, default="phasenet", help="model")

    ## PhaseNet
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results")

    ## ADLOC
    parser.add_argument("--iter", type=int, default=0, help="iteration")

    ## CCTorch
    parser.add_argument("--dtct_pair", action="store_true", help="run convert_dtcc.py")

    return parser.parse_args()

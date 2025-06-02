import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gamma on NCEDC/SCEDC data")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--root_path", type=str, default="local")
    parser.add_argument("--region", type=str, default="Cal")
    parser.add_argument("--bucket", type=str, default="quakeflow_catalog")
    parser.add_argument("--config", type=str, default="local/Mendocino/config.json")
    return parser.parse_args()

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", nargs="?", type=str, default="local", help="root path")
    parser.add_argument("region", nargs="?", type=str, default="demo", help="region")

    ## CCTorch
    parser.add_argument("--dtct_pair", action="store_true", help="run convert_dtcc.py")

    return parser.parse_args()

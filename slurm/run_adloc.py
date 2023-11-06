from typing import Dict, List, NamedTuple

from kfp import dsl


@dsl.component(base_image="zhuwq0/quakeflow:latest")
def run_gamma(
    root_path: str,
    region: str,
    config: Dict,
    index: int = 0,
    picks_csv: str = "picks.csv",
    events_csv: str = "events.csv",
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> NamedTuple("outputs", events=str, picks=str):
    import json
    import os

    import fsspec
    import numpy as np
    import pandas as pd
    from pyproj import Proj

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    result_path = f"{region}/gamma"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    station_json = f"{region}/obspy/stations.json"
    # gamma_events_csv = f"{root_path}/gamma/gamma_events_{index:03d}.csv"
    # gamma_picks_csv = f"{root_path}/gamma/gamma_picks_{index:03d}.csv"

    ## read picks
    if protocol == "file":
        picks = pd.read_csv(f"{root_path}/{picks_csv}")
    else:
        picks = pd.read_csv(f"{protocol}://{bucket}/{picks_csv}")

    ## read stations
    if protocol == "file":
        stations = pd.read_json(f"{root_path}/{station_json}", orient="index")
    else:
        with fs.open(f"{bucket}/{station_json}", "r") as fp:
            stations = pd.read_json(fp, orient="index")


if __name__ == "__main__":
    # import fire

    # fire.Fire(run_gamma)

    import json
    import os
    import sys

    root_path = "local"
    region = "demo"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]
    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    run_gamma.python_func(
        root_path,
        region=region,
        config=config,
        picks_csv=f"{region}/gamma/gamma_picks.csv",
        events_csv=f"{region}/gamma/gamma_events.csv",
    )

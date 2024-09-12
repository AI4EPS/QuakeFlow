import json
import os
from typing import Dict, NamedTuple

import fsspec
import pandas as pd
from args import parse_args
from gamma.utils import association, estimate_eps
from pyproj import Proj


def run_gamma(
    root_path: str,
    region: str,
    config: Dict,
    node_rank: int = 0,
    num_nodes: int = 1,
    picks_csv: str = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> NamedTuple("outputs", events=str, picks=str):

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    data_path = f"{region}/phasenet"
    result_path = f"{region}/gamma"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    station_json = f"{region}/obspy/stations.json"
    if picks_csv is None:
        picks_csv = f"{data_path}/phasenet_picks_{node_rank:03d}_{num_nodes:03d}.csv"
    gamma_events_csv = f"{result_path}/gamma_events_{node_rank:03d}_{num_nodes:03d}.csv"
    gamma_picks_csv = f"{result_path}/gamma_picks_{node_rank:03d}_{num_nodes:03d}.csv"

    # %%
    ## read picks
    if protocol == "file":
        picks = pd.read_csv(f"{root_path}/{picks_csv}")
    else:
        picks = pd.read_csv(f"{protocol}://{bucket}/{picks_csv}")
    picks["id"] = picks["station_id"]
    picks["timestamp"] = picks["phase_time"]
    if "phase_amp" in picks.columns:
        picks["amp"] = picks["phase_amp"]
        picks["phase_amplitude"] = picks["phase_amp"]
    if "phase_amplitude" in picks.columns:
        picks["amp"] = picks["phase_amplitude"]
    picks["type"] = picks["phase_type"]
    picks["prob"] = picks["phase_score"]

    ## read stations
    if protocol == "file":
        stations = pd.read_json(f"{root_path}/{station_json}", orient="index")
    else:
        with fs.open(f"{bucket}/{station_json}", "r") as fp:
            stations = pd.read_json(fp, orient="index")
    stations["id"] = stations.index
    if "longitude0" not in config:
        config["longitude0"] = (config["minlongitude"] + config["maxlongitude"]) / 2
    if "latitude0" not in config:
        config["latitude0"] = (config["minlatitude"] + config["maxlatitude"]) / 2
    proj = Proj(f"+proj=sterea +lon_0={config['longitude0']} +lat_0={config['latitude0']} +units=km")
    stations[["x(km)", "y(km)"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x / 1e3)

    ### setting GMMA configs
    config["use_dbscan"] = True
    config["use_amplitude"] = True
    config["method"] = "BGMM"
    if config["method"] == "BGMM":  ## BayesianGaussianMixture
        config["oversample_factor"] = 5
    if config["method"] == "GMM":  ## GaussianMixture
        config["oversample_factor"] = 1

    # earthquake location
    config["vel"] = {"p": 6.0, "s": 6.0 / 1.75}
    config["dims"] = ["x(km)", "y(km)", "z(km)"]
    xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
    xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
    # zmin, zmax = config["mindepth"], config["maxdepth"]
    zmin = config["mindepth"] if "mindepth" in config else 0
    zmax = config["maxdepth"] if "maxdepth" in config else 30
    config["x(km)"] = (xmin, xmax)
    config["y(km)"] = (ymin, ymax)
    config["z(km)"] = (zmin, zmax)
    config["bfgs_bounds"] = (
        (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
        (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
        (0, config["z(km)"][1] + 1),  # z
        (None, None),  # t
    )

    # DBSCAN
    config["dbscan_eps"] = estimate_eps(stations, config["vel"]["p"])  # s
    config["dbscan_min_samples"] = 3

    ## Eikonal for 1D velocity model
    # zz = [0.0, 5.5, 16.0, 32.0]
    # vp = [5.5, 5.5,  6.7,  7.8]
    # vp_vs_ratio = 1.73
    # vs = [v / vp_vs_ratio for v in vp]
    # h = 0.3
    # # h = 3
    # vel = {"z": zz, "p": vp, "s": vs}
    # config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}

    # filtering
    config["min_picks_per_eq"] = 5
    config["min_p_picks_per_eq"] = 0
    config["min_s_picks_per_eq"] = 0
    config["max_sigma11"] = 2.0  # s
    config["max_sigma22"] = 1.0  # log10(m/s)
    config["max_sigma12"] = 1.0  # covariance

    ## filter picks without amplitude measurements
    if config["use_amplitude"]:
        picks = picks[picks["amp"] != -1]

    for k, v in config.items():
        print(f"{k}: {v}")

    # %%
    event_idx0 = 0  ## current earthquake index
    assignments = []
    events, assignments = association(picks, stations, config, event_idx0, config["method"])
    event_idx0 += len(events)

    if len(events) > 0:
        ## create catalog
        events = pd.DataFrame(events)
        events[["longitude", "latitude"]] = events.apply(
            lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1
        )
        events["depth_km"] = events["z(km)"]
        events.sort_values("time", inplace=True)
        with open(f"{root_path}/{gamma_events_csv}", "w") as fp:
            events.to_csv(fp, index=False, float_format="%.3f", date_format="%Y-%m-%dT%H:%M:%S.%f")
        ## add assignment to picks
        assignments = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
        picks = picks.join(assignments.set_index("pick_index")).fillna(-1).astype({"event_index": int})
        picks.sort_values(["phase_time"], inplace=True)
        with open(f"{root_path}/{gamma_picks_csv}", "w") as fp:
            picks.to_csv(fp, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")

        if protocol != "file":
            fs.put(f"{root_path}/{gamma_events_csv}", f"{bucket}/{gamma_events_csv}")
            fs.put(f"{root_path}/{gamma_picks_csv}", f"{bucket}/{gamma_picks_csv}")

    else:
        print(f"No events associated in {picks_csv}")
        with open(f"{root_path}/{gamma_events_csv}", "w") as fp:
            pass
        with open(f"{root_path}/{gamma_picks_csv}", "w") as fp:
            pass

    # %% copy to results/phase_association
    if not os.path.exists(f"{root_path}/{region}/results/phase_association"):
        os.makedirs(f"{root_path}/{region}/results/phase_association")
    os.system(
        f"cp {root_path}/{gamma_events_csv} {root_path}/{region}/results/phase_association/events_{node_rank:03d}_{num_nodes:03d}.csv"
    )
    os.system(
        f"cp {root_path}/{gamma_picks_csv} {root_path}/{region}/results/phase_association/picks_{node_rank:03d}_{num_nodes:03d}.csv"
    )
    if protocol != "file":
        fs.put(
            f"{root_path}/{gamma_events_csv}",
            f"{bucket}/{region}/results/phase_association/events_{node_rank:03d}_{num_nodes:03d}.csv",
        )
        fs.put(
            f"{root_path}/{gamma_picks_csv}",
            f"{bucket}/{region}/results/phase_association/picks_{node_rank:03d}_{num_nodes:03d}.csv",
        )

    outputs = NamedTuple("outputs", events=str, picks=str)
    return outputs(events=gamma_events_csv, picks=gamma_picks_csv)


if __name__ == "__main__":

    os.environ["OMP_NUM_THREADS"] = "8"

    args = parse_args()
    root_path = args.root_path
    region = args.region

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    run_gamma(root_path=root_path, region=region, config=config)

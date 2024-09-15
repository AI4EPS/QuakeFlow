import json
import multiprocessing as mp
import os
import sys
from typing import Dict, List, NamedTuple

import fsspec
import numpy as np
import pandas as pd
from gamma.utils import association, estimate_eps
from pyproj import Proj


def run_gamma(
    root_path: str,
    region: str,
    config: Dict,
    jdays: list,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
):

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    for jday in jdays:
        print(f"Processing {jday}")

        year = int(jday.split(".")[0])
        jday = int(jday.split(".")[1])

        # %%
        result_path = f"{region}/gamma/{year:04d}"
        if not os.path.exists(f"{root_path}/{result_path}"):
            os.makedirs(f"{root_path}/{result_path}")

        # %%
        # station_csv = data_path / "stations.csv"
        station_json = f"{region}/results/data/stations.json"
        # if picks_csv is None:
        #     picks_csv = f"{region}/results/phase_picking/phase_picks_{rank:03d}.csv"
        picks_csv = f"{region}/phasenet/picks/{year:04d}/{jday:03d}.csv"
        gamma_events_csv = f"{result_path}/gamma_events_{jday:03d}.csv"
        gamma_picks_csv = f"{result_path}/gamma_picks_{jday:03d}.csv"

        if os.path.exists(f"{root_path}/{gamma_events_csv}") and os.path.exists(f"{root_path}/{gamma_picks_csv}"):
            print(f"Skipping {jday}")
            continue

        # %%
        ## read picks
        if protocol == "file":
            picks = pd.read_csv(f"{root_path}/{picks_csv}")
        else:
            picks = pd.read_csv(f"{protocol}://{bucket}/{picks_csv}")

        picks.rename(
            columns={
                "station_id": "id",
                "phase_time": "timestamp",
                "phase_type": "type",
                "phase_score": "prob",
                "phase_amplitude": "amp",
            },
            inplace=True,
        )
        # FIXME: hard-coded converting nano m/s
        picks["amp"] = picks["amp"] / 1e9

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
        # config["x(km)"] = (
        #     np.array([config["minlongitude"] - config["longitude0"], config["maxlongitude"] - config["longitude0"]])
        #     * config["degree2km"]
        #     * np.cos(np.deg2rad(config["latitude0"]))
        # )
        # config["y(km)"] = (
        #     np.array([config["minlatitude"] - config["latitude0"], config["maxlatitude"] - config["latitude0"]])
        #     * config["degree2km"]
        # )
        xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
        xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
        config["x(km)"] = (xmin, xmax)
        config["y(km)"] = (ymin, ymax)
        if "gamma" not in config:
            config["z(km)"] = (0, 60)
        else:
            config["z(km)"] = [config["gamma"]["zmin_km"], config["gamma"]["zmax_km"]]
        config["bfgs_bounds"] = (
            (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
            (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
            (0, config["z(km)"][1] + 1),  # z
            (None, None),  # t
        )

        # DBSCAN
        # config["dbscan_eps"] = estimate_eps(stations, config["vel"]["p"])  # s
        config["dbscan_eps"] = 10
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
        config["max_sigma11"] = 3.0  # s
        config["max_sigma22"] = 2.0  # log10(m/s)
        config["max_sigma12"] = 2.0  # covariance

        ## filter picks without amplitude measurements
        if config["use_amplitude"]:
            picks = picks[picks["amp"] != -1]

        # ## %%
        config["ncpu"] = 32

        for k, v in config.items():
            print(f"{k}: {v}")

        # %%
        event_idx0 = 0  ## current earthquake index
        assignments = []
        events, assignments = association(picks, stations, config, event_idx0, config["method"])
        event_idx0 += len(events)

        if len(events) > 0:
            ## create catalog
            # events = pd.DataFrame(
            #     events,
            #     columns=["time"]
            #     + config["dims"]
            #     + ["magnitude", "sigma_time", "sigma_amp", "cov_time_amp", "event_index", "gamma_score"],
            # )
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
            picks.rename(
                columns={
                    "id": "station_id",
                    "timestamp": "phase_time",
                    "type": "phase_type",
                    "prob": "phase_score",
                    "amp": "phase_amplitude",
                },
                inplace=True,
            )
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

        # # %% copy to results/phase_association
        # if not os.path.exists(f"{root_path}/{region}/results/phase_association"):
        #     os.makedirs(f"{root_path}/{region}/results/phase_association")
        # os.system(
        #     f"cp {root_path}/{gamma_events_csv} {root_path}/{region}/results/phase_association/events_{rank:03d}.csv"
        # )
        # os.system(
        #     f"cp {root_path}/{gamma_picks_csv} {root_path}/{region}/results/phase_association/picks_{rank:03d}.csv"
        # )
        # if protocol != "file":
        #     fs.put(
        #         f"{root_path}/{gamma_events_csv}",
        #         f"{bucket}/{region}/results/phase_association/events_{rank:03d}.csv",
        #     )
        #     fs.put(
        #         f"{root_path}/{gamma_picks_csv}",
        #         f"{bucket}/{region}/results/phase_association/picks_{rank:03d}.csv",
        #     )

        # outputs = NamedTuple("outputs", events=str, picks=str)
        # return outputs(events=gamma_events_csv, picks=gamma_picks_csv)


if __name__ == "__main__":

    os.environ["OMP_NUM_THREADS"] = "8"

    root_path = "local"
    region = "hinet"
    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    # run_gamma.execute(root_path=root_path, region=region, config=config)
    years = os.listdir(f"{root_path}/{region}/phasenet/picks")
    jdays = []
    for year in years:
        jdays += [f"{year}.{jday}" for jday in os.listdir(f"{root_path}/{region}/phasenet/picks/{year}")]
    jdays = sorted(jdays, reverse=True)
    run_gamma(root_path=root_path, region=region, config=config, jdays=jdays)

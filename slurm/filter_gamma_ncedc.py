from typing import Dict, List, NamedTuple

from kfp import dsl


@dsl.component(base_image="zhuwq0/quakeflow:latest")
def filt_gamma(
    root_path: str,
    region: str,
    config: Dict,
    year: int = 2023,
    jday: int = 0,
    picks_csv: str = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> NamedTuple("outputs", events=str, picks=str):
    import os

    import fsspec
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    from tqdm import tqdm

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    result_path = f"{region}/gamma/{year:04d}/"

    # %%
    station_json = f"{region}/results/network/stations.json"
    gamma_events_csv = f"{result_path}/gamma_events_{jday:03d}.csv"
    gamma_picks_csv = f"{result_path}/gamma_picks_{jday:03d}.csv"

    gamma_events_filt_csv = f"{result_path}/gamma_events_filt_{jday:03d}.csv"
    gamma_picks_filt_csv = f"{result_path}/gamma_picks_filt_{jday:03d}.csv"

    # %%
    if not os.path.exists(f"{root_path}/{gamma_events_csv}"):
        return NamedTuple("outputs", events=str, picks=str)(events=gamma_events_filt_csv, picks=gamma_picks_filt_csv)

    # %%
    events = pd.read_csv(f"{root_path}/{gamma_events_csv}", parse_dates=["time"])
    picks = pd.read_csv(f"{root_path}/{gamma_picks_csv}", parse_dates=["phase_time"])
    stations = pd.read_json(f"{root_path}/{station_json}", orient="index")
    stations["station_id"] = stations.index

    # $$
    MIN_NEAREST_STATION_RATIO = 0.2
    MIN_PICKS = 5
    MIN_P_PICKS = 3
    MIN_S_PICKS = 3

    # %%
    stations = stations[stations["station_id"].isin(picks["station_id"].unique())]

    neigh = NearestNeighbors(n_neighbors=min(len(stations), 10))
    neigh.fit(stations[["longitude", "latitude"]].values)

    # %%
    picks = picks.merge(events[["event_index", "longitude", "latitude"]], on="event_index", suffixes=("", "_event"))
    picks = picks.merge(stations[["station_id", "longitude", "latitude"]], on="station_id", suffixes=("", "_station"))

    # %%
    filtered_events = []
    for i, event in tqdm(events.iterrows(), total=len(events)):
        # sid = neigh.kneighbors([[event["longitude"], event["latitude"]]])[1][0]
        picks_ = picks[picks["event_index"] == event["event_index"]]

        if len(picks_) < MIN_PICKS:
            continue
        if len(picks_[picks_["phase_type"] == "P"]) < MIN_P_PICKS:
            continue
        if len(picks_[picks_["phase_type"] == "S"]) < MIN_S_PICKS:
            continue

        longitude, latitude = picks_[["longitude", "latitude"]].mean().values
        sid = neigh.kneighbors([[longitude, latitude]])[1][0]
        stations_neigh = stations.iloc[sid]["station_id"].values
        picks_neigh = picks_[picks_["station_id"].isin(stations_neigh)]
        stations_with_picks = picks_neigh["station_id"].unique()
        if len(stations_with_picks) / len(stations_neigh) > MIN_NEAREST_STATION_RATIO:
            filtered_events.append(event)

    # %%
    print(f"Events before filtering: {len(events)}")
    print(f"Events after filtering: {len(filtered_events)}")
    filtered_events = pd.DataFrame(filtered_events)
    filtered_events.to_csv(f"{root_path}/{gamma_events_filt_csv}", index=False)

    filtered_picks = picks[picks["event_index"].isin(filtered_events["event_index"])]
    filtered_picks.to_csv(f"{root_path}/{gamma_picks_filt_csv}", index=False)

    # %% copy to results/phase_association
    os.system(f"cp {root_path}/{gamma_events_filt_csv} {root_path}/{region}/results/phase_association/events_{jday:03d}.csv")
    os.system(f"cp {root_path}/{gamma_picks_filt_csv} {root_path}/{region}/results/phase_association/picks_{jday:03d}.csv")
    if protocol != "file":
        fs.put(
            f"{root_path}/{gamma_events_filt_csv}",
            f"{bucket}/{region}/results/phase_association/events_{jday:03d}.csv",
        )
        fs.put(
            f"{root_path}/{gamma_picks_filt_csv}",
            f"{bucket}/{region}/results/phase_association/picks_{jday:03d}.csv",
        )

    outputs = NamedTuple("outputs", events=str, picks=str)
    return outputs(events=gamma_events_filt_csv, picks=gamma_picks_filt_csv)

if __name__ == "__main__":
    import json
    import os
    import sys

    root_path = "local"
    region = "ncedc"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]
    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    year = 2023
    # jday = 1
    for jday in range(1, 366):
        filt_gamma.execute(root_path=root_path, region=region, config=config, year=year, jday=jday)
# %%
from pathlib import Path

import numpy as np
import pandas as pd
from eikonal2d import calc_ray_param, init_eikonal2d
from polarity import calc_radiation_pattern
from tqdm import tqdm

# %%
if __name__ == "__main__":
    
    # %%
    data_path = Path("../NCEDC/dataset")
    result_path = Path("./results")
    result_path.mkdir(parents=True, exist_ok=True)

    year = 2023
    jday = 1
    
    events_path = data_path / f"{year:04d}" / f"{jday:03d}" / "events.csv"
    phases_path = data_path / f"{year:04d}" / f"{jday:03d}" / "phases.csv"
    mechanisms_path = data_path / f"{year:04d}" / f"{jday:03d}" / "focal_mechanisms.csv"
    stations = data_path / "stations.csv"
    
    # %% Load data
    events = pd.read_csv(events_path, dtype=str)
    phases = pd.read_csv(phases_path, dtype=str)
    mechanisms = pd.read_csv(mechanisms_path, dtype=str)
    stations = pd.read_csv(stations, dtype=str)
    events = events.fillna("")
    phases = phases.fillna("")
    mechanisms = mechanisms.fillna("")
    stations = stations.fillna("")
    events["latitude"] = events["latitude"].astype("float32")
    events["longitude"] = events["longitude"].astype("float32")
    events["depth_km"] = events["depth_km"].astype("float32")
    mechanisms["strike"] = mechanisms["strike"].astype("float32")
    mechanisms["dip"] = mechanisms["dip"].astype("float32")
    mechanisms["rake"] = mechanisms["rake"].astype("float32")
    stations["latitude"] = stations["latitude"].astype("float32")
    stations["longitude"] = stations["longitude"].astype("float32")
    stations["depth_km"] = stations["depth_km"].astype("float32")

    # %%
    event_ids = set(mechanisms["event_id"])
    phases = phases[phases["event_id"].isin(event_ids)]
    events = events[events["event_id"].isin(event_ids)]
    phases = phases[["event_id", "network", "station", "location", "instrument", "phase_type"]]
    events = events[["event_id", "latitude", "longitude", "depth_km"]]
    mechanisms = mechanisms[["event_id", "strike", "dip", "rake"]]
    if "polarity_misfit" in mechanisms.columns:
        mechanisms["first_motion_misfit"] = mechanisms["first_motion_misfit"].astype
    mechanisms = mechanisms.drop_duplicates(subset="event_id", keep="first")
    mechanisms = mechanisms.set_index("event_id")
    stations = stations[["network", "station", "location", "instrument", "latitude", "longitude", "depth_km"]]
    stations.drop_duplicates(inplace=True)
    phases = phases.merge(events[['event_id', 'latitude', 'longitude', 'depth_km']], on='event_id', how='inner')
    phases.rename(columns={
        "latitude": "event_latitude",
        "longitude": "event_longitude",
        "depth_km": "event_depth_km"
    }, inplace=True)
    phases = phases.merge(stations, on=['network', 'station', 'location', 'instrument'], how='inner')
    phases.rename(columns={
        "latitude": "station_latitude",
        "longitude": "station_longitude",
        "depth_km": "station_depth_km"
    }, inplace=True)


    # %%
    ray_param = calc_ray_param(
        phases["event_longitude"].values,
        phases["event_latitude"].values,
        phases["event_depth_km"].values,
        phases["station_longitude"].values,
        phases["station_latitude"].values,
        phases["station_depth_km"].values,
        phases["phase_type"].values,
        None
    )
    phases["azimuth"] = ray_param["azimuth"]
    phases["takeoff0"] = ray_param["takeoff"]
    phases["distance_km"] = ray_param["distance_km"]

    # %% Eikonal Config
    # Velocity model 
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 6.0, 6.7, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    
    # Eikonal config
    R_max = phases["distance_km"].max() + 10
    Z_max = phases["event_depth_km"].max() - phases["station_depth_km"].min() + 10
    h = 1.0
    
    vel = {"Z": zz, "P": vp, "S": vs}
    config = {
        "vel": vel,
        "h": h,
        "xlim_km": [0, R_max],
        "ylim_km": [0, R_max], 
        "zlim_km": [0, Z_max]
    }
    

    config["eikonal"] = init_eikonal2d(config)

    # %%
    ray_param = calc_ray_param(
            phases["event_longitude"].values,
            phases["event_latitude"].values,
            phases["event_depth_km"].values,
            phases["station_longitude"].values,
            phases["station_latitude"].values,
            phases["station_depth_km"].values,
            phases["phase_type"].values,
            None
        )
    phases["takeoff"] = ray_param["takeoff"]

    # selected_event = ["nc73827381"]
    # selected_stations = ['CE.57383.10.HN', 'NC.HGS.03.EH', 'NC.HGS.03.HN', 'NC.HLP..EH', 'NC.HLP..HN', 'NC.HMP.01.EH', 'NC.HMP.01.HN', 'NC.HMS.01.EH', 'NC.HMS.01.HN']
    # phases = phases[(phases["event_id"].isin(selected_event)) & (phases["network"] + "." + phases["station"] + "." + phases["location"] + "." + phases["instrument"]).isin(selected_stations)]

    # %% 
    # need to loop event by event to update takeoff angle
    polarities = []
    for event_id, group in tqdm(phases.groupby("event_id")):
        mech = mechanisms.loc[event_id]
        strike = mech["strike"]
        dip = mech["dip"]
        rake = mech["rake"]
        takeoff = group["takeoff"].values
        azimuth = group["azimuth"].values
        radiation = calc_radiation_pattern(
            strike, dip, rake, takeoff, azimuth
        )
        sub_df = group[["event_id","network", "station", "location"]].copy()
        P_ENZ = radiation["P_ENZ"]
        S_ENZ = radiation["S_ENZ"]
        sub_df["p_polarity_e"] = np.round(P_ENZ[:,0], 6)
        sub_df["p_polarity_n"] = np.round(P_ENZ[:,1], 6)
        sub_df["p_polarity_z"] = np.round(P_ENZ[:,2], 6)
        sub_df["s_polarity_e"] = np.round(S_ENZ[:,0], 6)
        sub_df["s_polarity_n"] = np.round(S_ENZ[:,1], 6)
        sub_df["s_polarity_z"] = np.round(S_ENZ[:,2], 6)
        sub_df["log_sp_ratio"] = np.round(radiation["log_SP"], 3)
        polarities.append(sub_df)

    polarity_df = pd.concat(polarities, ignore_index=True)
    polarity_df.to_csv(data_path/f"{year:04d}/{jday:03d}/polarities.csv", index=False)

# %%

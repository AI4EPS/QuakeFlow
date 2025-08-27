# %%
import json
import os
from glob import glob
from typing import Dict

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from args import parse_args
from pyproj import Proj
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def cartesian_distance(lon1, lat1, lon2, lat2):
    degree2km = 111.32
    x1 = lon1 * degree2km * np.cos(lat1 * np.pi / 180)
    y1 = lat1 * degree2km
    x2 = lon2 * degree2km * np.cos(lat2 * np.pi / 180)
    y2 = lat2 * degree2km
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def link_unassociated_picks(event_picks, events, DT=2, VP=6, VPVS_RATIO=1.73, discarded_indexes=None, num_workers=1):
    """
    Update event_picks rows (where event_index == -1) by matching candidate events from `events`
    in a ±DT window around the pick's event_time. For candidate events, compute the distance
    using the haversine formula and compare expected travel time (distance / VP) with the pick's
    travel_time (within ±DT). The best candidate (smallest sum time difference) is used to update event_index.

    This version uses pandarallel to parallelize the row-wise processing.
    
    Parameters:
      event_picks : pandas.DataFrame
        Must contain 'event_index', 'event_time', 'travel_time', 'latitude', and 'longitude' columns.
      events : pandas.DataFrame
        Must contain 'event_index', 'time', 'latitude', and 'longitude' columns.
      DT : float
        Time window in seconds for candidate events (±DT around event_time).
      VP : float
        Seismic velocity (km/s) for travel time computation.
        
    Returns:
      Updated event_picks DataFrame with new event_index values.
    """
    # Work on copies
    events = events.copy()
    event_picks = event_picks.copy()
    
    # Sort events by time for fast search
    events_sorted = events.sort_values('timestamp').reset_index(drop=True)
    events_time = events_sorted['timestamp'].values
    if discarded_indexes is not None:
        discarded_indexes = discarded_indexes.union(set([-1]))
    else:
        discarded_indexes = set([-1])

    def process_pick(row):
        # Only process rows where event_index is -1
        if row['event_index'] not in discarded_indexes:
            return row['event_index']
            
        t_pick = row['timestamp']
        # Find candidate events within ±DT seconds using np.searchsorted
        left = np.searchsorted(events_time, t_pick - DT, side='left')
        right = np.searchsorted(events_time, t_pick + DT, side='right')
        if left == right:
            return -1  # No candidates found
        
        candidates = events_sorted.iloc[left:right]
        candidate_times = events_time[left:right]
        time_diffs = np.abs(candidate_times - t_pick)
        
        # Station location for the current pick
        station_lat = row['latitude']
        station_lon = row['longitude']
        candidate_lat = candidates['latitude'].values
        candidate_lon = candidates['longitude'].values
        
        # Compute distances and expected travel times
        distances = cartesian_distance(station_lon, station_lat, candidate_lon, candidate_lat)
        expected_travel_time = distances / VP
        
        travel_time = row['travel_time'] # Travel time is actually travel_time_p + (s-p) / 2
        # Filter candidates based on travel time closeness (within ±DT seconds)
        travel_time_diffs = np.abs(expected_travel_time - travel_time*2/(VPVS_RATIO + 1))
        valid_mask = (travel_time_diffs <= DT)
        if not valid_mask.any():
            return -1
        
        valid_time_diffs = time_diffs[valid_mask] + travel_time_diffs[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        best_candidate_idx = valid_indices[np.argmin(valid_time_diffs)]
        best_event_index = candidates.iloc[best_candidate_idx]['event_index']
        return best_event_index

    # Use parallel_apply from pandarallel to process each row in parallel
    if num_workers > 1:
        event_picks['event_index'] = event_picks.parallel_apply(process_pick, axis=1)
    else:
        # Fallback to regular apply if pandarallel is not available
        event_picks['event_index'] = event_picks.apply(process_pick, axis=1)
    return event_picks

def calc_association_ratio(events, stations, weight_ratio=1.0, num_workers=1):
    """
    Update each row in the events DataFrame as follows:
      - For each event, compute the weighted ratio between its related stations and
        all stations within the coverage distance.
      - If the ratio is lower than the threshold, set event_index to -1.
    The computation is parallelized using pandarallel.
    """
    events = events.copy()
    stations = stations.copy()
    def process_event(row):
        """
        For a given event row, do the following:
          1. From the event's related stations (in row['station_id'] which is assumed to be a list),
             compute distances from the event location (row['latitude'], row['longitude']).
          2. Determine d_max (the largest distance among related stations).
          3. Compute distances from the event to all stations (vectorized haversine).
          4. Identify stations within d_max (“covered stations”).
          5. Compute weights using linear scaling:
                weight = 1 + (d_max - d) / (d_max - 1)
             so that a station at d_max gets weight 1 and a station at 1 km gets weight 2.
          6. Compute the weighted ratio:
                 ratio = (sum of weights for stations that are related) / (sum of weights for all covered stations)
          7. If the ratio is lower than the given threshold, return -1 (to mark event_index as -1);
             otherwise, return the original event_index.
        """
        event_lat = row['latitude']
        event_lon = row['longitude']
        related_ids = row['station_id'].split(',')

        # If there are no related stations, mark this event as invalid.
        if len(related_ids) < 3:
            return -1

        stations['distance'] = stations.apply(lambda x: cartesian_distance(event_lon, event_lat, x['longitude'], x['latitude']), axis=1)
        related_distances = stations[stations['station_id'].isin(related_ids)]['distance'].values
        if len(related_distances) == 0:
            return -1  # None of the related stations are in our stations data.
        # The maximum distance among related stations defines the coverage radius.
        d_max = 50#max(related_distances)
        # For the weighting formula to work, assume d_max >= 1. Otherwise, set d_max = 1.
        if d_max < 1:
            d_max = 1.1

        covered_distances = stations[stations['distance'] <= related_distances.max()]['distance'].values

        if weight_ratio != 0:
            # Compute weights for the covered stations.
            # Linear scaling: weight = 1 + (d_max - d) / (d_max - 1)
            weights = 1 + np.clip(weight_ratio*(d_max - covered_distances) / (d_max - 1), a_min=-0.8, a_max=1)
            total_weight = weights.sum()
            related_weights = 1 + np.clip(weight_ratio*(d_max - related_distances) / (d_max - 1), a_min=-0.8, a_max=1)
            related_weight = related_weights.sum()
            ratio = related_weight / total_weight if total_weight > 0 else 0
        else:
            # If weight_ratio is 0, just count the number of related stations.
            ratio = len(related_ids) / len(covered_distances) if len(covered_distances) > 0 else 0

        return ratio

    if num_workers > 1:
        events['association_ratio'] = events.parallel_apply(process_event, axis=1)
    else:
        events['association_ratio'] = events.apply(process_event, axis=1)
    return events

def link_picks_to_events(phase_picks, event_picks, VPVS_RATIO=1.73):
    """
    group: A DataFrame of `events` for a single station_id. 
           The group is identified by group.name (the station_id).
           
    This function:
      1. Sorts the events by 'num_picks'
      2. Computes ps_delta and the time windows t1, t2
      3. Retrieves picks for this station from the global `picks` DataFrame
      4. Updates event_index, sp_ratio, event_amplitude for the picks that fall in the events' time windows
      5. Returns a small updated DataFrame of picks for this station
    """
    
    def process_group(group):
        event = group.copy()
        event = event[event['event_index'] != -1]  # Filter out events with event_index == -1
        if len(event) == 0:
            return pd.DataFrame(columns=phase_picks.columns)
        
        station_id = event['station_id'].iloc[0]  # The station_id for this group
        # travel time tt = (tp + ts) / 2 = (1 + ps_ratio)/2 * tp => tp = tt * 2 / (1 + ps_ratio)
        # (ts - tp) = (ps_ratio - 1) tp = tt * 2 * (ps_ratio - 1) / (ps_ratio + 1)
        #event = event.sort_values(by='num_picks', ascending=True)
        # since the argmax returns the first of the max, the largest num_picks should be the first
        event = event.sort_values(by='num_picks', ascending=False) 
        #ps_delta = event["travel_time"].values * 2 * (VPVS_RATIO - 1) / (VPVS_RATIO + 1)
        ps_delta = np.max(np.stack([event["ps_delta"].values, event["travel_time"].values * 2 * (VPVS_RATIO - 1) / (VPVS_RATIO + 1)], axis=0), axis=0)
        t1 = event["timestamp_center"].values - (ps_delta*0.6 + 1.0) # the width is half ps_delta with a extension of 0.5s
        t2 = event["timestamp_center"].values + (ps_delta*0.6 + 1.0)
        t_center = event["timestamp_center"].values
        # Retrieve picks for this station
        try:
            picks_for_station = phase_picks.loc[station_id].copy()  # or picks.loc[station_id].copy()
            #picks.loc[station_id].copy()
        except KeyError:
            # If no picks for this station, just return empty or do nothing
            return pd.DataFrame(columns=phase_picks.columns)

        # We have shape (n_events, ) and picks_for_station has shape (n_picks_for_station, ...).
        # Let's match them up by broadcast.
        picks_ = picks_for_station['timestamp'].values  # shape (Npk, )
        phase_types = picks_for_station['phase_type'].values  # shape (Npk, )
        #mask = (picks_[None, :] >= t1[:, None]) & (picks_[None, :] <= t2[:, None])  # (Nev, Npk)
        mask = ((picks_[None, :] >= t1[:, None]) & (picks_[None, :] <= t_center[:, None]) & (phase_types[None, :] == 'P')) | \
               ((picks_[None, :] >= t_center[:, None]) & (picks_[None, :] <= t2[:, None]) & (phase_types[None, :] == 'S'))  # (Nev, Npk)
        # picks.loc[group_id, "event_index"] = np.where(
        #     mask.any(axis=0), index.values[mask.argmax(axis=0)], picks.loc[group_id, "event_index"]
        # )
        mask_true = mask.any(axis=0)
        mask_idx = mask.argmax(axis=0)
        picks_for_station["event_index"] = np.where(mask_true, event["event_index"].values[mask_idx], -1)
        picks_for_station["sp_ratio"] = np.where(mask_true, event["sp_ratio"].values[mask_idx], np.nan)
        picks_for_station["event_amplitude"] = np.where(mask_true, event["event_amplitude"].values[mask_idx], np.nan)

        # Return the updated picks for this station
        return picks_for_station
    
    updated_picks_list = event_picks.groupby("station_id").parallel_apply(process_group)
    updated_picks_list.index = updated_picks_list.index.droplevel(0)
    updated_picks_list = updated_picks_list.reset_index()
    
    return updated_picks_list


def plotting_debug(xt, hist, topk_index, topk_score, picks, events, stations, config):

    # timestamp0 = config["timestamp0"]
    # events_compare = pd.read_csv("local/Ridgecrest_debug5/adloc_gamma/ransac_events.csv")
    # picks_compare = pd.read_csv("local/Ridgecrest_debug5/adloc_gamma/ransac_picks.csv")
    # # events_compare = pd.read_csv("local/Ridgecrest_debug5/adloc_plus2/ransac_events_sst_0.csv")
    # # picks_compare = pd.read_csv("local/Ridgecrest_debug5/adloc_plus2/ransac_picks_sst_0.csv")
    # events_compare["time"] = pd.to_datetime(events_compare["time"])
    # events_compare["timestamp"] = events_compare["time"].apply(lambda x: (x - timestamp0).total_seconds())
    # picks_compare["phase_time"] = pd.to_datetime(picks_compare["phase_time"])
    # picks_compare["timestamp"] = picks_compare["phase_time"].apply(lambda x: (x - timestamp0).total_seconds())

    DT = config["DT"]
    MIN_STATION = config["MIN_STATION"]

    # map station_id to int
    stations["xy"] = stations["longitude"] - stations["latitude"]
    stations.sort_values(by="xy", inplace=True)
    mapping_id = {v: i for i, v in enumerate(stations["station_id"])}
    mapping_color = {v: f"C{i}" if v != -1 else "k" for i, v in enumerate(events["event_index"].unique())}

    NX = 100
    for i in tqdm(range(0, len(hist), NX)):
        bins = np.arange(i, i + NX, DT)

        fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # plot hist
        idx = (xt > i) & (xt < i + NX)
        ax[0].bar(xt[idx], hist[idx], width=DT)

        ylim = ax[0].get_ylim()
        idx = (xt[topk_index] > i) & (xt[topk_index] < i + NX)
        ax[0].vlines(xt[topk_index][idx], ylim[0], ylim[1], color="k", linewidth=1)

        # idx = (events_compare["timestamp"] > i) & (events_compare["timestamp"] < i + NX)
        # ax[0].vlines(events_compare["timestamp"][idx], ylim[0], ylim[1], color="r", linewidth=1, linestyle="--")

        # plot picks-events match
        idx = (events["timestamp"] > i) & (events["timestamp"] < i + NX)
        ax[1].scatter(
            events["timestamp"][idx],
            events["station_id"][idx].map(mapping_id),
            c=events["event_index"][idx].map(mapping_color),
            marker=".",
            s=30,
        )

        idx = (picks["timestamp"] > i) & (picks["timestamp"] < i + NX)
        ax[1].scatter(
            picks["timestamp"][idx],
            picks["station_id"][idx].map(mapping_id),
            c=picks["event_index"][idx].map(mapping_color),
            marker="x",
            linewidth=0.5,
            s=10,
        )

        # idx = (picks_compare["timestamp"] > i) & (picks_compare["timestamp"] < i + NX)
        # ax[1].scatter(
        #     picks_compare["timestamp"][idx],
        #     picks_compare["station_id"][idx].map(mapping_id),
        #     facecolors="none",
        #     edgecolors="r",
        #     linewidths=0.1,
        #     s=30,
        # )

        if not os.path.exists(f"figures"):
            os.makedirs(f"figures")
        plt.savefig(f"figures/debug_{i:04d}.png", dpi=300, bbox_inches="tight")


def associate(
    picks: pd.DataFrame,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    config: Dict,
    association_ratio_threshold: float = 0.08,
    num_workers: int = 1,
    gpu: bool = False,
):

    VPVS_RATIO = config["VPVS_RATIO"]
    VP = config["VP"]
    DT = 2.0  # seconds
    MIN_STATION = 3

    # %%
    timestamp0 = min(events["event_time"].min(), picks["phase_time"].min())

    events["timestamp"] = events["event_time"].apply(lambda x: (x - timestamp0).total_seconds())
    events["timestamp_center"] = events["center_time"].apply(lambda x: (x - timestamp0).total_seconds())
    picks["timestamp"] = picks["phase_time"].apply(lambda x: (x - timestamp0).total_seconds())

    t0 = min(events["timestamp"].min(), picks["timestamp"].min())
    t1 = max(events["timestamp"].max(), picks["timestamp"].max())

    # %% Using DBSCAN to cluster events
    # proj = Proj(proj="merc", datum="WGS84", units="km")
    # stations[["x_km", "y_km"]] = stations.apply(lambda x: pd.Series(proj(x.longitude, x.latitude)), axis=1)
    # events = events.merge(stations[["station_id", "x_km", "y_km"]], on="station_id", how="left")
    # scaling = np.array([1.0, 1.0 / eps_xy, 1.0 / eps_xy])
    # clustering = DBSCAN(eps=2.0, min_samples=4).fit(events[["timestamp", "x_km", "y_km"]] * scaling)
    # # clustering = DBSCAN(eps=2.0, min_samples=4).fit(events[["timestamp"]])
    # events["event_index"] = clustering.labels_

    ## Using histogram to cluster events
    events["event_index"] = -1
    t = np.arange(t0, t1, DT)
    hist, edge = np.histogram(events["timestamp"], bins=t, weights=events["event_score"])
    xt = (edge[:-1] + edge[1:]) / 2  # center of the bin
    # hist_numpy = hist.copy()

    hist = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available():
            hist = hist.cuda()
            hist.requires_grad = False
    hist_pool = F.max_pool1d(hist, kernel_size=3, padding=1, stride=1)
    mask = hist_pool == hist
    hist = hist * mask
    hist = hist.squeeze(0).squeeze(0).detach().cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    K = int((t[-1] - t[0]) / 5)  # assume max 1 event per 10 seconds on average
    topk_score, topk_index = torch.topk(hist, k=K)
    topk_index = topk_index[topk_score >= MIN_STATION]  # min 3 stations
    topk_score = topk_score[topk_score >= MIN_STATION]
    topk_index = topk_index.numpy()
    topk_score = topk_score.numpy()
    num_events = len(topk_index)
    t00 = xt[topk_index - 1]
    t11 = xt[topk_index + 1]
    timestamp = events["timestamp"].values
    for i in tqdm(range(num_events), desc="Assigning event index"):
        mask = (timestamp >= t00[i]) & (timestamp <= t11[i])
        events.loc[mask, "event_index"] = i
    #events["num_picks"] = events.groupby("event_index").size()
    ## logPGV = -4.75 + 1.68 * logR + 0.93M => M = (logPGV - 4.175 - 1.68 * logR) / 0.93
    events["magnitude"] = (
        np.log10(events["event_amplitude"])
        + 4.175
        + 1.68 * np.log10(events["travel_time"] * VP * (VPVS_RATIO + 1.0) / 2.0)
    ) / 0.93

    # # refine event index using DBSCAN
    # events["group_index"] = -1
    # for group_id, event in tqdm(events.groupby("event_index"), desc="DBSCAN clustering"):
    #     if len(event) < MIN_STATION:
    #         events.loc[event.index, "event_index"] = -1
    #     clustering = DBSCAN(eps=20, min_samples=MIN_STATION).fit(event[["x_km", "y_km"]])
    #     events.loc[event.index, "group_index"] = clustering.labels_
    # events["dummy_index"] = events["event_index"].astype(str) + "." + events["group_index"].astype(str)
    # mapping = {v: i for i, v in enumerate(events["dummy_index"].unique())}
    # events["dummy_index"] = events["dummy_index"].map(mapping)
    # events.loc[(events["event_index"] == -1) | (events["group_index"] == -1), "dummy_index"] = -1
    # events["event_index"] = events["dummy_index"]
    # events.drop(columns=["dummy_index"], inplace=True)

    #picks.drop(columns=["timestamp"], inplace=True)
    #events.drop(columns=["timestamp", "timestamp_center"], inplace=True)

    events = events.merge(stations[["station_id", "latitude", "longitude"]], on="station_id", how="left")
    events_catalog = (
        events.groupby("event_index")
        .agg(
            {
                "event_time": "median",
                "event_score": "sum",
                "latitude": "median",
                "longitude": "median",
                "magnitude": "median",
                "station_id": lambda x: ','.join(x)
            }
        )
        .reset_index()
    )
    events_catalog.rename(columns={"event_time": "time"}, inplace=True)
    events_catalog["timestamp"] = events_catalog["time"].apply(lambda x: (x - timestamp0).total_seconds())
    # drop event index -1
    events_catalog = events_catalog[events_catalog['event_index'] != -1]
    events_catalog = events_catalog[events_catalog['event_score'] >= MIN_STATION//2]
    
    print(f"Calculating association ratio for {len(events_catalog)} events")
    events_catalog = calc_association_ratio(events_catalog, stations, weight_ratio=0, num_workers=num_workers)
    print(f"Removing {len(events_catalog[events_catalog['association_ratio'] < association_ratio_threshold])} events with association ratio < {association_ratio_threshold}")
    discarded_indexes = set(events_catalog[events_catalog['association_ratio'] < association_ratio_threshold]['event_index'].values.tolist())
    print(f"{discarded_indexes=}")
    events_catalog = events_catalog[events_catalog['association_ratio'] >= association_ratio_threshold]
    
    print(f"Assigning unassociated event picks using origin time prediction and travel time prediction")
    events = link_unassociated_picks(events, events_catalog, DT=DT, VP=VP, VPVS_RATIO=VPVS_RATIO, discarded_indexes=discarded_indexes, num_workers=num_workers)
    events['num_picks'] = events.groupby('event_index').transform('size')
    # update the event_score and station_id of events_catalog
    tmp = events.groupby('event_index').agg({
        'event_score': 'sum',
        'station_id': lambda x: ','.join(x),
    }).reset_index()
    events_catalog.drop(columns=['event_score', 'station_id'], inplace=True)
    events_catalog = events_catalog.merge(tmp, on='event_index', how='left')
    events_catalog = events_catalog[events_catalog['event_score'] >= MIN_STATION]
    events_catalog = calc_association_ratio(events_catalog, stations, weight_ratio=0)
    events_catalog.drop(columns=['timestamp'], inplace=True)
    
    # %% link picks to events
    picks["event_index"] = -1
    picks.set_index("station_id", inplace=True)

    if num_workers <= 1:
        for group_id, event in tqdm(events.groupby("station_id"), desc="Linking picks to events"):
            event = event[event['event_index'] != -1]  # Filter out events with event_index == -1
            if len(event) == 0:
                continue
            
            # travel time tt = (tp + ts) / 2 = (1 + ps_ratio)/2 * tp => tp = tt * 2 / (1 + ps_ratio)
            # (ts - tp) = (ps_ratio - 1) tp = tt * 2 * (ps_ratio - 1) / (ps_ratio + 1)

            # event = event.sort_values(by='num_picks', ascending=True)
            # since the argmax returns the first of the max, the largest num_picks should be the first
            event = event.sort_values(by='num_picks', ascending=False) 
            ps_delta = event["travel_time"].values * 2 * (VPVS_RATIO - 1) / (VPVS_RATIO + 1)
            # t1 = event["timestamp_center"].values - ps_delta * 1.1 - 1.0
            # t2 = event["timestamp_center"].values + ps_delta * 1.1 + 1.0
            t1 = event["timestamp_center"].values - (ps_delta * 0.6 + 1.0)
            t2 = event["timestamp_center"].values + (ps_delta * 0.6 + 1.0)
            t_center = event["timestamp_center"].values

            picks_ = picks.loc[group_id, "timestamp"].values  # (Npk, )
            phase_types = picks.loc[group_id, 'phase_type'].values  # shape (Npk, )
            #mask = (picks_[None, :] >= t1[:, None]) & (picks_[None, :] <= t2[:, None])  # (Nev, Npk)
            mask = ((picks_[None, :] >= t1[:, None]) & (picks_[None, :] <= t_center[:, None]) & (phase_types[None, :] == 'P')) | \
                   ((picks_[None, :] >= t_center[:, None]) & (picks_[None, :] <= t2[:, None]) & (phase_types[None, :] == 'S'))  # (Nev, Npk)
            mask = (picks_[None, :] >= t1[:, None]) & (picks_[None, :] <= t2[:, None])  # (Nev, Npk)
            # picks.loc[group_id, "event_index"] = np.where(
            #     mask.any(axis=0), index.values[mask.argmax(axis=0)], picks.loc[group_id, "event_index"]
            # )
            mask_true = mask.any(axis=0)
            mask_idx = mask.argmax(axis=0)
            picks.loc[group_id, "event_index"] = np.where(mask_true, event["event_index"].values[mask_idx], -1)
            picks.loc[group_id, "sp_ratio"] = np.where(mask_true, event["sp_ratio"].values[mask_idx], np.nan)
            picks.loc[group_id, "event_amplitude"] = np.where(mask_true, event["event_amplitude"].values[mask_idx], np.nan)

        picks.reset_index(inplace=True)
    else:
        print(f"Using {args.num_workers} workers to link picks to events")
        picks = link_picks_to_events(phase_picks=picks, event_picks=events, VPVS_RATIO=VPVS_RATIO)
    
    print(f"Number of associated events: {len(events['event_index'].unique()):,} ({len(events[events['event_index'] != -1]):,} / {len(events):,})")
    print(f"Number of associated picks: {len(picks[picks['event_index'] != -1]):,} / {len(picks):,}")

    # plotting_debug(
    #     xt,
    #     hist_numpy,
    #     topk_index,
    #     topk_score,
    #     picks,
    #     events,
    #     stations,
    #     {"DT": DT, "MIN_STATION": MIN_STATION, "timestamp0": timestamp0},
    # )
    
    return events, picks


def run_association(
    root_path: str,
    region: str,
    jdays: list,
    config: Dict,
    num_workers: int = 1,
    gpu: bool = False,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    for jday in jdays:

        # %%
        print(f"Processing {jday}")

        year = int(jday.split(".")[0])
        jday = int(jday.split(".")[1])

        # %%
        data_path = f"{region}/phasenet_plus/{year:04d}"
        result_path = f"{region}/phasenet_plus/{year:04d}"
        if not os.path.exists(f"{root_path}/{result_path}"):
            os.makedirs(f"{root_path}/{result_path}")

        # %%
        stations = pd.read_json(f"{root_path}/{region}/obspy/stations.json", orient="index")
        stations["station_id"] = stations.index
        events = pd.read_csv(
            f"{root_path}/{data_path}/{year:04d}.{jday:03d}.events.csv", parse_dates=["center_time", "event_time"]
        )
        picks = pd.read_csv(f"{root_path}/{data_path}/{year:04d}.{jday:03d}.picks.csv", parse_dates=["phase_time"])

        events, picks = associate(picks, events, stations, config, num_workers=num_workers, gpu=gpu)

        # %%
        # plt.figure(figsize=(10, 5))
        # plt.scatter(
        #     picks["phase_time"],
        #     picks["station_id"].map(mapping),
        #     # c=[f"C{x}" if x != -1 else "k" for x in picks["phase_type"]],
        #     c=["b" if x == "P" else "r" if x == "S" else "k" for x in picks["phase_type"]],
        #     marker=".",
        #     s=3,
        # )
        # plt.scatter([], [], c="b", label="P")
        # plt.scatter([], [], c="r", label="S")
        # plt.legend(loc="upper right")
        # plt.ylabel("Station #")
        # plt.xlim(pd.Timestamp("2019-07-04T17:40:00"), pd.Timestamp("2019-07-04T17:45:00"))
        # # plt.xlim(pd.Timestamp("2019-07-04T18:01:50"), pd.Timestamp("2019-07-04T18:05:00"))
        # plt.savefig("demo_phasenet_plus_picks.png")

        # plt.figure(figsize=(10, 5))
        # plt.scatter(
        #     events["event_time"],
        #     events["station_id"].map(mapping),
        #     # c=[f"C{x}" if x != -1 else "k" for x in events["event_index"]],
        #     c=["g" for x in events["event_index"]],
        #     marker="x",
        #     s=10,
        # )
        # plt.scatter(
        #     picks["phase_time"],
        #     picks["station_id"].map(mapping),
        #     # c=[f"C{x}" if x != -1 else "k" for x in picks["event_index"]],
        #     c=["b" if x == "P" else "r" if x == "S" else "k" for x in picks["phase_type"]],
        #     marker=".",
        #     s=3,
        #     alpha=0.2,
        # )
        # plt.scatter([], [], c="b", label="P")
        # plt.scatter([], [], c="r", label="S")
        # plt.scatter([], [], c="g", marker="x", label="Event OT")
        # plt.legend(loc="upper right")
        # plt.ylabel("Station #")
        # plt.xlim(pd.Timestamp("2019-07-04T17:40:00"), pd.Timestamp("2019-07-04T17:45:00"))
        # plt.savefig("demo_phasenet_plus_events.png")

        # plt.figure(figsize=(10, 5))
        # plt.scatter(
        #     events["event_time"],
        #     events["station_id"].map(mapping),
        #     c=[f"C{x}" if x != -1 else "k" for x in events["event_index"]],
        #     marker="x",
        #     s=10,
        # )
        # plt.scatter(
        #     picks["phase_time"],
        #     picks["station_id"].map(mapping),
        #     # c=[f"C{x}" if x != -1 else "k" for x in picks["event_index"]],
        #     c=["b" if x == "P" else "r" if x == "S" else "k" for x in picks["phase_type"]],
        #     marker=".",
        #     s=3,
        #     alpha=0.2,
        # )
        # plt.scatter([], [], c="b", label="P")
        # plt.scatter([], [], c="r", label="S")
        # plt.scatter([], [], c="k", marker="x", label="Event OT")
        # plt.legend(loc="upper right")
        # plt.ylabel("Station #")
        # plt.xlim(pd.Timestamp("2019-07-04T17:40:00"), pd.Timestamp("2019-07-04T17:45:00"))
        # # plt.xlim(pd.Timestamp("2019-07-04T18:01:50"), pd.Timestamp("2019-07-04T18:05:00"))
        # plt.savefig("demo_phasenet_plus.png")

        # %%
        # events.drop(columns=["timestamp", "timestamp_center"], inplace=True, errors="ignore")
        # picks.drop(columns=["timestamp"], inplace=True, errors="ignore")
        events.to_csv(f"{root_path}/{result_path}/{year:04d}.{jday:03d}.events_associated.csv", index=False)
        picks.to_csv(f"{root_path}/{result_path}/{year:04d}.{jday:03d}.picks_associated.csv", index=False)

        # if protocol != "file":
        #     fs.put(
        #         f"{root_path}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv",
        #         f"{bucket}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv",
        #     )
        #     fs.put(
        #         f"{root_path}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv",
        #         f"{bucket}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv",
        #     )

        # # copy to results/phase_association
        # if not os.path.exists(f"{root_path}/{region}/results/phase_association"):
        #     os.makedirs(f"{root_path}/{region}/results/phase_association")
        # os.system(
        #     f"cp {root_path}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv {root_path}/{region}/results/phase_association/events_{rank:03d}.csv"
        # )
        # os.system(
        #     f"cp {root_path}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv {root_path}/{region}/results/phase_association/picks_{rank:03d}.csv"
        # )
        # if protocol != "file":
        #     fs.put(
        #         f"{root_path}/{result_path}/phasenet_plus_association_events_{rank:03d}.csv",
        #         f"{bucket}/{region}/results/phase_association/events_{rank:03d}.csv",
        #     )
        #     fs.put(
        #         f"{root_path}/{result_path}/phasenet_plus_association_picks_{rank:03d}.csv",
        #         f"{bucket}/{region}/results/phase_association/picks_{rank:03d}.csv",
        #     )


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    region = args.region
    num_nodes = args.num_nodes
    node_rank = args.node_rank
    gpu = args.gpu
    num_workers = args.num_workers
    if num_workers > 1:
        try:
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=num_workers, progress_bar=False)
        except ImportError:
            print("pandarallel not installed, using single process instead.")
            num_workers = 1

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    config.update({"VPVS_RATIO": 1.73, "VP": 6.0})

    jdays = sorted(glob(f"{root_path}/{region}/phasenet_plus/????/????.???.events.csv"))
    jdays = [jday.split("/")[-1].replace(".csv", "") for jday in jdays]
    print(f"Number of event files: {len(jdays)}")

    jdays = [jdays[i::num_nodes] for i in range(num_nodes)][node_rank]

    run_association(root_path=root_path, region=region, jdays=jdays, config=config, num_workers=num_workers, gpu=gpu)

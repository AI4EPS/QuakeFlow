# %%
"""
Predict P and S wave polarities from focal mechanisms.

Supports both HDF5 and Parquet input. Computes takeoff angles using one of
three methods: uniform velocity (simple), eikonal solver, or ObsPy TauP.
Uses SCSN 1D velocity model (Hadley & Kanamori, 1977).

Automatically uses inverted focal mechanisms (focal_mechanisms.csv) from the
output directory if available, falling back to catalog FM from the dataset.

Only saves polarity records where catalog label, FM prediction, and waveform
first-motion detection all agree ("consistent" polarities).

Usage:
    python predict.py --data_path ../NCEDC/dataset/2024/001.h5 --takeoff_method uniform
    python predict.py --data_path ../NCEDC/dataset/2024/001.h5 --takeoff_method taup
"""
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from eikonal2d import calc_ray_param, init_eikonal2d
from polarity import calc_radiation_pattern
from tqdm import tqdm
from utils import SAMPLING_RATE, detect_first_motion, load_dataset, load_waveforms

# SCSN 1D velocity model (Hadley & Kanamori, 1977)
VELOCITY_MODEL = {
    "Z": [0.0, 5.5, 16.0, 32.0],
    "P": [5.5, 6.3, 6.7, 7.8],
    "vp_vs_ratio": 1.73,
}


# %%
def build_taup_velocity_model():
    """Build TauP model: SCSN crust on top of iasp91 mantle/core."""
    from obspy.taup import TauPyModel
    from obspy.taup.taup_create import build_taup_model

    zz = VELOCITY_MODEL["Z"]
    vp = VELOCITY_MODEL["P"]
    ratio = VELOCITY_MODEL["vp_vs_ratio"]
    vs = [v / ratio for v in vp]
    rho = [0.32 * v + 0.77 for v in vp]  # Nafe-Drake approximation

    from importlib.resources import files
    iasp91_path = str(files("obspy.taup.data") / "iasp91.tvel")
    mantle_lines = []
    with open(iasp91_path) as f:
        f.readline()  # header1
        f.readline()  # header2
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                depth = float(parts[0])
                if depth >= zz[-1]:
                    mantle_lines.append(line)

    tvel_path = os.path.join(tempfile.gettempdir(), "scsn.tvel")
    with open(tvel_path, "w") as f:
        f.write("scsn - Hadley & Kanamori 1977 crust + iasp91 mantle\n")
        f.write("scsn\n")
        for i in range(len(zz)):
            f.write(f"{zz[i]:.1f}  {vp[i]:.4f}  {vs[i]:.4f}  {rho[i]:.4f}\n")
            if i < len(zz) - 1:
                f.write(f"{zz[i+1]:.1f}  {vp[i]:.4f}  {vs[i]:.4f}  {rho[i]:.4f}\n")
        for line in mantle_lines:
            f.write(line)

    build_taup_model(tvel_path, output_folder=tempfile.gettempdir())
    model = TauPyModel(model=os.path.join(tempfile.gettempdir(), "scsn"))
    return model


def compute_takeoff_taup(model, depth_km, distance_km, phase_list=("p", "P")):
    """Compute takeoff angle using TauP for a single source-receiver pair."""
    dist_deg = distance_km / 111.19
    try:
        arrivals = model.get_travel_times(
            source_depth_in_km=max(depth_km, 0.001),
            distance_in_degree=max(dist_deg, 0.001),
            phase_list=phase_list,
        )
        if arrivals:
            return arrivals[0].takeoff_angle
    except Exception:
        pass
    return np.nan


# %%
def compute_polarities(phases, takeoff_method="taup"):
    """
    Compute radiation pattern polarities for all phases.

    Args:
        phases: DataFrame with event/station coordinates, strike/dip/rake
        takeoff_method: "uniform", "eikonal", or "taup"

    Returns:
        DataFrame with predicted polarity columns and takeoff angles
    """
    fm_mask = phases["strike"].notna() & phases["dip"].notna() & phases["rake"].notna()
    phases = phases[fm_mask].copy()
    if len(phases) == 0:
        print("No events with focal mechanisms found.")
        return pd.DataFrame()

    print(f"Computing polarities for {len(phases)} records from {phases['event_id'].nunique()} events with FM")

    # Step 1: Compute azimuth and distance
    ray_simple = calc_ray_param(
        phases["event_longitude"].values,
        phases["event_latitude"].values,
        phases["event_depth_km"].values,
        phases["station_longitude"].values,
        phases["station_latitude"].values,
        phases["station_depth_km"].values,
        np.zeros(len(phases), dtype=int),
        None,
    )
    phases["azimuth"] = ray_simple["azimuth"]
    phases["distance_km"] = ray_simple["distance_km"]
    phases["takeoff_uniform"] = ray_simple["takeoff"]

    # Step 2: Compute takeoff angles
    if takeoff_method == "uniform":
        takeoff_col = "takeoff_uniform"

    elif takeoff_method == "eikonal":
        zz = VELOCITY_MODEL["Z"]
        vp = VELOCITY_MODEL["P"]
        vs = [v / VELOCITY_MODEL["vp_vs_ratio"] for v in vp]

        R_max = phases["distance_km"].max() + 10
        Z_max = phases["event_depth_km"].max() - phases["station_depth_km"].min() + 10

        config = {
            "vel": {"Z": zz, "P": vp, "S": vs},
            "h": 1.0,
            "xlim_km": [0, R_max],
            "ylim_km": [0, R_max],
            "zlim_km": [0, Z_max],
        }
        config["eikonal"] = init_eikonal2d(config)

        ray_eikonal = calc_ray_param(
            phases["event_longitude"].values,
            phases["event_latitude"].values,
            phases["event_depth_km"].values,
            phases["station_longitude"].values,
            phases["station_latitude"].values,
            phases["station_depth_km"].values,
            np.zeros(len(phases), dtype=int),
            config["eikonal"],
        )
        phases["takeoff_eikonal"] = ray_eikonal["takeoff"]
        takeoff_col = "takeoff_eikonal"

    elif takeoff_method == "taup":
        print("Building TauP velocity model...")
        taup_model = build_taup_velocity_model()

        print("Computing takeoff angles with TauP...")
        takeoff_angles = np.full(len(phases), np.nan)
        for event_id, group in tqdm(phases.groupby("event_id"), desc="TauP takeoff"):
            depth = group["event_depth_km"].iloc[0]
            for idx, (_, row) in zip(group.index, group.iterrows()):
                pos = phases.index.get_loc(idx)
                takeoff_angles[pos] = compute_takeoff_taup(
                    taup_model, depth, row["distance_km"]
                )
        phases["takeoff_taup"] = takeoff_angles
        takeoff_col = "takeoff_taup"

    else:
        raise ValueError(f"Unknown takeoff_method: {takeoff_method}")

    # Step 3: Compute radiation patterns per event
    polarities = []
    for event_id, group in tqdm(phases.groupby("event_id"), desc="Computing polarities"):
        strike = group["strike"].iloc[0]
        dip = group["dip"].iloc[0]
        rake = group["rake"].iloc[0]

        radiation = calc_radiation_pattern(
            strike, dip, rake,
            group[takeoff_col].values, group["azimuth"].values,
        )

        sub_df = group[["event_id", "network", "station", "location", "instrument",
                        "p_phase_polarity", "p_phase_index", "s_phase_index",
                        "azimuth", "distance_km",
                        takeoff_col]].copy()
        P_ENZ = radiation["P_ENZ"]
        S_ENZ = radiation["S_ENZ"]
        sub_df["p_polarity_e"] = np.round(P_ENZ[:, 0], 6)
        sub_df["p_polarity_n"] = np.round(P_ENZ[:, 1], 6)
        sub_df["p_polarity_z"] = np.round(P_ENZ[:, 2], 6)
        sub_df["s_polarity_e"] = np.round(S_ENZ[:, 0], 6)
        sub_df["s_polarity_n"] = np.round(S_ENZ[:, 1], 6)
        sub_df["s_polarity_z"] = np.round(S_ENZ[:, 2], 6)
        sub_df["log_sp_ratio"] = np.round(radiation["log_SP"], 3)
        polarities.append(sub_df)

    return pd.concat(polarities, ignore_index=True)


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict polarities from focal mechanisms")
    parser.add_argument("--data_path", type=str, required=True, help="Path to HDF5 or Parquet dataset file")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: <data_path>.polarities.csv)")
    parser.add_argument("--takeoff_method", type=str, default="uniform", choices=["uniform", "eikonal", "taup"],
                        help="Takeoff angle method: uniform, eikonal, or taup (default: uniform)")
    args = parser.parse_args()

    data_path = Path(args.data_path)

    # Resolve output directory
    resolved = data_path.resolve()
    parts = resolved.parts
    try:
        ds_idx = parts.index("dataset")
        region = parts[ds_idx - 1]  # NCEDC or SCEDC
        sub_path = Path(*parts[ds_idx+1:]).parent / data_path.stem
        out_dir = Path(__file__).parent / region / sub_path
    except ValueError:
        out_dir = Path(__file__).parent / data_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(out_dir / "polarities.csv")

    print(f"Loading data from {data_path}")
    phases = load_dataset(data_path)
    print(f"Loaded {len(phases)} records from {phases['event_id'].nunique()} events")

    # Auto-detect inverted focal mechanisms, fall back to catalog FM
    fm_file = out_dir / "focal_mechanisms.csv"
    if fm_file.exists():
        fm_df = pd.read_csv(fm_file)
        fm_ok = fm_df[["event_id", "strike", "dip", "rake"]]
        print(f"Using {len(fm_ok)} inverted FMs from {fm_file}")
        phases = phases.drop(columns=["strike", "dip", "rake"], errors="ignore")
        phases = phases.merge(fm_ok, on="event_id", how="left")
    else:
        print("No inverted FM found, using catalog strike/dip/rake from dataset")

    polarity_df = compute_polarities(phases, takeoff_method=args.takeoff_method)

    if len(polarity_df) == 0:
        print("No polarities computed.")
    else:
        # Detect first motion from waveforms
        print("Loading waveforms for first-motion detection...")
        waveforms = load_waveforms(data_path)

        detected_labels = []
        for _, row in tqdm(polarity_df.iterrows(), total=len(polarity_df), desc="Detecting first motion"):
            key = (row["event_id"], row["network"], row["station"],
                   row["location"], row["instrument"])
            wf = waveforms.get(key)
            if wf is None:
                detected_labels.append("")
                continue
            # Look up p_phase_index from the original phases DataFrame
            match = phases[
                (phases["event_id"] == row["event_id"]) &
                (phases["network"] == row["network"]) &
                (phases["station"] == row["station"])
            ]
            if len(match) == 0 or match.iloc[0].get("p_phase_index") is None:
                detected_labels.append("")
                continue
            p_idx = match.iloc[0]["p_phase_index"]
            result = detect_first_motion(wf, p_idx)
            detected_labels.append(result["label"] if result else "")

        polarity_df["detected_polarity"] = detected_labels
        polarity_df["predicted_polarity"] = np.where(polarity_df["p_polarity_z"] > 0, "U", "D")

        # Filter to consistent polarities: catalog label, prediction, and detection all agree
        has_catalog = polarity_df["p_phase_polarity"].isin(["U", "D"])
        has_detection = polarity_df["detected_polarity"].isin(["U", "D"])
        all_three = has_catalog & has_detection

        n_total = len(polarity_df)
        n_with_all = all_three.sum()
        if n_with_all > 0:
            match_pred_cat = (polarity_df.loc[all_three, "predicted_polarity"] == polarity_df.loc[all_three, "p_phase_polarity"]).sum()
            match_det_cat = (polarity_df.loc[all_three, "detected_polarity"] == polarity_df.loc[all_three, "p_phase_polarity"]).sum()
            consistent = (
                (polarity_df["p_phase_polarity"] == polarity_df["predicted_polarity"]) &
                (polarity_df["p_phase_polarity"] == polarity_df["detected_polarity"])
            )
            n_consistent = consistent.sum()

            print(f"\nConsistency check ({n_with_all} records with all 3 sources):")
            print(f"  Prediction vs catalog:  {match_pred_cat}/{n_with_all} ({match_pred_cat/n_with_all*100:.1f}%)")
            print(f"  Detection vs catalog:   {match_det_cat}/{n_with_all} ({match_det_cat/n_with_all*100:.1f}%)")
            print(f"  All three agree:        {n_consistent}/{n_with_all} ({n_consistent/n_with_all*100:.1f}%)")

            # Save only consistent records
            consistent_df = polarity_df[consistent].copy()
            consistent_df.to_csv(output_path, index=False)
            print(f"\nSaved {len(consistent_df)} consistent polarity predictions to {output_path}")
        else:
            print("No records with all three polarity sources (catalog, prediction, detection).")
            polarity_df.to_csv(output_path, index=False)
            print(f"Saved {len(polarity_df)} polarity predictions (unfiltered) to {output_path}")

# %%

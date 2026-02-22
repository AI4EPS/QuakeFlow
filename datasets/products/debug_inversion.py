# %%
"""Diagnose why inverted FM underperforms SCEDC catalog."""
import numpy as np
from utils import load_dataset, load_waveforms
from inversion import (
    build_focal_mechanism_grid, _build_moment_tensors,
    invert_focal_mechanism, prepare_observations, measure_sp_ratio,
    _forward_model_batch, _build_ray_vectors,
    polarity_misfit, sp_ratio_misfit, combined_misfit,
)
from polarity import calc_radiation_pattern

# %% Load SCEDC 2024
data_path = "../SCEDC/dataset/2024/001.h5"
df = load_dataset(data_path)
waveforms = load_waveforms(data_path)

has_fm = df["strike"].notna() & df["dip"].notna() & df["rake"].notna()
has_pol = df["p_phase_polarity"].isin(["U", "D"])
fm_events = df[has_fm & has_pol]["event_id"].unique()

grid = build_focal_mechanism_grid(5.0, 5.0, 5.0)
M_batch = _build_moment_tensors(grid)

# %% Pick a well-constrained event
print("=== Event-level diagnostics ===")
for eid in fm_events[:5]:
    event_data = df[df["event_id"] == eid]
    n_pol = event_data["p_phase_polarity"].isin(["U", "D"]).sum()
    cat_s, cat_d, cat_r = event_data["strike"].iloc[0], event_data["dip"].iloc[0], event_data["rake"].iloc[0]
    print(f"\n--- {eid} (nPol={n_pol}, catalog={cat_s:.0f}/{cat_d:.0f}/{cat_r:.0f}) ---")

    obs = prepare_observations(event_data, waveforms, takeoff_method="uniform")
    if obs is None:
        print("  No observations")
        continue

    # Check S/P ratio measurement
    n_sp_measured = np.sum(~np.isnan(obs["log_SP_obs"]))
    print(f"  Observations: nPol={obs['n_pol']}, nSP={n_sp_measured}, nSta={obs['n_stations']}")
    if n_sp_measured > 0:
        valid_sp = ~np.isnan(obs["log_SP_obs"])
        print(f"  S/P obs values: {obs['log_SP_obs'][valid_sp]}")

    # Forward model at catalog solution
    rad_cat = calc_radiation_pattern(cat_s, cat_d, cat_r,
                                     obs["takeoff"], obs["azimuth"])
    pol_mask = obs["polarity_obs"] != 0
    pred_cat = np.sign(rad_cat["P"][pol_mask])
    cat_acc = np.mean(pred_cat == obs["polarity_obs"][pol_mask])
    print(f"  Catalog FM polarity fit: {cat_acc*100:.1f}%")
    print(f"  Catalog FM log_SP predicted: min={rad_cat['log_SP'].min():.2f} max={rad_cat['log_SP'].max():.2f}")

    # Forward model at all grid points - check S/P misfit behavior
    gamma, e_sv, e_sh = _build_ray_vectors(obs["takeoff"], obs["azimuth"])
    P_all, log_SP_pred = _forward_model_batch(M_batch, gamma, e_sv, e_sh)

    # Check how many grid points have non-NaN S/P misfit
    if n_sp_measured > 0:
        mf_sp = sp_ratio_misfit(log_SP_pred, obs["log_SP_obs"], obs["weights_sp"],
                                P_all, null_zone_fraction=0.1)
        n_valid_sp_misfit = np.sum(~np.isnan(mf_sp))
        print(f"  S/P misfit: {n_valid_sp_misfit}/{len(grid)} grid points have valid S/P misfit")
        if n_valid_sp_misfit > 0:
            print(f"  S/P misfit range: {np.nanmin(mf_sp):.3f} - {np.nanmax(mf_sp):.3f}")

    # Check polarity misfit
    mf_pol = polarity_misfit(P_all, obs["polarity_obs"], obs["weights_pol"], 0.1)
    print(f"  Polarity misfit range: {mf_pol.min():.3f} - {mf_pol.max():.3f}")
    best_pol_idx = np.argmin(mf_pol)
    print(f"  Best polarity-only: {grid[best_pol_idx]} misfit={mf_pol[best_pol_idx]:.3f}")

    # Check what happens at catalog mechanism (find nearest grid point)
    diffs = np.abs(grid - np.array([cat_s, cat_d, cat_r]))
    diffs[:, 0] = np.minimum(diffs[:, 0], 360 - diffs[:, 0])  # circular strike
    diffs[:, 2] = np.minimum(np.abs(diffs[:, 2]), 360 - np.abs(diffs[:, 2]))  # circular rake
    nearest_idx = np.argmin(np.sum(diffs, axis=1))
    print(f"  Nearest grid to catalog: {grid[nearest_idx]} polMisfit={mf_pol[nearest_idx]:.3f}")

# %% Takeoff angle comparison
print("\n\n=== Takeoff angle impact ===")
eid = fm_events[0]
event_data = df[df["event_id"] == eid]
cat_s, cat_d, cat_r = event_data["strike"].iloc[0], event_data["dip"].iloc[0], event_data["rake"].iloc[0]

obs_uniform = prepare_observations(event_data, waveforms, takeoff_method="uniform")

from eikonal2d import init_eikonal2d
from predict import build_taup_velocity_model
from inversion import VELOCITY_MODEL

# Eikonal
zz = VELOCITY_MODEL["Z"]
vp = VELOCITY_MODEL["P"]
vs = [v / VELOCITY_MODEL["vp_vs_ratio"] for v in vp]
config = {
    "vel": {"Z": zz, "P": vp, "S": vs},
    "h": 1.0,
    "xlim_km": [0, 500],
    "ylim_km": [0, 500],
    "zlim_km": [0, 50],
}
eikonal_config = init_eikonal2d(config)
obs_eikonal = prepare_observations(event_data, waveforms, takeoff_method="eikonal",
                                    eikonal_config=eikonal_config)

# TauP
taup_model = build_taup_velocity_model()
obs_taup = prepare_observations(event_data, waveforms, takeoff_method="taup",
                                 taup_model=taup_model)

print(f"\nEvent {eid} — catalog: {cat_s:.0f}/{cat_d:.0f}/{cat_r:.0f}")
print(f"  Takeoff angles (first 10 stations):")
print(f"  {'Station':<12} {'Uniform':>8} {'Eikonal':>8} {'TauP':>8} {'Pol':>4}")
for i in range(min(10, len(obs_uniform["takeoff"]))):
    sid = obs_uniform["station_ids"][i]
    tu = obs_uniform["takeoff"][i]
    te = obs_eikonal["takeoff"][i] if obs_eikonal else float("nan")
    tt = obs_taup["takeoff"][i] if obs_taup else float("nan")
    p = obs_uniform["polarity_obs"][i]
    pl = "U" if p > 0 else ("D" if p < 0 else "-")
    print(f"  {sid:<12} {tu:>8.1f} {te:>8.1f} {tt:>8.1f}   {pl}")

# Compare accuracy at catalog FM for each takeoff method
for label, obs in [("uniform", obs_uniform), ("eikonal", obs_eikonal), ("taup", obs_taup)]:
    if obs is None:
        continue
    pol_mask = obs["polarity_obs"] != 0
    if pol_mask.sum() == 0:
        continue
    rad = calc_radiation_pattern(cat_s, cat_d, cat_r, obs["takeoff"][pol_mask], obs["azimuth"][pol_mask])
    acc = np.mean(np.sign(rad["P"]) == obs["polarity_obs"][pol_mask])
    print(f"  Catalog FM accuracy with {label} takeoff: {acc*100:.1f}%")

# Run inversion with each method
for label, method, ek, tp in [
    ("uniform", "uniform", None, None),
    ("eikonal", "eikonal", eikonal_config, None),
    ("taup", "taup", None, taup_model),
]:
    result = invert_focal_mechanism(
        event_data, waveforms=waveforms, grid=grid, M_batch=M_batch,
        alpha=0.5, min_n_pol=6, takeoff_method=method,
        eikonal_config=ek, taup_model=tp,
    )
    if result["status"] == "OK":
        print(f"  Inverted with {label}: {result['strike']:.0f}/{result['dip']:.0f}/{result['rake']:.0f} "
              f"mfPol={result['misfit_pol']:.3f} Q={result['quality']}")

# %% Run full SCEDC 2024 with eikonal and taup
print("\n\n=== Full SCEDC 2024 with different takeoff methods ===")
from inversion import invert_focal_mechanism as inv_fm

for method_label, method, ek, tp in [
    ("uniform", "uniform", None, None),
    ("eikonal", "eikonal", eikonal_config, None),
    ("taup", "taup", None, taup_model),
]:
    total_inv = 0
    total_cat = 0
    total_n = 0
    for eid in fm_events:
        event_data = df[df["event_id"] == eid]
        cat_s = event_data["strike"].iloc[0]
        cat_d = event_data["dip"].iloc[0]
        cat_r = event_data["rake"].iloc[0]

        result = inv_fm(
            event_data, waveforms=waveforms, grid=grid, M_batch=M_batch,
            alpha=0.5, min_n_pol=6, takeoff_method=method,
            eikonal_config=ek, taup_model=tp,
        )
        if result["status"] != "OK":
            continue

        obs = prepare_observations(event_data, waveforms, takeoff_method=method,
                                    eikonal_config=ek, taup_model=tp)
        if obs is None:
            continue
        pol_mask = obs["polarity_obs"] != 0
        if pol_mask.sum() == 0:
            continue

        takeoff = obs["takeoff"][pol_mask]
        azimuth = obs["azimuth"][pol_mask]
        pol_obs = obs["polarity_obs"][pol_mask]

        rad_inv = calc_radiation_pattern(result["strike"], result["dip"], result["rake"], takeoff, azimuth)
        total_inv += np.sum(np.sign(rad_inv["P"]) == pol_obs)

        rad_cat = calc_radiation_pattern(cat_s, cat_d, cat_r, takeoff, azimuth)
        total_cat += np.sum(np.sign(rad_cat["P"]) == pol_obs)

        total_n += len(pol_obs)

    if total_n > 0:
        print(f"  {method_label:>8}: Inv={total_inv}/{total_n} ({total_inv/total_n*100:.1f}%)  "
              f"Cat={total_cat}/{total_n} ({total_cat/total_n*100:.1f}%)")

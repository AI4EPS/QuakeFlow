# %%
"""
Focal mechanism inversion via grid search over (strike, dip, rake).

Combines P-wave first-motion polarity and S/P amplitude ratio observations.
Uses HASH-style null-zone exclusion for outlier robustness and L1 S/P misfit.

Usage:
    python inversion.py --data_path ../NCEDC/dataset/2024/001.h5
    python inversion.py --data_path ../NCEDC/dataset/2024/001.parquet --alpha 0.5
    python inversion.py --data_path ../NCEDC/dataset/2024/001.h5 --event_id nc75110621
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from eikonal2d import calc_ray_param, init_eikonal2d
from polarity import calc_radiation_pattern
from predict import build_taup_velocity_model, compute_takeoff_taup
from utils import SAMPLING_RATE, detect_first_motion, load_dataset, load_waveforms

# SCSN 1D velocity model (Hadley & Kanamori, 1977)
VELOCITY_MODEL = {
    "Z": [0.0, 5.5, 16.0, 32.0],
    "P": [5.5, 6.3, 6.7, 7.8],
    "vp_vs_ratio": 1.73,
}


# %%
def build_focal_mechanism_grid(strike_step=5.0, dip_step=5.0, rake_step=5.0):
    """Build array of trial (strike, dip, rake) mechanisms.

    Args:
        strike_step: Grid spacing for strike in degrees
        dip_step: Grid spacing for dip in degrees
        rake_step: Grid spacing for rake in degrees

    Returns:
        ndarray of shape (N_mechanisms, 3) with columns [strike, dip, rake]
    """
    strikes = np.arange(0, 360, strike_step)
    dips = np.arange(0, 90 + dip_step / 2, dip_step)  # inclusive of 90
    rakes = np.arange(-180, 180, rake_step)
    grid = np.array(np.meshgrid(strikes, dips, rakes, indexing="ij")).reshape(3, -1).T
    return grid


# %%
def _build_moment_tensors(grid):
    """Vectorized moment tensor construction for all grid mechanisms.

    Args:
        grid: (N_mech, 3) array of [strike, dip, rake] in degrees

    Returns:
        M_batch: (N_mech, 3, 3) moment tensor array in NED coordinates
    """
    d2r = np.pi / 180.0
    s = grid[:, 0] * d2r
    d = grid[:, 1] * d2r
    r = grid[:, 2] * d2r

    # Normal vector n in NED — shape (N_mech, 3)
    n = np.stack([
        -np.sin(d) * np.sin(s),
         np.sin(d) * np.cos(s),
        -np.cos(d),
    ], axis=1)

    # Slip vector d_vec in NED — shape (N_mech, 3)
    d_vec = np.stack([
         np.cos(r) * np.cos(s) + np.cos(d) * np.sin(r) * np.sin(s),
         np.cos(r) * np.sin(s) - np.cos(d) * np.sin(r) * np.cos(s),
        -np.sin(r) * np.sin(d),
    ], axis=1)

    # M = d_vec * n^T + n * d_vec^T — shape (N_mech, 3, 3)
    M_batch = np.einsum("ki,kj->kij", d_vec, n) + np.einsum("ki,kj->kij", n, d_vec)
    return M_batch


def _build_ray_vectors(takeoff, azimuth):
    """Build ray geometry vectors from takeoff angles and azimuths.

    Args:
        takeoff: (N_sta,) takeoff angles in degrees (0=down)
        azimuth: (N_sta,) azimuths in degrees (0=north)

    Returns:
        gamma: (N_sta, 3) ray direction in NED
        e_sv: (N_sta, 3) SV polarization in NED
        e_sh: (N_sta, 3) SH polarization in NED
    """
    d2r = np.pi / 180.0
    inc = takeoff * d2r
    azi = azimuth * d2r

    gamma = np.stack([
        np.sin(inc) * np.cos(azi),
        np.sin(inc) * np.sin(azi),
        np.cos(inc),
    ], axis=1)

    e_sv = np.stack([
        np.cos(inc) * np.cos(azi),
        np.cos(inc) * np.sin(azi),
        -np.sin(inc),
    ], axis=1)

    e_sh = np.stack([
        -np.sin(azi),
        np.cos(azi),
        np.zeros_like(azi),
    ], axis=1)

    return gamma, e_sv, e_sh


def _forward_model_batch(M_batch, gamma, e_sv, e_sh, chunk_size=10000):
    """Vectorized forward model for all mechanisms and stations.

    Args:
        M_batch: (N_mech, 3, 3) moment tensors
        gamma: (N_sta, 3) ray direction vectors
        e_sv: (N_sta, 3) SV polarization vectors
        e_sh: (N_sta, 3) SH polarization vectors
        chunk_size: mechanisms per chunk for memory management

    Returns:
        P_all: (N_mech, N_sta) P-wave scalar amplitudes (sign = polarity)
        log_SP_all: (N_mech, N_sta) predicted log10(S/P) ratio, clipped to [-2, 3]
    """
    N_mech = M_batch.shape[0]
    N_sta = gamma.shape[0]
    P_all = np.empty((N_mech, N_sta), dtype=np.float32)
    log_SP_all = np.empty((N_mech, N_sta), dtype=np.float32)

    for i in range(0, N_mech, chunk_size):
        j = min(i + chunk_size, N_mech)
        M_chunk = M_batch[i:j]

        # P = gamma^T M gamma — (chunk, N_sta)
        P_chunk = np.einsum("si,kij,sj->ks", gamma, M_chunk, gamma)
        SV_chunk = np.einsum("si,kij,sj->ks", e_sv, M_chunk, gamma)
        SH_chunk = np.einsum("si,kij,sj->ks", e_sh, M_chunk, gamma)

        S_total = np.sqrt(SV_chunk**2 + SH_chunk**2)
        log_SP_chunk = np.log10(S_total / (np.abs(P_chunk) + 1e-10))
        log_SP_chunk = np.clip(log_SP_chunk, -2.0, 3.0)

        P_all[i:j] = P_chunk
        log_SP_all[i:j] = log_SP_chunk

    return P_all, log_SP_all


# %%
def measure_sp_ratio(waveform, p_phase_index, s_phase_index,
                     p_window_s=0.5, s_window_s=1.0, noise_window_s=0.5,
                     min_snr=3.0, sampling_rate=SAMPLING_RATE):
    """Measure observed log10(S/P) amplitude ratio from 3-component waveforms.

    Uses vector amplitude (L2 norm across E,N,Z) to avoid
    component-orientation bias.

    Args:
        waveform: (3, N) array with channels [E, N, Z]
        p_phase_index: sample index of P arrival
        s_phase_index: sample index of S arrival
        p_window_s: P measurement window in seconds after P
        s_window_s: S measurement window in seconds after S
        noise_window_s: pre-P noise window for SNR
        min_snr: minimum P SNR to return valid measurement
        sampling_rate: samples per second

    Returns:
        dict with 'log_sp_obs', 'p_amplitude', 's_amplitude', 'snr_p',
        or None if measurement not possible.
    """
    if p_phase_index is None or s_phase_index is None:
        return None
    if isinstance(p_phase_index, float) and np.isnan(p_phase_index):
        return None
    if isinstance(s_phase_index, float) and np.isnan(s_phase_index):
        return None

    p_idx = int(p_phase_index)
    s_idx = int(s_phase_index)
    n_samples = waveform.shape[1]
    p_win = int(p_window_s * sampling_rate)
    s_win = int(s_window_s * sampling_rate)
    noise_win = int(noise_window_s * sampling_rate)

    # Bounds check
    if p_idx - noise_win < 0 or p_idx + p_win > n_samples:
        return None
    if s_idx + s_win > n_samples:
        return None
    if s_idx <= p_idx:
        return None

    # Vector amplitude at each sample: sqrt(E^2 + N^2 + Z^2)
    noise_seg = waveform[:, p_idx - noise_win : p_idx]
    noise_rms = np.sqrt(np.mean(np.sum(noise_seg**2, axis=0)))
    if noise_rms == 0:
        return None

    p_seg = waveform[:, p_idx : p_idx + p_win]
    p_vec_amp = np.sqrt(np.sum(p_seg**2, axis=0))
    p_amplitude = np.max(p_vec_amp)

    s_seg = waveform[:, s_idx : s_idx + s_win]
    s_vec_amp = np.sqrt(np.sum(s_seg**2, axis=0))
    s_amplitude = np.max(s_vec_amp)

    snr_p = p_amplitude / noise_rms
    if snr_p < min_snr or p_amplitude == 0 or s_amplitude == 0:
        return None

    log_sp_obs = np.log10(s_amplitude / p_amplitude)
    return {
        "log_sp_obs": float(log_sp_obs),
        "p_amplitude": float(p_amplitude),
        "s_amplitude": float(s_amplitude),
        "snr_p": float(snr_p),
    }


# %%
def polarity_misfit(P_predicted, polarity_obs, weights=None, null_zone_fraction=0.0):
    """HASH-style polarity misfit with null-zone exclusion.

    Stations near nodal planes (where |P| is small relative to the maximum)
    are excluded — this is the primary mechanism for outlier robustness.

    Args:
        P_predicted: (N_mech, N_sta) signed P amplitudes
        polarity_obs: (N_sta,) observed polarities: +1=Up, -1=Down, 0=unknown
        weights: (N_sta,) per-station weights (default: uniform)
        null_zone_fraction: threshold for nodal-plane exclusion (default 0.1)

    Returns:
        misfit: (N_mech,) polarity misfit fraction in [0, 1]
    """
    N_mech, N_sta = P_predicted.shape
    if weights is None:
        weights = np.ones(N_sta, dtype=np.float32)

    has_polarity = polarity_obs != 0  # (N_sta,)

    # Null zone: per-mechanism, exclude stations where |P| is small
    # relative to the maximum |P| across stations for that mechanism
    abs_P = np.abs(P_predicted)  # (N_mech, N_sta)
    max_abs_P = np.max(abs_P, axis=1, keepdims=True)  # (N_mech, 1)
    not_nodal = abs_P >= null_zone_fraction * max_abs_P  # (N_mech, N_sta)

    # Valid mask: has observed polarity AND not in null zone
    valid = has_polarity[np.newaxis, :] & not_nodal  # (N_mech, N_sta)

    # Mismatch: predicted sign != observed sign
    # Use np.sign but treat P_predicted==0 as excluded (already handled by null zone,
    # but explicitly exclude to avoid sign(0)=0 mismatching both +1 and -1)
    pred_sign = np.sign(P_predicted)  # (N_mech, N_sta)
    obs_sign = polarity_obs[np.newaxis, :]  # (1, N_sta)
    valid = valid & (pred_sign != 0)  # exclude exactly-zero predictions
    mismatch = pred_sign != obs_sign  # (N_mech, N_sta)

    # Weighted misfit
    w = weights[np.newaxis, :]  # (1, N_sta)
    wrong_weight = np.sum(w * valid * mismatch, axis=1)  # (N_mech,)
    total_weight = np.sum(w * valid, axis=1)  # (N_mech,)

    misfit = np.where(total_weight > 0, wrong_weight / total_weight, 1.0)
    return misfit.astype(np.float32)


def sp_ratio_misfit(log_SP_predicted, log_SP_obs, weights=None,
                    P_predicted=None, null_zone_fraction=0.0):
    """S/P ratio misfit in log10 space using L1 norm.

    Uses absolute deviation (L1) for robustness to outlier measurements.
    Applies null-zone exclusion based on P amplitude, but only among
    S/P stations (not all stations) to avoid over-exclusion when most
    stations lack S/P observations.

    Args:
        log_SP_predicted: (N_mech, N_sta) predicted log10(S/P)
        log_SP_obs: (N_sta,) observed log10(S/P), NaN if unavailable
        weights: (N_sta,) per-station weights (default: uniform)
        P_predicted: (N_mech, N_sta) P amplitudes for null-zone check (optional)
        null_zone_fraction: nodal-plane exclusion threshold

    Returns:
        misfit: (N_mech,) mean absolute deviation in log10 units
    """
    N_mech, N_sta = log_SP_predicted.shape
    if weights is None:
        weights = np.ones(N_sta, dtype=np.float32)

    has_sp = ~np.isnan(log_SP_obs)  # (N_sta,)

    # Null zone: compute max |P| only among S/P stations, not all stations.
    # Using all stations inflates the threshold and excludes most S/P stations
    # when S/P coverage is sparse.
    if P_predicted is not None and has_sp.any():
        abs_P_sp = np.abs(P_predicted[:, has_sp])  # (N_mech, N_sp_stations)
        max_abs_P_sp = np.max(abs_P_sp, axis=1, keepdims=True)  # (N_mech, 1)
        # Apply threshold only to S/P stations
        not_nodal_sp = abs_P_sp >= null_zone_fraction * max_abs_P_sp  # (N_mech, N_sp)

        # Expand back to full station array
        not_nodal = np.ones((N_mech, N_sta), dtype=bool)
        not_nodal[:, has_sp] = not_nodal_sp
    else:
        not_nodal = np.ones((N_mech, N_sta), dtype=bool)

    valid = has_sp[np.newaxis, :] & not_nodal  # (N_mech, N_sta)

    # Fill NaN obs with 0 before computing residuals to prevent NaN propagation
    log_SP_obs_filled = np.where(has_sp, log_SP_obs, 0.0)
    residuals = np.abs(log_SP_predicted - log_SP_obs_filled[np.newaxis, :])  # (N_mech, N_sta)

    w = weights[np.newaxis, :]
    weighted_abs = np.sum(w * valid * residuals, axis=1)
    total_weight = np.sum(w * valid, axis=1)

    misfit = np.where(total_weight > 0, weighted_abs / total_weight, np.nan)
    return misfit.astype(np.float32)


def combined_misfit(misfit_pol, misfit_sp, alpha=0.5):
    """Combine polarity and S/P ratio misfits with percentile normalization.

    Each component is normalized to its 5th-95th percentile range so they
    contribute equally regardless of their natural scales.

    Args:
        misfit_pol: (N_mech,) polarity misfit fraction [0, 1]
        misfit_sp: (N_mech,) S/P ratio misfit (log10 units), may contain NaN
        alpha: weight of polarity misfit in [0, 1]

    Returns:
        misfit_total: (N_mech,) combined misfit
    """
    has_sp = np.any(~np.isnan(misfit_sp))
    has_pol = np.any(np.isfinite(misfit_pol))

    if not has_pol and not has_sp:
        return np.full_like(misfit_pol, np.nan)

    if not has_sp:
        return misfit_pol

    if not has_pol:
        sp_filled = np.where(np.isnan(misfit_sp), np.nanmax(misfit_sp), misfit_sp)
        return sp_filled

    # Percentile normalization
    eps = 1e-10
    p5_pol, p95_pol = np.percentile(misfit_pol, [5, 95])
    pol_norm = (misfit_pol - p5_pol) / (p95_pol - p5_pol + eps)

    sp_valid = misfit_sp[~np.isnan(misfit_sp)]
    if len(sp_valid) == 0:
        return misfit_pol
    p5_sp, p95_sp = np.percentile(sp_valid, [5, 95])
    sp_filled = np.where(np.isnan(misfit_sp), np.nanmax(misfit_sp), misfit_sp)
    sp_norm = (sp_filled - p5_sp) / (p95_sp - p5_sp + eps)

    return alpha * pol_norm + (1 - alpha) * sp_norm


# %%
def _circular_std(angles_deg):
    """Circular standard deviation for angular data.

    Args:
        angles_deg: array of angles in degrees

    Returns:
        Circular standard deviation in degrees
    """
    if len(angles_deg) < 2:
        return 999.0
    rad = np.deg2rad(angles_deg)
    mean_cos = np.mean(np.cos(rad))
    mean_sin = np.mean(np.sin(rad))
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    R = np.clip(R, 1e-10, 1.0)
    return np.rad2deg(np.sqrt(-2.0 * np.log(R)))


def estimate_uncertainty(misfit_surface, grid, confidence_level=0.1):
    """Estimate focal mechanism uncertainty from misfit surface.

    All mechanisms within (min_misfit + confidence_level) are considered
    acceptable. Uncertainty is the spread (std) of acceptable mechanisms.

    Args:
        misfit_surface: (N_mech,) combined misfit values
        grid: (N_mech, 3) [strike, dip, rake] in degrees
        confidence_level: misfit threshold above minimum

    Returns:
        dict with strike_unc, dip_unc, rake_unc, n_acceptable, quality
    """
    if np.all(np.isnan(misfit_surface)):
        return {
            "strike_unc": 999.0, "dip_unc": 999.0, "rake_unc": 999.0,
            "n_acceptable": 0, "quality": "D",
        }

    min_misfit = np.nanmin(misfit_surface)
    acceptable = misfit_surface <= min_misfit + confidence_level
    n_acc = np.sum(acceptable)

    if n_acc < 2:
        return {
            "strike_unc": 0.0,
            "dip_unc": 0.0,
            "rake_unc": 0.0,
            "n_acceptable": int(n_acc),
            "quality": "A",
        }

    acc_grid = grid[acceptable]
    strike_unc = _circular_std(acc_grid[:, 0])
    dip_unc = float(np.std(acc_grid[:, 1]))
    rake_unc = _circular_std(acc_grid[:, 2])

    # Quality grade based on HASH-style thresholds
    if strike_unc < 25 and dip_unc < 15:
        quality = "A"
    elif strike_unc < 35 and dip_unc < 25:
        quality = "B"
    elif strike_unc < 45 and dip_unc < 35:
        quality = "C"
    else:
        quality = "D"

    return {
        "strike_unc": float(strike_unc),
        "dip_unc": float(dip_unc),
        "rake_unc": float(rake_unc),
        "n_acceptable": int(n_acc),
        "quality": quality,
    }


# %%
def _evaluate_grid(gamma, e_sv, e_sh, polarity_obs, log_SP_obs,
                    weights_pol, weights_sp, grid, M_batch,
                    alpha, null_zone_fraction, chunk_size):
    """Evaluate misfit: polarity first to constrain, then S/P ratio to select.

    Two-stage approach:
    1. Compute polarity misfit for all mechanisms
    2. Among mechanisms within the best polarity misfit tolerance, pick the one
       with the lowest S/P ratio misfit (if S/P data available)

    Returns (mf_total, mf_pol, mf_sp) where mf_total reflects this hierarchy.
    """
    P_all, log_SP_pred = _forward_model_batch(M_batch, gamma, e_sv, e_sh, chunk_size)

    n_pol = int(np.sum(polarity_obs != 0))
    n_sp = int(np.sum(~np.isnan(log_SP_obs)))

    mf_pol = polarity_misfit(P_all, polarity_obs, weights_pol, null_zone_fraction)

    if n_sp > 0:
        mf_sp = sp_ratio_misfit(log_SP_pred, log_SP_obs, weights_sp, P_all, null_zone_fraction)
    else:
        mf_sp = np.full(len(grid), np.nan, dtype=np.float32)

    # Two-stage: polarity constrains, S/P ratio selects
    if n_sp > 0 and n_pol > 0:
        # Accept mechanisms within tolerance of best polarity misfit
        best_pol = np.nanmin(mf_pol)
        pol_tolerance = max(1.0 / n_pol, 0.05)  # allow ~1 extra mismatch
        pol_acceptable = mf_pol <= best_pol + pol_tolerance
        # Among polarity-acceptable, rank by S/P ratio misfit
        mf_total = np.full_like(mf_pol, np.inf)
        mf_total[pol_acceptable] = mf_sp[pol_acceptable]
        # Break S/P ties with polarity misfit
        mf_total[pol_acceptable] += mf_pol[pol_acceptable] * 0.01
    elif n_pol > 0:
        mf_total = mf_pol
    elif n_sp > 0:
        sp_filled = np.where(np.isnan(mf_sp), np.nanmax(mf_sp), mf_sp)
        mf_total = sp_filled
    else:
        mf_total = np.full_like(mf_pol, np.nan)

    return mf_total, mf_pol, mf_sp


def grid_search_event(takeoff, azimuth, polarity_obs, log_SP_obs,
                      weights_pol, weights_sp, grid, M_batch,
                      alpha=0.5, null_zone_fraction=0.0,
                      chunk_size=10000, confidence_level=0.1,
                      refine=True, refine_step=2.0, refine_n_best=50):
    """Core grid search for a single event with optional refinement.

    Args:
        takeoff: (N_sta,) takeoff angles in degrees
        azimuth: (N_sta,) azimuths in degrees
        polarity_obs: (N_sta,) +1/-1/0
        log_SP_obs: (N_sta,) log10(S/P) or NaN
        weights_pol: (N_sta,) polarity weights
        weights_sp: (N_sta,) S/P ratio weights
        grid: (N_mech, 3) trial mechanisms
        M_batch: (N_mech, 3, 3) pre-computed moment tensors
        alpha: polarity weight [0, 1]
        null_zone_fraction: nodal-plane exclusion threshold
        chunk_size: mechanisms per chunk
        confidence_level: uncertainty estimation threshold
        refine: whether to do 2nd pass with finer grid around best solutions
        refine_step: grid spacing for refinement (degrees)
        refine_n_best: number of top coarse solutions to refine around

    Returns:
        dict with best mechanism, misfits, and uncertainty
    """
    gamma, e_sv, e_sh = _build_ray_vectors(takeoff, azimuth)

    n_pol = int(np.sum(polarity_obs != 0))
    n_sp = int(np.sum(~np.isnan(log_SP_obs)))

    # Pass 1: coarse grid
    mf_total, mf_pol, mf_sp = _evaluate_grid(
        gamma, e_sv, e_sh, polarity_obs, log_SP_obs,
        weights_pol, weights_sp, grid, M_batch,
        alpha, null_zone_fraction, chunk_size,
    )

    best_idx = np.nanargmin(mf_total)
    best = grid[best_idx]

    # Pass 2: refine around top solutions
    if refine and refine_step > 0:
        # Select top-N coarse solutions
        sorted_idx = np.argsort(mf_total)
        top_idx = sorted_idx[:refine_n_best]
        top_mechs = grid[top_idx]

        # Build fine grid: ±1 coarse step around each top mechanism
        coarse_step = max(grid[1, 0] - grid[0, 0], 5.0) if len(grid) > 1 else 5.0
        fine_points = set()
        for s, d, r in top_mechs:
            for ds in np.arange(-coarse_step, coarse_step + 0.1, refine_step):
                for dd in np.arange(-coarse_step, coarse_step + 0.1, refine_step):
                    for dr in np.arange(-coarse_step, coarse_step + 0.1, refine_step):
                        ns = (s + ds) % 360
                        nd = np.clip(d + dd, 0, 90)
                        nr = ((r + dr) + 180) % 360 - 180
                        fine_points.add((ns, nd, nr))

        fine_grid = np.array(sorted(fine_points))
        fine_M = _build_moment_tensors(fine_grid)

        mf_total_fine, mf_pol_fine, mf_sp_fine = _evaluate_grid(
            gamma, e_sv, e_sh, polarity_obs, log_SP_obs,
            weights_pol, weights_sp, fine_grid, fine_M,
            alpha, null_zone_fraction, chunk_size,
        )

        fine_best_idx = np.nanargmin(mf_total_fine)
        if mf_total_fine[fine_best_idx] <= mf_total[best_idx]:
            best = fine_grid[fine_best_idx]
            # Use fine grid for uncertainty estimation
            unc = estimate_uncertainty(mf_total_fine, fine_grid, confidence_level)
            return {
                "strike": float(best[0]),
                "dip": float(best[1]),
                "rake": float(best[2]),
                "misfit_total": float(mf_total_fine[fine_best_idx]),
                "misfit_pol": float(mf_pol_fine[fine_best_idx]),
                "misfit_sp": float(mf_sp_fine[fine_best_idx]) if n_sp > 0 else np.nan,
                "n_pol": n_pol,
                "n_sp": n_sp,
                **unc,
            }

    unc = estimate_uncertainty(mf_total, grid, confidence_level)

    return {
        "strike": float(best[0]),
        "dip": float(best[1]),
        "rake": float(best[2]),
        "misfit_total": float(mf_total[best_idx]),
        "misfit_pol": float(mf_pol[best_idx]),
        "misfit_sp": float(mf_sp[best_idx]) if n_sp > 0 else np.nan,
        "n_pol": n_pol,
        "n_sp": n_sp,
        **unc,
    }


# %%
def _compute_takeoff_taup_direct(model, depth_km, distance_km):
    """Compute takeoff angle using TauP, preferring the direct upgoing ray.

    For regional distances, TauP's first arrival is often a refracted head wave
    (P) with a constant takeoff angle near the critical angle. For focal
    mechanism inversion, we need the direct ray (p) whose takeoff angle varies
    with distance and properly maps onto the focal sphere.

    Strategy: request both "p" and "P" phases, prefer the upgoing direct "p"
    if available; fall back to "P" only for larger distances where "p" doesn't
    exist (post-critical).
    """
    dist_deg = distance_km / 111.19
    try:
        arrivals = model.get_travel_times(
            source_depth_in_km=max(depth_km, 0.001),
            distance_in_degree=max(dist_deg, 0.001),
            phase_list=("p", "P"),
        )
        if not arrivals:
            return np.nan
        # Prefer direct upgoing "p" phase if available
        for arr in arrivals:
            if arr.phase.name == "p":
                return arr.takeoff_angle
        # Fall back to first arrival (typically refracted P)
        return arrivals[0].takeoff_angle
    except Exception:
        return np.nan


def _compute_takeoff_azimuth(df, takeoff_method="uniform", eikonal_config=None,
                              taup_model=None):
    """Compute takeoff angles and azimuths for all stations of one event.

    Args:
        df: DataFrame rows for a single event with coordinate columns
        takeoff_method: "uniform", "eikonal", or "taup"
        eikonal_config: pre-initialized eikonal dict (for "eikonal" method)
        taup_model: pre-built TauPyModel (for "taup" method)

    Returns:
        (takeoff, azimuth, distance_km) arrays, each shape (N_sta,)
    """
    event_lon = df["event_longitude"].values
    event_lat = df["event_latitude"].values
    event_dep = df["event_depth_km"].values
    station_lon = df["station_longitude"].values
    station_lat = df["station_latitude"].values
    station_dep = df["station_depth_km"].values
    phase_types = np.zeros(len(df), dtype=int)  # P-wave

    if takeoff_method == "uniform":
        ray = calc_ray_param(event_lon, event_lat, event_dep,
                             station_lon, station_lat, station_dep,
                             phase_types, None)
        return ray["takeoff"], ray["azimuth"], ray["distance_km"]

    elif takeoff_method == "eikonal":
        if eikonal_config is None:
            raise ValueError("eikonal_config required for takeoff_method='eikonal'")
        ray = calc_ray_param(event_lon, event_lat, event_dep,
                             station_lon, station_lat, station_dep,
                             phase_types, eikonal_config)
        return ray["takeoff"], ray["azimuth"], ray["distance_km"]

    elif takeoff_method == "taup":
        ray_simple = calc_ray_param(event_lon, event_lat, event_dep,
                                    station_lon, station_lat, station_dep,
                                    phase_types, None)
        if taup_model is None:
            raise ValueError("taup_model required for takeoff_method='taup'")
        depth_km = df["event_depth_km"].iloc[0]
        takeoff = np.full(len(df), np.nan)
        for i, dist in enumerate(ray_simple["distance_km"]):
            takeoff[i] = _compute_takeoff_taup_direct(taup_model, depth_km, dist)
        return takeoff, ray_simple["azimuth"], ray_simple["distance_km"]

    else:
        raise ValueError(f"Unknown takeoff_method: {takeoff_method}")


def prepare_observations(event_df, waveforms=None, min_snr=2.0,
                         p_window_s=0.5, s_window_s=1.0,
                         takeoff_method="uniform", eikonal_config=None,
                         taup_model=None):
    """Extract and validate observations for one event.

    Args:
        event_df: DataFrame rows for a single event (from load_dataset)
        waveforms: dict from load_waveforms (optional, for S/P measurement)
        min_snr: minimum SNR to include a polarity observation
        p_window_s: P window for S/P measurement
        s_window_s: S window for S/P measurement
        takeoff_method: "uniform", "eikonal", or "taup"
        eikonal_config: pre-initialized eikonal config (for "eikonal")
        taup_model: pre-built TauPyModel (for "taup")

    Returns:
        dict with takeoff, azimuth, polarity_obs, log_SP_obs, weights, station info,
        or None if no usable observations.
    """
    df = event_df.copy()

    # Need coordinate columns to compute ray geometry
    required = ["event_longitude", "event_latitude", "event_depth_km",
                "station_longitude", "station_latitude", "station_depth_km"]
    if not all(c in df.columns for c in required):
        return None

    # Filter rows with valid coordinates
    valid_coords = df[required].notna().all(axis=1)
    df = df[valid_coords]
    if len(df) == 0:
        return None

    takeoff, azimuth, distance_km = _compute_takeoff_azimuth(
        df, takeoff_method, eikonal_config, taup_model,
    )

    # Drop stations where takeoff computation failed (NaN)
    valid = np.isfinite(takeoff) & np.isfinite(azimuth)
    if not np.any(valid):
        return None
    df = df.iloc[valid].copy()
    takeoff = takeoff[valid]
    azimuth = azimuth[valid]
    distance_km = distance_km[valid]

    # Encode polarities: U=+1, D=-1, other=0
    polarity_obs = np.zeros(len(df), dtype=np.float32)
    pol_str = df["p_phase_polarity"].fillna("").values
    polarity_obs[pol_str == "U"] = 1.0
    polarity_obs[pol_str == "D"] = -1.0

    # Apply SNR filter: set low-SNR polarities to 0
    if "snr" in df.columns:
        snr = df["snr"].fillna(0).values
        low_snr = snr < min_snr
        polarity_obs[low_snr] = 0.0
        # Weights based on SNR
        weights_pol = np.clip(snr / 10.0, 0.1, 3.0).astype(np.float32)
    else:
        weights_pol = np.ones(len(df), dtype=np.float32)

    # Measure S/P ratios from waveforms
    log_SP_obs = np.full(len(df), np.nan, dtype=np.float32)
    weights_sp = np.ones(len(df), dtype=np.float32)

    if waveforms is not None:
        for i, (_, row) in enumerate(df.iterrows()):
            key = (row["event_id"], row["network"], row["station"],
                   row.get("location", ""), row.get("instrument", ""))
            wf = waveforms.get(key)
            if wf is None:
                continue
            result = measure_sp_ratio(
                wf, row.get("p_phase_index"), row.get("s_phase_index"),
                p_window_s=p_window_s, s_window_s=s_window_s,
            )
            if result is not None:
                log_SP_obs[i] = result["log_sp_obs"]
                weights_sp[i] = np.clip(result["snr_p"] / 5.0, 0.1, 2.0)

    n_pol = int(np.sum(polarity_obs != 0))
    n_sp = int(np.sum(~np.isnan(log_SP_obs)))

    station_ids = [
        f"{row['network']}.{row['station']}"
        for _, row in df.iterrows()
    ]

    return {
        "takeoff": takeoff,
        "azimuth": azimuth,
        "polarity_obs": polarity_obs,
        "log_SP_obs": log_SP_obs,
        "weights_pol": weights_pol,
        "weights_sp": weights_sp,
        "station_ids": station_ids,
        "n_pol": n_pol,
        "n_sp": n_sp,
        "n_stations": len(df),
    }


def invert_focal_mechanism(event_df, waveforms=None, grid=None, M_batch=None,
                           strike_step=5.0, dip_step=5.0, rake_step=5.0,
                           alpha=0.5, null_zone_fraction=0.0, min_n_pol=5,
                           confidence_level=0.1, chunk_size=10000,
                           min_snr=2.0, p_window_s=0.5, s_window_s=1.0,
                           takeoff_method="uniform", eikonal_config=None,
                           taup_model=None):
    """Invert focal mechanism for a single event using grid search.

    Args:
        event_df: DataFrame rows for one event (from load_dataset)
        waveforms: dict from load_waveforms (optional, for S/P ratios)
        grid: pre-built mechanism grid (None -> build from step params)
        M_batch: pre-computed moment tensors matching grid (None -> compute)
        strike_step/dip_step/rake_step: grid spacing in degrees
        alpha: weight of polarity vs S/P (0=S/P only, 1=polarity only)
        null_zone_fraction: HASH nodal-plane exclusion threshold
        min_n_pol: minimum polarity observations required
        confidence_level: misfit threshold for uncertainty estimation
        chunk_size: mechanisms per chunk for memory management
        min_snr: minimum SNR for polarity observations
        p_window_s/s_window_s: windows for S/P measurement
        takeoff_method: "uniform", "eikonal", or "taup"
        eikonal_config: pre-initialized eikonal config (for "eikonal")
        taup_model: pre-built TauPyModel (for "taup")

    Returns:
        dict with event_id, strike, dip, rake, misfits, uncertainty, quality
    """
    event_id = event_df["event_id"].iloc[0]

    obs = prepare_observations(event_df, waveforms, min_snr, p_window_s, s_window_s,
                               takeoff_method, eikonal_config, taup_model)
    if obs is None or obs["n_pol"] < min_n_pol:
        return {
            "event_id": event_id,
            "strike": np.nan, "dip": np.nan, "rake": np.nan,
            "misfit_total": np.nan, "misfit_pol": np.nan, "misfit_sp": np.nan,
            "n_pol": obs["n_pol"] if obs else 0,
            "n_sp": obs["n_sp"] if obs else 0,
            "n_stations": obs["n_stations"] if obs else 0,
            "strike_unc": np.nan, "dip_unc": np.nan, "rake_unc": np.nan,
            "n_acceptable": 0, "quality": "F",
            "status": "INSUFFICIENT_DATA",
        }

    if grid is None:
        grid = build_focal_mechanism_grid(strike_step, dip_step, rake_step)
    if M_batch is None:
        M_batch = _build_moment_tensors(grid)

    result = grid_search_event(
        obs["takeoff"], obs["azimuth"], obs["polarity_obs"], obs["log_SP_obs"],
        obs["weights_pol"], obs["weights_sp"], grid, M_batch,
        alpha=alpha, null_zone_fraction=null_zone_fraction,
        chunk_size=chunk_size, confidence_level=confidence_level,
    )

    result["event_id"] = event_id
    result["n_stations"] = obs["n_stations"]
    result["status"] = "OK"
    return result


# %%
def invert_focal_mechanisms_batch(dataset_path, output_path=None,
                                  waveform_path=None, event_ids=None,
                                  strike_step=5.0, dip_step=5.0, rake_step=5.0,
                                  alpha=0.5, null_zone_fraction=0.0, min_n_pol=5,
                                  confidence_level=0.1, chunk_size=10000,
                                  min_snr=2.0, p_window_s=0.5, s_window_s=1.0,
                                  takeoff_method="uniform"):
    """Batch focal mechanism inversion over all events in a dataset.

    Args:
        dataset_path: path to HDF5 or Parquet dataset
        output_path: CSV output path (None -> auto-derived)
        waveform_path: separate waveform file (None -> same as dataset_path)
        event_ids: subset of event IDs to process (None -> all)
        takeoff_method: "uniform", "eikonal", or "taup"
        **kwargs: passed to invert_focal_mechanism

    Returns:
        DataFrame with one row per event
    """
    print(f"Loading dataset from {dataset_path}")
    phases = load_dataset(dataset_path)
    print(f"Loaded {len(phases)} records from {phases['event_id'].nunique()} events")

    # Load waveforms if S/P ratio is desired (alpha < 1.0)
    waveforms = None
    if alpha < 1.0:
        wf_path = waveform_path or dataset_path
        print(f"Loading waveforms from {wf_path} for S/P ratio measurement...")
        waveforms = load_waveforms(wf_path)
        print(f"Loaded {len(waveforms)} waveforms")

    # Filter to requested events
    if event_ids is not None:
        phases = phases[phases["event_id"].isin(event_ids)]

    # Pre-build grid and moment tensors (shared across events)
    print(f"Building focal mechanism grid (step={strike_step}°)...")
    grid = build_focal_mechanism_grid(strike_step, dip_step, rake_step)
    M_batch = _build_moment_tensors(grid)
    print(f"Grid: {len(grid)} trial mechanisms")

    # Pre-build takeoff model (shared across events)
    eikonal_config = None
    taup_model = None
    if takeoff_method == "eikonal":
        print("Initializing eikonal solver...")
        zz = VELOCITY_MODEL["Z"]
        vp = VELOCITY_MODEL["P"]
        vs = [v / VELOCITY_MODEL["vp_vs_ratio"] for v in vp]
        R_max = phases["distance_km"].max() + 10 if "distance_km" in phases.columns else 500
        Z_max = phases["event_depth_km"].max() + 10 if "event_depth_km" in phases.columns else 50
        config = {
            "vel": {"Z": zz, "P": vp, "S": vs},
            "h": 1.0,
            "xlim_km": [0, R_max],
            "ylim_km": [0, R_max],
            "zlim_km": [0, Z_max],
        }
        eikonal_config = init_eikonal2d(config)
    elif takeoff_method == "taup":
        print("Building TauP velocity model...")
        taup_model = build_taup_velocity_model()

    print(f"Takeoff method: {takeoff_method}")

    results = []
    for event_id, group in tqdm(phases.groupby("event_id"), desc="Inverting"):
        result = invert_focal_mechanism(
            group, waveforms=waveforms, grid=grid, M_batch=M_batch,
            alpha=alpha, null_zone_fraction=null_zone_fraction,
            min_n_pol=min_n_pol, confidence_level=confidence_level,
            chunk_size=chunk_size, min_snr=min_snr,
            p_window_s=p_window_s, s_window_s=s_window_s,
            takeoff_method=takeoff_method, eikonal_config=eikonal_config,
            taup_model=taup_model,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    # Summary
    ok = results_df["status"] == "OK"
    print(f"\nResults: {ok.sum()} succeeded, {(~ok).sum()} insufficient data")
    if ok.any():
        for q in ["A", "B", "C", "D"]:
            n = (results_df.loc[ok, "quality"] == q).sum()
            if n > 0:
                print(f"  Quality {q}: {n}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Save only succeeded events, drop status column
        ok_df = results_df[ok].drop(columns=["status"])
        ok_df.to_csv(output_path, index=False)
        print(f"Saved {len(ok_df)} focal mechanisms to {output_path}")

    return results_df


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Focal mechanism inversion via grid search")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to HDF5 or Parquet dataset file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: auto-derived)")
    parser.add_argument("--waveform_path", type=str, default=None,
                        help="Separate waveform file (default: same as data_path)")
    parser.add_argument("--event_id", type=str, default=None,
                        help="Process only this event ID")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Polarity weight [0,1]: 1=polarity only, 0=S/P only (default: 0.5)")
    parser.add_argument("--strike_step", type=float, default=5.0,
                        help="Strike grid spacing in degrees (default: 5)")
    parser.add_argument("--dip_step", type=float, default=5.0,
                        help="Dip grid spacing in degrees (default: 5)")
    parser.add_argument("--rake_step", type=float, default=5.0,
                        help="Rake grid spacing in degrees (default: 5)")
    parser.add_argument("--null_zone", type=float, default=0.0,
                        help="Null-zone fraction for nodal-plane exclusion (default: 0.0)")
    parser.add_argument("--min_n_pol", type=int, default=5,
                        help="Minimum polarity observations per event (default: 5)")
    parser.add_argument("--min_snr", type=float, default=2.0,
                        help="Minimum SNR for polarity observations (default: 2.0)")
    parser.add_argument("--confidence_level", type=float, default=0.1,
                        help="Misfit threshold for uncertainty estimation (default: 0.1)")
    parser.add_argument("--takeoff_method", type=str, default="uniform",
                        choices=["uniform", "eikonal", "taup"],
                        help="Takeoff angle method: uniform, eikonal, or taup (default: uniform)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    event_ids = [args.event_id] if args.event_id else None

    if args.output:
        output_path = args.output
    else:
        resolved = data_path.resolve()
        parts = resolved.parts
        try:
            ds_idx = parts.index("dataset")
            region = parts[ds_idx - 1]
            sub_path = Path(*parts[ds_idx + 1 :]).parent / data_path.stem
            out_dir = Path(__file__).parent / region / sub_path
        except ValueError:
            out_dir = Path(__file__).parent / data_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "focal_mechanisms.csv")

    invert_focal_mechanisms_batch(
        dataset_path=str(data_path),
        output_path=output_path,
        waveform_path=args.waveform_path,
        event_ids=event_ids,
        strike_step=args.strike_step,
        dip_step=args.dip_step,
        rake_step=args.rake_step,
        alpha=args.alpha,
        null_zone_fraction=args.null_zone,
        min_n_pol=args.min_n_pol,
        min_snr=args.min_snr,
        confidence_level=args.confidence_level,
        takeoff_method=args.takeoff_method,
    )

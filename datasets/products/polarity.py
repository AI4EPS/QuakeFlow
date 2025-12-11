# %%
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eikonal2d import grad_traveltime, init_eikonal2d
from pyproj import Proj


def calc_radiation_pattern_aki(
    strike: float,
    dip: float,
    rake: float,
    takeoff: Union[float, np.ndarray],
    azimuth: Union[float, np.ndarray],
) -> dict:
    """
    Calculate radiation pattern using Aki & Richards formulas.

    Reference: Aki and Richards-2002-p.108-110

    Args:
        strike: Strike angle in degrees (scalar)
        dip: Dip angle in degrees (scalar)
        rake: Rake angle in degrees (scalar)
        takeoff: Take-off angle(s) in degrees (zero points downward)
        azimuth: Azimuth angle(s) in degrees (zero points north)

    Returns:
        Dictionary with keys:
            'P', 'SV', 'SH': Radiation amplitudes
            'P_NED': P-wave displacement vector in NED coordinates (N, 3)
            'S_NED': S-wave displacement vector in NED coordinates (N, 3)
            'log_SP': Log10 of S/P amplitude ratio
    """
    takeoff = np.atleast_1d(np.asarray(takeoff, dtype=float))
    azimuth = np.atleast_1d(np.asarray(azimuth, dtype=float))

    # Convert to radians
    inc = np.deg2rad(takeoff)
    azi = np.deg2rad(azimuth)
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)
    rake_rad = np.deg2rad(rake)

    # Trigonometric terms
    si = np.sin(inc)
    ci = np.cos(inc)
    s2i = np.sin(2 * inc)
    c2i = np.cos(2 * inc)

    sd = np.sin(dip_rad)
    cd = np.cos(dip_rad)
    s2d = np.sin(2 * dip_rad)
    c2d = np.cos(2 * dip_rad)
    sr = np.sin(rake_rad)
    cr = np.cos(rake_rad)
    sas = np.sin(azi - strike_rad)
    cas = np.cos(azi - strike_rad)
    s2as = 2 * sas * cas
    c2as = cas**2 - sas**2

    # P-wave amplitude
    P = (
        -cas * cd * cr * s2i
        + cr * s2as * sd * si**2
        + c2d * s2i * sas * sr
        + s2d * (ci**2 - sas**2 * si**2) * sr
    )

    # SV-wave amplitude
    SV = (
        -c2i * cas * cd * cr
        + 0.5 * cr * s2as * s2i * sd
        + c2d * c2i * sas * sr
        - 0.5 * s2d * s2i * (1 + sas**2) * sr
    )

    # SH-wave amplitude
    SH = (
        cd * ci * cr * sas
        + c2as * cr * sd * si
        + c2d * cas * ci * sr
        - 0.5 * s2as * s2d * si * sr
    )

    # Ray direction (gamma) in NED coordinates
    gamma = np.stack([
        si * np.cos(azi),  # North
        si * np.sin(azi),  # East
        ci                  # Down
    ], axis=-1)

    # SV unit vector (perpendicular to ray in vertical plane)
    e_sv = np.stack([
        ci * np.cos(azi),
        ci * np.sin(azi),
        -si
    ], axis=-1)

    # SH unit vector (perpendicular to ray in horizontal plane)
    e_sh = np.stack([
        -np.sin(azi),
        np.cos(azi),
        np.zeros_like(azi)
    ], axis=-1)

    # P-wave vector: amplitude * ray direction
    P_NED = P[..., np.newaxis] * gamma

    # S-wave vector: SV component + SH component
    S_NED = SV[..., np.newaxis] * e_sv + SH[..., np.newaxis] * e_sh

    # S/P ratio (log10)
    S_total = np.sqrt(SV**2 + SH**2)
    epsilon = 1e-10
    log_SP = np.log10(S_total / (np.abs(P) + epsilon))

    return {
        "P": P, "SV": SV, "SH": SH,
        "P_NED": P_NED, "S_NED": S_NED, "log_SP": log_SP
    }




def calc_radiation_pattern(
    strike: float,
    dip: float,
    rake: float,
    takeoff: Union[float, np.ndarray],
    azimuth: Union[float, np.ndarray],
) -> dict:
    """
    Calculate radiation pattern using moment tensor approach.

    Uses the moment tensor formulation to compute P, SV, SH amplitudes
    via projection onto ray direction and polarization vectors.

    Args:
        strike: Strike angle in degrees (scalar)
        dip: Dip angle in degrees (scalar)
        rake: Rake angle in degrees (scalar)
        takeoff: Take-off angle(s) in degrees (zero points downward)
        azimuth: Azimuth angle(s) in degrees (zero points north)

    Returns:
        Dictionary with keys:
            'P', 'SV', 'SH': Radiation amplitudes
            'P_NED': P-wave displacement vector in NED coordinates (N, 3)
            'S_NED': S-wave displacement vector in NED coordinates (N, 3)
            'log_SP': Log10 of S/P amplitude ratio
    """
    takeoff = np.atleast_1d(np.asarray(takeoff, dtype=float))
    azimuth = np.atleast_1d(np.asarray(azimuth, dtype=float))
    output_shape = takeoff.shape

    # Convert to radians
    d2r = np.pi / 180.0
    s_rad = strike * d2r
    d_rad = dip * d2r
    r_rad = rake * d2r
    inc = takeoff * d2r
    azi = azimuth * d2r

    # Construct fault vectors (Slip d and Normal n) in NED
    # Normal vector (n)
    n_vec = np.array([
        -np.sin(d_rad) * np.sin(s_rad),
         np.sin(d_rad) * np.cos(s_rad),
        -np.cos(d_rad)
    ])

    # Slip vector (d)
    d_vec = np.array([
         np.cos(r_rad) * np.cos(s_rad) + np.cos(d_rad) * np.sin(r_rad) * np.sin(s_rad),
         np.cos(r_rad) * np.sin(s_rad) - np.cos(d_rad) * np.sin(r_rad) * np.cos(s_rad),
        -np.sin(r_rad) * np.sin(d_rad)
    ])

    # Moment tensor M = d*n.T + n*d.T
    M = np.outer(d_vec, n_vec) + np.outer(n_vec, d_vec)

    # Flatten arrays for batch processing
    inc_flat = inc.flatten()
    azi_flat = azi.flatten()

    # Ray direction gamma (propagation vector) - shape (N, 3)
    gamma = np.stack([
        np.sin(inc_flat) * np.cos(azi_flat),  # North
        np.sin(inc_flat) * np.sin(azi_flat),  # East
        np.cos(inc_flat)                       # Down
    ], axis=1)

    # SV unit vector (perpendicular to ray in vertical plane)
    e_sv = np.stack([
        np.cos(inc_flat) * np.cos(azi_flat),
        np.cos(inc_flat) * np.sin(azi_flat),
        -np.sin(inc_flat)
    ], axis=1)

    # SH unit vector (perpendicular to ray in horizontal plane)
    e_sh = np.stack([
        -np.sin(azi_flat),
        np.cos(azi_flat),
        np.zeros_like(azi_flat)
    ], axis=1)

    # Calculate amplitudes using einsum
    # P_amp = gamma^T * M * gamma
    P = np.einsum('ni,ij,nj->n', gamma, M, gamma)
    SH = np.einsum('ni,ij,nj->n', e_sh, M, gamma)
    SV = np.einsum('ni,ij,nj->n', e_sv, M, gamma)

    # P-wave vector: amplitude * ray direction
    P_NED = P[:, np.newaxis] * gamma

    # S-wave vector: SV component + SH component
    S_NED = SV[:, np.newaxis] * e_sv + SH[:, np.newaxis] * e_sh

    # S/P ratio (log10)
    S_total = np.sqrt(SV**2 + SH**2)
    epsilon = 1e-10
    log_SP = np.log10(S_total / (np.abs(P) + epsilon))

    # convert to ENZ
    P_ENZ = P_NED[:, [1, 0, 2]]
    S_ENZ = S_NED[:, [1, 0, 2]]
    P_ENZ[:, 2] *= -1
    S_ENZ[:, 2] *= -1

    return {
        "P": P.reshape(output_shape),
        "SV": SV.reshape(output_shape),
        "SH": SH.reshape(output_shape),
        # "P_NED": P_NED.reshape(output_shape + (3,)),
        # "S_NED": S_NED.reshape(output_shape + (3,)),
        "P_ENZ": P_ENZ.reshape(output_shape + (3,)),
        "S_ENZ": S_ENZ.reshape(output_shape + (3,)),
        "log_SP": log_SP.reshape(output_shape)
    }
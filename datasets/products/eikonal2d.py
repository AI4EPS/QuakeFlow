import time

import numpy as np
from numba import njit

np.random.seed(0)

###################################### Eikonal Solver ######################################

# |\nabla u| = f
# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2


@njit
def calculate_unique_solution(a, b, f, h):
    d = abs(a - b)
    if d >= f * h:
        return min([a, b]) + f * h
    else:
        return (a + b + np.sqrt(2.0 * f * f * h * h - (a - b) ** 2)) / 2.0


@njit
def sweeping_over_I_J_K(u, I, J, f, h):
    m = len(I)
    n = len(J)

    for i in I:
        for j in J:
            if i == 0:
                uxmin = u[i + 1, j]
            elif i == m - 1:
                uxmin = u[i - 1, j]
            else:
                uxmin = min([u[i - 1, j], u[i + 1, j]])

            if j == 0:
                uymin = u[i, j + 1]
            elif j == n - 1:
                uymin = u[i, j - 1]
            else:
                uymin = min([u[i, j - 1], u[i, j + 1]])

            u_new = calculate_unique_solution(uxmin, uymin, f[i, j], h)

            u[i, j] = min([u_new, u[i, j]])

    return u


@njit
def sweeping(u, v, h):
    f = 1.0 / v  ## slowness

    m, n = u.shape
    I = np.arange(m)
    iI = I[::-1]
    J = np.arange(n)
    iJ = J[::-1]

    u = sweeping_over_I_J_K(u, I, J, f, h)
    u = sweeping_over_I_J_K(u, iI, J, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, f, h)

    return u


def eikonal_solve(u, v, h):
    print("Eikonal Solver: ")
    t0 = time.time()
    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, v, h)

        err = np.max(np.abs(u - u_old))
        print(f"Iter {i}, error = {err:.3f}")
        if err < 1e-6:
            break
    print(f"Time: {time.time() - t0:.3f}")
    return u


###################################### Traveltime based on Eikonal Timetable ######################################
@njit
def _get_index(ir, iz, nr, nz, order="C"):
    if order == "C":
        return ir * nz + iz
    elif order == "F":
        return iz * nr + ir
    else:
        raise ValueError("order must be either C or F")


@njit
def _interp(time_table, r, z, rgrid0, zgrid0, nr, nz, h):
    shape = r.shape
    r = r.reshape(-1)
    z = z.reshape(-1)

    ir0 = np.floor((r - rgrid0) / h).clip(0, nr - 2).astype(np.int64)
    iz0 = np.floor((z - zgrid0) / h).clip(0, nz - 2).astype(np.int64)
    ir1 = ir0 + 1
    iz1 = iz0 + 1

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    r0 = ir0 * h + rgrid0
    r1 = ir1 * h + rgrid0
    z0 = iz0 * h + zgrid0
    z1 = iz1 * h + zgrid0

    Q00 = time_table[_get_index(ir0, iz0, nr, nz)]
    Q01 = time_table[_get_index(ir0, iz1, nr, nz)]
    Q10 = time_table[_get_index(ir1, iz0, nr, nz)]
    Q11 = time_table[_get_index(ir1, iz1, nr, nz)]

    t = (
        1.0
        / (r1 - r0)
        / (z1 - z0)
        * (
            Q00 * (r1 - r) * (z1 - z)
            + Q10 * (r - r0) * (z1 - z)
            + Q01 * (r1 - r) * (z - z0)
            + Q11 * (r - r0) * (z - z0)
        )
    )

    return t.reshape(shape)


def grad_traveltime(event_index, station_index, phase_type, events, stations, eikonal):

    if isinstance(event_index, int):
        event_index = np.array([event_index] * len(phase_type))

    x = events[event_index, 0] - stations[station_index, 0]
    y = events[event_index, 1] - stations[station_index, 1]
    z = events[event_index, 2] - stations[station_index, 2]
    r = np.sqrt(x**2 + y**2)

    rgrid0 = eikonal["rgrid"][0]
    zgrid0 = eikonal["zgrid"][0]
    nr = eikonal["nr"]
    nz = eikonal["nz"]
    h = eikonal["h"]

    if isinstance(phase_type, list):
        phase_type = np.array(phase_type)

    dt_dr = np.zeros(len(phase_type))
    dt_dz = np.zeros(len(phase_type))

    p_index = phase_type == 0
    s_index = phase_type == 1

    dt_dr[p_index] = _interp(eikonal["grad_up"][0], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    dt_dr[s_index] = _interp(eikonal["grad_us"][0], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)
    dt_dz[p_index] = _interp(eikonal["grad_up"][1], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    dt_dz[s_index] = _interp(eikonal["grad_us"][1], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

    dt_dx = dt_dr * x / (r + 1e-6)
    dt_dy = dt_dr * y / (r + 1e-6)

    grad = np.column_stack((dt_dx, dt_dy, dt_dz))

    return grad


def calc_ray_param(event_lon, event_lat, event_dep, station_lon, station_lat, station_dep, phase_types, eikonal):
    """Compute ray parameters (takeoff angle, azimuth, distance) for event-station pairs.

    Takeoff angle convention: 0° = straight down, 90° = horizontal, 180° = straight up.
    The eikonal grid u(r, z) has source at (r=0, z=0), axis 0 = r (horizontal),
    axis 1 = z (upward from event toward surface, z = event_dep - station_dep).
    np.gradient(u, h) returns [du/dr, du/dz] where dz points upward.
    To get takeoff from downward vertical, negate dz: takeoff = arctan2(du/dr, -du/dz).

    For the simple (no eikonal) case with uniform velocity:
    r = horizontal distance, z = event_dep - station_dep (positive when event is deeper,
    i.e., station is above event, so z points upward). Since the ray goes upward from
    event to station, takeoff from downward = arctan2(r, -z).
    """
    R = 6371.0  # Earth radius in km

    # Convert degrees to radians
    event_phi, event_lambda = np.radians(event_lat), np.radians(event_lon)
    station_phi, station_lambda = np.radians(station_lat), np.radians(station_lon)
    delta_phi = station_phi - event_phi
    delta_lambda = station_lambda - event_lambda

    # --- 1. Vectorized Haversine Distance ---
    a = np.sin(delta_phi / 2)**2 + \
        np.cos(event_phi) * np.cos(station_phi) * np.sin(delta_lambda / 2)**2

    # np.arctan2 is numerically stable
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    # --- 2. Vectorized Initial Azimuth ---
    y = np.sin(delta_lambda) * np.cos(station_phi)
    x = np.cos(event_phi) * np.sin(station_phi) - \
        np.sin(event_phi) * np.cos(station_phi) * np.cos(delta_lambda)

    theta = np.arctan2(y, x)
    azimuth = (np.degrees(theta) + 360) % 360

    # --- 3. Vectorized Takeoff Angle ---
    r = distance
    z = event_dep - station_dep

    if eikonal is None:
        # Uniform velocity: straight-ray takeoff angle
        # 0° = straight down, 90° = horizontal, 180° = straight up
        # z = event_dep - station_dep points UPWARD (from event toward station),
        # so negate to get the downward component for the takeoff convention.
        takeoff = np.arctan2(r, -z) * 180 / np.pi
        takeoff = (takeoff + 360) % 360

    else:
        rgrid0 = eikonal["rgrid"][0]
        zgrid0 = eikonal["zgrid"][0]
        nr = eikonal["nr"]
        nz = eikonal["nz"]
        h = eikonal["h"]

        dt_dr = np.zeros(len(phase_types))
        dt_dz = np.zeros(len(phase_types))

        if isinstance(phase_types, list):
            phase_types = np.array(phase_types)
        p_index = phase_types == 0
        s_index = phase_types == 1

        dt_dr[p_index] = _interp(eikonal["grad_up"][0], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
        dt_dr[s_index] = _interp(eikonal["grad_us"][0], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)
        dt_dz[p_index] = _interp(eikonal["grad_up"][1], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
        dt_dz[s_index] = _interp(eikonal["grad_us"][1], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

        # Slowness vector (dt_dr, dt_dz) points in direction of wave propagation.
        # dt_dz is along the grid z-axis which points UPWARD (from event toward surface),
        # so negate to get the downward component for the takeoff convention (0° = down).
        takeoff = np.arctan2(dt_dr, -dt_dz) * 180 / np.pi
        takeoff = (takeoff + 360) % 360

    return {"takeoff": takeoff, "azimuth": azimuth, "distance_km": distance}


def init_eikonal2d(config):

    rlim = [
        0,
        np.sqrt(
            (config["xlim_km"][1] - config["xlim_km"][0]) ** 2 + (config["ylim_km"][1] - config["ylim_km"][0]) ** 2
        ),
    ]
    zlim = config["zlim_km"]
    h = config["h"]

    rgrid = np.arange(rlim[0], rlim[1] + h, h)
    zgrid = np.arange(zlim[0], zlim[1] + h, h)
    nr = len(rgrid)
    nz = len(zgrid)

    vel = config["vel"]
    zz, vp, vs = np.array(vel["Z"]), np.array(vel["P"]), np.array(vel["S"])

    vp1d = np.interp(zgrid, zz, vp)
    vs1d = np.interp(zgrid, zz, vs)
    vp = np.tile(vp1d, (nr, 1))
    vs = np.tile(vs1d, (nr, 1))
    ir0 = np.round(0 - rlim[0] / h).astype(np.int64)
    iz0 = np.round(0 - zlim[0] / h).astype(np.int64)
    up = 1000 * np.ones((nr, nz))
    up[ir0, iz0] = 0.0

    up = eikonal_solve(up, vp, h)
    grad_up = np.gradient(up, h, edge_order=2)
    up = up.ravel()
    grad_up = [x.ravel() for x in grad_up]

    us = 1000 * np.ones((nr, nz))
    us[ir0, iz0] = 0.0

    us = eikonal_solve(us, vs, h)
    grad_us = np.gradient(us, h, edge_order=2)
    us = us.ravel()
    grad_us = [x.ravel() for x in grad_us]

    config.update(
        {
            "up": up,
            "us": us,
            "grad_up": grad_up,
            "grad_us": grad_us,
            "rgrid": rgrid,
            "zgrid": zgrid,
            "nr": nr,
            "nz": nz,
            "h": h,
        }
    )

    return config


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt

    data_path = "data"
    os.makedirs(data_path, exist_ok=True)

    # Test with uniform velocity to validate takeoff angle fix
    R = np.sqrt(2) * 100
    Z = 100
    zz = [-100, 100]
    vp = [1000, 1000]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 2.0

    vel = {"Z": zz, "P": vp, "S": vs}
    config = {
        "vel": vel,
        "h": h,
        "xlim_km": [0, R],
        "ylim_km": [0, R],
        "zlim_km": [-Z, Z],
    }
    config = init_eikonal2d(config)
    nr = config["nr"]
    nz = config["nz"]

    # --- Validate takeoff: simple vs eikonal should match for uniform velocity ---
    print("\n=== Takeoff angle validation (uniform velocity) ===")
    # Events at varying depths, station at surface
    n_test = 20
    ez = np.linspace(1, 80, n_test)
    event_lon = np.full(n_test, -122.0)
    event_lat = np.full(n_test, 37.0)
    event_dep = ez  # depth in km
    # Station ~100km away at surface
    station_lon = np.full(n_test, -121.0)
    station_lat = np.full(n_test, 37.0)
    station_dep = np.zeros(n_test)
    phase_types = np.zeros(n_test, dtype=int)  # P wave

    ray_simple = calc_ray_param(event_lon, event_lat, event_dep,
                                station_lon, station_lat, station_dep,
                                phase_types, None)
    ray_eikonal = calc_ray_param(event_lon, event_lat, event_dep,
                                 station_lon, station_lat, station_dep,
                                 phase_types, config)

    print(f"{'Depth':>8} {'Simple':>10} {'Eikonal':>10} {'Diff':>10}")
    for i in range(n_test):
        diff = ray_simple["takeoff"][i] - ray_eikonal["takeoff"][i]
        print(f"{ez[i]:8.1f} {ray_simple['takeoff'][i]:10.2f} {ray_eikonal['takeoff'][i]:10.2f} {diff:10.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(ez, ray_simple["takeoff"], "b-o", label="Simple (uniform)")
    ax[0].plot(ez, ray_eikonal["takeoff"], "r--x", label="Eikonal (uniform)")
    ax[0].set_xlabel("Event Depth (km)")
    ax[0].set_ylabel("Takeoff Angle (deg)")
    ax[0].set_title("Takeoff Angle: Simple vs Eikonal")
    ax[0].legend()
    ax[0].axhline(90, color="gray", ls="--", alpha=0.5)

    ax[1].plot(ez, ray_simple["takeoff"] - ray_eikonal["takeoff"], "k-o")
    ax[1].set_xlabel("Event Depth (km)")
    ax[1].set_ylabel("Difference (deg)")
    ax[1].set_title("Simple - Eikonal (should be ~0 for uniform)")
    ax[1].axhline(0, color="gray", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{data_path}/takeoff_validation.png", dpi=150)
    print(f"\nSaved {data_path}/takeoff_validation.png")

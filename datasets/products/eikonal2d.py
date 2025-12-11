import itertools
import shelve
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from numba import njit
from numba.typed import List

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


def test_get_index():
    vr, vz = np.meshgrid(np.arange(10), np.arange(20), indexing="ij")
    vr = vr.ravel()
    vz = vz.ravel()
    nr = 10
    nz = 20
    for ir in range(nr):
        for iz in range(nz):
            assert vr[_get_index(ir, iz, nr, nz)] == ir
            assert vz[_get_index(ir, iz, nr, nz)] == iz


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


# def traveltime(event_loc, station_loc, phase_type, eikonal):
def traveltime(event_index, station_index, phase_type, events, stations, eikonal, vel={0: 6.0, 1: 6.0 / 1.73}):
    """
    event_index: list of event index
    station_index: list of station index
    phase_type: list of phase type
    events: list of event location
    stations: list of station location
    """
    if eikonal is None:
        v = np.array([vel[x] for x in phase_type])
        tt = np.linalg.norm(events[event_index] - stations[station_index], axis=-1, keepdims=False) / v
    else:
        if isinstance(event_index, int):
            event_index = np.array([event_index] * len(phase_type))

        # r = np.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)
        # z = event_loc[:, 2] - station_loc[:, 2]
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
        # if isinstance(station_index, list):
        #     station_index = np.array(station_index)

        tt = np.zeros(len(phase_type), dtype=np.float32)

        if isinstance(phase_type[0], str):
            p_index = phase_type == "P"
            s_index = phase_type == "S"
        elif isinstance(phase_type[0].item(), int):
            p_index = phase_type == 0
            s_index = phase_type == 1
        else:
            raise ValueError("phase_type must be either P/S or 0/1")

        if len(tt[p_index]) > 0:
            tt[p_index] = _interp(eikonal["up"], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
        if len(tt[s_index]) > 0:
            tt[s_index] = _interp(eikonal["us"], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

    return tt


def calc_traveltime(event_locs, station_locs, phase_type, eikonal):
    """
    event_locs: (num_event, 3) array of event locations
    station_locs: (num_station, 3) array of station locations
    phase_type: (num_event,) array of phase type
    eikonal: dictionary of eikonal solver
    """

    if eikonal is None:
        v = np.array([vel[x] for x in phase_type])
        tt = np.linalg.norm(event_locs - station_locs, axis=-1, keepdims=False) / v
    else:
        x = event_locs[:, 0] - station_locs[:, 0]
        y = event_locs[:, 1] - station_locs[:, 1]
        z = event_locs[:, 2] - station_locs[:, 2]
        r = np.sqrt(x**2 + y**2)

        rgrid0 = eikonal["rgrid"][0]
        zgrid0 = eikonal["zgrid"][0]
        nr = eikonal["nr"]
        nz = eikonal["nz"]
        h = eikonal["h"]

        if isinstance(phase_type, list):
            phase_type = np.array(phase_type)

        tt = np.zeros(len(phase_type), dtype=np.float32)
        if isinstance(phase_type[0], str):
            p_index = phase_type == "P"
            s_index = phase_type == "S"
        elif isinstance(phase_type[0].item(), int):
            p_index = phase_type == 0
            s_index = phase_type == 1
        else:
            raise ValueError("phase_type must be either P/S or 0/1")

        if len(tt[p_index]) > 0:
            tt[p_index] = _interp(eikonal["up"], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
        if len(tt[s_index]) > 0:
            tt[s_index] = _interp(eikonal["us"], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

    return tt


# def grad_traveltime(event_loc, station_loc, phase_type, eikonal):
def grad_traveltime(event_index, station_index, phase_type, events, stations, eikonal):

    if isinstance(event_index, int):
        event_index = np.array([event_index] * len(phase_type))

    # r = np.linalg.norm(event_loc[:, :2] - station_loc[:, :2], axis=-1, keepdims=False)
    # z = event_loc[:, 2] - station_loc[:, 2]
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
    # if isinstance(station_index, list):
    #     station_index = np.array(station_index)

    dt_dr = np.zeros(len(phase_type))
    dt_dz = np.zeros(len(phase_type))

    # p_index = phase_type == "p"
    # s_index = phase_type == "s"
    p_index = phase_type == 0
    s_index = phase_type == 1

    dt_dr[p_index] = _interp(eikonal["grad_up"][0], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    dt_dr[s_index] = _interp(eikonal["grad_us"][0], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)
    dt_dz[p_index] = _interp(eikonal["grad_up"][1], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
    dt_dz[s_index] = _interp(eikonal["grad_us"][1], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

    # dr_dxy = (event_loc[:, :2] - station_loc[:, :2]) / (r[:, np.newaxis] + 1e-6)
    # dt_dxy = dt_dr[:, np.newaxis] * dr_dxy
    # grad = np.column_stack((dt_dxy, dt_dz[:, np.newaxis]))
    dt_dx = dt_dr * x / (r + 1e-6)
    dt_dy = dt_dr * y / (r + 1e-6)

    grad = np.column_stack((dt_dx, dt_dy, dt_dz))

    return grad


def calc_ray_angle(event_locs, station_locs, phase_types, eikonal):

    x = event_locs[:, 0] - station_locs[:, 0]
    y = event_locs[:, 1] - station_locs[:, 1]
    z = event_locs[:, 2] - station_locs[:, 2] 
    r = np.sqrt(x**2 + y**2)

    azimuth = np.arctan2(x, y) * 180 / np.pi # tan = x/y
    azimuth = (azimuth + 360) % 360

    if eikonal is None:
        takeoff = np.arctan2(r, -z) * 180 / np.pi # tan = r / z
        takeoff = (takeoff + 360) % 360

    else:
        rgrid0 = eikonal["rgrid"][0]
        zgrid0 = eikonal["zgrid"][0]
        nr = eikonal["nr"]
        nz = eikonal["nz"]
        h = eikonal["h"]

        dt_dr = np.zeros(len(phase_types))
        dt_dz = np.zeros(len(phase_types))

        # p_index = phase_type == "p"
        # s_index = phase_type == "s"
        if isinstance(phase_types, list):
            phase_types = np.array(phase_types)
        p_index = phase_types == 0
        s_index = phase_types == 1

        dt_dr[p_index] = _interp(eikonal["grad_up"][0], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
        dt_dr[s_index] = _interp(eikonal["grad_us"][0], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)
        dt_dz[p_index] = _interp(eikonal["grad_up"][1], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
        dt_dz[s_index] = _interp(eikonal["grad_us"][1], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

        ## TODO: double check if -dt_dz or dt_dz
        takeoff = np.arctan2(dt_dr, -dt_dz) * 180 / np.pi # tan = dr / -dz
        takeoff = (takeoff + 360) % 360

    return {"takeoff": takeoff, "azimuth": azimuth}

def calc_ray_param(event_lon, event_lat, event_dep, station_lon, station_lat, station_dep, phase_types, eikonal):

    R = 6371.0  # Earth radius in km

    # Convert degrees to radians
    print(f"{event_lat.shape = }")
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
        takeoff = np.arctan2(r, -z) * 180 / np.pi # tan = r / z
        takeoff = (takeoff + 360) % 360

    else:
        rgrid0 = eikonal["rgrid"][0]
        zgrid0 = eikonal["zgrid"][0]
        nr = eikonal["nr"]
        nz = eikonal["nz"]
        h = eikonal["h"]

        dt_dr = np.zeros(len(phase_types))
        dt_dz = np.zeros(len(phase_types))

        # p_index = phase_type == "p"
        # s_index = phase_type == "s"
        if isinstance(phase_types, list):
            phase_types = np.array(phase_types)
        p_index = phase_types == 0
        s_index = phase_types == 1

        dt_dr[p_index] = _interp(eikonal["grad_up"][0], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
        dt_dr[s_index] = _interp(eikonal["grad_us"][0], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)
        dt_dz[p_index] = _interp(eikonal["grad_up"][1], r[p_index], z[p_index], rgrid0, zgrid0, nr, nz, h)
        dt_dz[s_index] = _interp(eikonal["grad_us"][1], r[s_index], z[s_index], rgrid0, zgrid0, nr, nz, h)

        ## TODO: double check if -dt_dz or dt_dz
        takeoff = np.arctan2(dt_dr, -dt_dz) * 180 / np.pi # tan = dr / -dz
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

    # ##############################################
    # ## make the velocity staircase not linear
    # zz_grid = zz[1:] - h
    # vp_grid = vp[:-1]
    # vs_grid = vs[:-1]
    # zz = np.concatenate([zz, zz_grid])
    # vp = np.concatenate([vp, vp_grid])
    # vs = np.concatenate([vs, vs_grid])
    # idx = np.argsort(zz)
    # zz = zz[idx]
    # vp = vp[idx]
    # vs = vs[idx]
    # ##############################################

    vp1d = np.interp(zgrid, zz, vp)
    vs1d = np.interp(zgrid, zz, vs)
    vp = np.tile(vp1d, (nr, 1))
    vs = np.tile(vs1d, (nr, 1))
    # ir0 = np.floor(config["source_loc"][0] / h).astype(np.int64)
    # iz0 = np.floor(config["source_loc"][1] / h).astype(np.int64)
    ir0 = np.round(0 - rlim[0] / h).astype(np.int64)
    iz0 = np.round(0 - zlim[0] / h).astype(np.int64)
    up = 1000 * np.ones((nr, nz))
    # up[0, 0] = 0.0
    up[ir0, iz0] = 0.0

    up = eikonal_solve(up, vp, h)
    grad_up = np.gradient(up, h, edge_order=2)
    up = up.ravel()
    grad_up = [x.ravel() for x in grad_up]

    us = 1000 * np.ones((nr, nz))
    # us[0, 0] = 0.0
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

    import matplotlib as mpl

    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    # nr = 21
    # nz = 21
    # vel = {"p": 6.0, "s": 6.0 / 1.73}
    # vp = np.ones((nr, nz)) * vel["p"]
    # vs = np.ones((nr, nz)) * vel["s"]
    # h = 1.0

    # up = 1000 * np.ones((nr, nz))
    # # up[nr//2, nz//2] = 0.0
    # up[0, 0] = 0.0

    # up = eikonal_solve(up, vp, h)
    # grad_up = np.gradient(up, h, edge_order=2)
    # up = up.ravel()
    # grad_up = [x.ravel() for x in grad_up]

    # us = 1000 * np.ones((nr, nz))
    # # us[nr//2, nz//2] = 0.0
    # us[0, 0] = 0.0

    # us = eikonal_solve(us, vs, h)
    # grad_us = np.gradient(us, h, edge_order=2)
    # us = us.ravel()
    # grad_us = [x.ravel() for x in grad_us]

    # config = {
    #         "up": up,
    #         "us": us,
    #         "grad_up": grad_up,
    #         "grad_us": grad_us,
    #         "rgrid": np.arange(nr) * h,
    #         "zgrid": np.arange(nz) * h,
    #         "nr": nr,
    #         "nz": nz,
    #         "h": h,
    # }

    # %%
    R = np.sqrt(2) * 100
    Z = 100
    # zz = [0.0, 5.5, 16.0, 32.0]
    # # vp = [5.5, 5.5, 6.7, 7.8]
    # vp = [5.5, 5.5, 5.5, 5.5]
    zz = [-100, 100]
    vp = [1000, 1000]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 2.0

    vel = {"Z": zz, "P": vp, "S": vs}
    vel0 = {"P": 1000, "S": 1000/vp_vs_ratio}
    config = {
        "vel": vel,
        "h": h,
        "xlim_km": [0, R],
        "ylim_km": [0, R], 
        "zlim_km": [-Z, Z]
    }
    config = init_eikonal2d(config)
    up = config["up"]
    us = config["us"]
    grad_up = config["grad_up"]
    grad_us = config["grad_us"]
    nr = config["nr"]
    nz = config["nz"]

    num_event = 10
    event_loc = np.random.rand(num_event, 3) * np.array([nr * h / np.sqrt(2), nr * h / np.sqrt(2), nz * h])
    event_index = np.arange(num_event)
    print(f"{event_loc = }")
    # event_loc = np.round(event_loc, 0)
    # station_loc = np.random.rand(1, 3) * np.array([nr*h/np.sqrt(2), nr*h/np.sqrt(2), 0])
    station_loc = np.array([0, 0, 0])
    print(f"{station_loc = }")
    station_loc = np.tile(station_loc, (num_event, 1))
    station_index = [0] * num_event
    # phase_type = np.random.choice(["p", "s"], num_event, replace=True)
    # print(f"{list(phase_type) = }")
    phase_type = np.array(["P"] * (num_event // 2) + ["S"] * (num_event - num_event // 2))
    v = np.array([vel0[x] for x in phase_type])
    t = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / v
    grad_t = (
        (event_loc - station_loc) / np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=True) / v[:, np.newaxis]
    )
    print(f"True traveltime: {t = }")
    print(f"True grad traveltime: {grad_t = }")

    # tp = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / vel["p"]
    # print(f"{tp = }")
    # ts = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=False) / vel["s"]
    # print(f"{ts = }")

    mapping_int = {"P": 0, "S": 1}
    phase_type = np.array([mapping_int[x] for x in phase_type])
    # t = traveltime(event_loc, station_loc, phase_type, config)
    # grad_t = grad_traveltime(event_loc, station_loc, phase_type, config)
    t = traveltime(event_index, station_index, phase_type, event_loc, station_loc, config)
    grad_t = grad_traveltime(event_index, station_index, phase_type, event_loc, station_loc, config)
    print(f"Computed traveltime: {t = }")
    print(f"Computed grad traveltime: {grad_t = }")

    up = up.reshape((nr, nz))
    plt.figure()
    plt.pcolormesh(up[:, :].T)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(f"{data_path}/slice_tp_2d.png")

    us = us.reshape((nr, nz))
    plt.figure()
    plt.pcolormesh(us[:, :].T)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(f"{data_path}/slice_ts_2d.png")

    grad_up = [x.reshape((nr, nz)) for x in grad_up]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cax0 = ax[0].pcolormesh(grad_up[0][:, :].T)
    fig.colorbar(cax0, ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title("grad_tp_x")
    cax1 = ax[1].pcolormesh(grad_up[1][:, :].T)
    fig.colorbar(cax1, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title("grad_tp_z")
    plt.savefig(f"{data_path}/slice_grad_tp_2d.png")

    grad_us = [x.reshape((nr, nz)) for x in grad_us]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cax0 = ax[0].pcolormesh(grad_us[0][:, :].T)
    fig.colorbar(cax0, ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title("grad_ts_x")
    cax1 = ax[1].pcolormesh(grad_us[1][:, :].T)
    fig.colorbar(cax1, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_title("grad_ts_z")
    plt.savefig(f"{data_path}/slice_grad_ts_2d.png")

    ## check ray angle
    R = 100
    theta = np.arange(0, 360, 10)
    sx = R * np.cos(theta * np.pi / 180)
    sy = R * np.sin(theta * np.pi / 180)
    sz = np.zeros_like(sx)
    ez = np.linspace(-100, 100, 20)
    ex = np.zeros_like(ez)
    ey = np.zeros_like(ez)

    event_loc = np.column_stack((ex, ey, ez))
    station_loc = np.column_stack((sx, sy, sz))

    dummy_loc = np.column_stack((np.zeros_like(sx), np.zeros_like(sx), np.zeros_like(sx)))
    phase_type = np.ones_like(sx) * 0  # P wave
    ray_angle0 = calc_ray_angle(dummy_loc, station_loc, phase_type, None)
    azimuth0 = ray_angle0["azimuth"]
    ray_angle = calc_ray_angle(dummy_loc, station_loc, phase_type, eikonal=config)
    azimuth = ray_angle["azimuth"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im0 = ax[0].scatter(sx, sy, c=azimuth0, cmap="hsv", s=100, vmin=0, vmax=360)
    fig.colorbar(im0, ax=ax[0], label="Azimuth (deg)", fraction=0.046, pad=0.04)
    ax[0].set_title("Azimuth (no eikonal)")
    ax[0].set_xlabel("X (km)")
    ax[0].set_ylabel("Y (km)")
    ax[0].axis("equal")

    im1 = ax[1].scatter(sx, sy, c=azimuth, cmap="hsv", s=100, vmin=0, vmax=360)
    fig.colorbar(im1, ax=ax[1], label="Azimuth (deg)", fraction=0.046, pad=0.04)
    ax[1].set_title("Azimuth (with eikonal)")
    ax[1].set_xlabel("X (km)")
    ax[1].set_ylabel("Y (km)")
    ax[1].axis("equal")

    plt.tight_layout()
    plt.savefig(f"{data_path}/ray_azimuth_2d.png")

    
    dummy_loc = np.column_stack((np.ones_like(ez)*R, np.zeros_like(ez), np.zeros_like(ez)))
    phase_type = np.ones_like(ez) * 0  # P wave
    ray_angle0 = calc_ray_angle(event_loc, dummy_loc, phase_type, None)
    takeoff0 = ray_angle0["takeoff"]
    ray_angle = calc_ray_angle(event_loc, dummy_loc, phase_type, eikonal=config)
    takeoff = ray_angle["takeoff"]

    print(f"{ez = }")
    print(f"{takeoff0 = }")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im0 = ax[0].scatter(np.zeros_like(ez), ez, c=takeoff0, cmap="viridis", s=100, vmin=45, vmax=180-45)
    fig.colorbar(im0, ax=ax[0], label="Takeoff Angle (deg)", fraction=0.046, pad=0.04)
    ax[0].set_title("Takeoff Angle (no eikonal)")
    ax[0].invert_yaxis()
    ax[0].set_xlabel("R (km)")
    ax[0].set_ylabel("Z (km)")

    im1 = ax[1].scatter(np.zeros_like(ez), ez, c=takeoff, cmap="viridis", s=100, vmin=45, vmax=180-45)
    fig.colorbar(im1, ax=ax[1], label="Takeoff Angle (deg)", fraction=0.046, pad=0.04)
    ax[1].set_title("Takeoff Angle (with eikonal)")
    ax[1].invert_yaxis()
    ax[1].set_xlabel("R (km)")
    ax[1].set_ylabel("Z (km)")

    plt.tight_layout()
    plt.savefig(f"{data_path}/ray_takeoff_2d.png")

    





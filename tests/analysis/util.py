import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from collections import defaultdict
import fire


timestamp = lambda dt: (dt - datetime(2019, 1, 1)).total_seconds()

## ridgecrest
class Config:
    degree2km = np.pi * 6371 / 180
    center = (35.705, -117.504)
    horizontal = 0.5
    vertical = 0.5


def load_eqnet_catalog(fname, config=Config()):

    catalog = pd.read_csv(fname, sep="\t", parse_dates=['time'])
    catalog["date"] = catalog["time"]
    catalog["X"] = catalog["x(km)"]
    catalog["Y"] = catalog["y(km)"]
    catalog["Z"] = catalog["z(km)"]
    catalog["time"] = catalog["date"]
    catalog["magnitude"] = 0.0
    catalog["longitude"] = catalog["X"] / config.degree2km + (config.center[1] - config.horizontal)
    catalog["latitude"] = catalog["Y"] /  config.degree2km + (config.center[0] - config.vertical)
    catalog["depth(m)"] = catalog["Z"] * 1e3
    return catalog


def load_scsn(config=Config()):

    if not os.path.exists("2019.catalog"):
        os.system("wget https://raw.githubusercontent.com/SCEDC/SCEDC-catalogs/master/SCSN/2019.catalog")

    catalog = defaultdict(list)
    with open("2019.catalog", 'r') as fp:
        for line in fp:
            if line[0] in ['#', '\n', '\r\n']:
                continue
            catalog["YYY"].append(line[0:4].strip())
            catalog["MM"].append(line[4:7].strip())
            catalog["DD"].append(line[7:10].strip())
            catalog["HH"].append(line[10:14].strip())
            catalog["mm"].append(line[14:17].strip())
            catalog["SS.ss"].append(line[17:23].strip())
            catalog["LAT-deg"].append(line[23:27].strip())
            catalog["LAT-sec"].append(line[27:33].strip())
            catalog["LON-deg"].append(line[33:37].strip())
            catalog["LON-sec"].append(line[37:43].strip())
            catalog["Q"].append(line[43:45].strip())
            catalog["MAG"].append(line[45:49].strip())
            catalog["DEPTH"].append(line[49:59].strip())
            catalog["NPH"].append(line[59:62].strip())
            catalog["RMS"].append(line[62:71].strip())
            catalog["EVID"].append(line[71:80].strip())

    catalog = pd.DataFrame.from_dict(catalog)
    catalog["LON"] = -(-catalog["LON-deg"].astype('float') + catalog["LON-sec"].astype('float') / 60)
    catalog["LAT"] = catalog["LAT-deg"].astype('float').abs() + catalog["LAT-sec"].astype('float') / 60
    catalog['DEPTH'] = catalog['DEPTH'].astype('float')

    catalog["date"] = (
        catalog["YYY"]
        + "-"
        + catalog["MM"]
        + "-"
        + catalog["DD"]
        + "T"
        + catalog["HH"]
        + ":"
        + catalog["mm"]
        + ":"
        + catalog["SS.ss"]
        + "0"
    )
    catalog["date"] = catalog["date"].map(datetime.fromisoformat)
    catalog["X"] = (catalog["LON"].map(float) - (config.center[1] - config.horizontal)) * config.degree2km
    catalog["Y"] = (catalog["LAT"].map(float) - (config.center[0] - config.vertical)) * config.degree2km
    catalog["Z"] = catalog['DEPTH'].map(float)
    catalog["mag"] = catalog["MAG"].map(float)
    catalog["time"] = catalog["date"]
    catalog["magnitude"] = catalog["mag"]
    catalog["latitude"] = catalog["LAT"]
    catalog["longitude"] = catalog["LON"]
    catalog["depth(m)"] = catalog["Z"]*1e3
 
    return catalog


def load_Ross2019(config=Config()):

    if not os.path.exists("Ross2019.txt"):
        os.system("wget https://service.scedc.caltech.edu/ftp/QTMcatalog-ridgecrest/ridgecrest_qtm.tar.gz")
        os.system("tar -xzf ridgecrest_qtm.tar.gz")
        os.system("rm ridgecrest_qtm.tar.gz")
        os.system("mv ridgecrest_qtm.cat Ross2019.txt")

    catalog = pd.read_csv(
        "Ross2019.txt",
        sep='\s+',
        header=0,
        names=[
            "yr",
            "mon",
            "day",
            "hr",
            "min",
            "sec",
            "eID",
            "latR",
            "lonR",
            "depR",
            "mag",
            "qID",
            "cID",
            "nbranch",
            "qnpair",
            "qndiffP",
            "qndiffS",
            "rmsP",
            "rmsS",
            "eh",
            "ez",
            "et",
            "latC",
            "lonC",
            "depC",
        ],
        dtype={
            "yr": int,
            "mon": int,
            "day": int,
            "hr": int,
            "min": int,
            "sec": float,
            "eID": int,
            "latR": float,
            "lonR": float,
            "depR": float,
            "mag": float,
        },
    )

    catalog["date"] = (
        catalog["yr"].map("{:04d}".format)
        + "-"
        + catalog["mon"].map("{:02d}".format)
        + "-"
        + catalog["day"].map("{:02d}".format)
        + "T"
        + catalog["hr"].map("{:02d}".format)
        + ":"
        + catalog["min"].map("{:02d}".format)
        + ":"
        + catalog["sec"].map("{:06.3f}".format)
    )
    catalog["date"] = catalog["date"].map(datetime.fromisoformat)
    catalog["X"] = (catalog["lonR"] - (config.center[1] - config.horizontal)) * config.degree2km
    catalog["Y"] = (catalog["latR"] - (config.center[0] - config.vertical)) * config.degree2km
    catalog["Z"] = catalog['depR']
    catalog["time"] = catalog["date"]
    catalog["magnitude"] = catalog["mag"]
    catalog["latitude"] = catalog["latR"]
    catalog["longitude"] = catalog["lonR"]

    return catalog


def load_Shelly2020(config=Config()):
    if not os.path.exists("Shelly2020.txt"):
        os.system(
            "wget -O Shelly2020.txt 'https://gsw.silverchair-cdn.com/gsw/Content_public/Journal/srl/91/4/10.1785_0220190309/3/srl-2019309_supplement_hypos_ridgecrest_srl_header_mnew.txt?Expires=1631293840&Signature=jEK7KhqbSy-Tx7pwSGpIujTLKUKHis3SVkyKFabHAFAnF4PilKnv3oD4u8ng9HWHUYGBm68efxgAFZibRffNMICHNG6XEGQ~tg71kxL1pvc45rBC~AeByfrN85Dsw4EZB3mC79Od0V6uPlNFOQnuA00cWfVxzzX~mjUMzWSuJclBjoXb8Ux4PXvu28qNUu8xexXQR5HCGgh5toxIluVcqanHZCbhvyur~CVykRAjeS~sq6tD9S8hBBOIXjxwi5nPJrL4punHvbp36Q4z4IWaBjHscT7voC~I2E4TpFJ24FlPefdWJBCNnBFP1EGvJmJrGpy1pJxGAodbf1mRG6diBw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA'"
        )

    catalog = pd.read_csv(
        "Shelly2020.txt",
        sep='\s+',
        header=24,
        names=["yr", "mon", "day", "hr", "min", "sec", "lat", "lon", "dep", "x", "y", "z", "mag", "ID"],
        dtype=str,
    )

    catalog["date"] = (
        catalog["yr"]
        + "-"
        + catalog["mon"]
        + "-"
        + catalog["day"]
        + "T"
        + catalog["hr"]
        + ":"
        + catalog["min"]
        + ":"
        + catalog["sec"]
    )
    catalog["date"] = catalog["date"].map(datetime.fromisoformat)
    catalog["X"] = (catalog["lon"].map(float) - (config.center[1] - config.horizontal)) * config.degree2km
    catalog["Y"] = (catalog["lat"].map(float) - (config.center[0] - config.vertical)) * config.degree2km
    catalog["Z"] = catalog['dep'].map(float)
    catalog["mag"] = catalog["mag"].map(float)
    catalog["time"] = catalog["date"]
    catalog["magnitude"] = catalog["mag"]
    catalog["latitude"] = catalog["lat"]
    catalog["longitude"] = catalog["lon"]

    return catalog


def load_Liu2020(config=Config()):
    if not os.path.exists("Liu2020.txt"):
        os.system(
            "wget -O Liu2020.txt https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement\?doi\=10.1029%2F2019GL086189\&file\=grl60250-sup-0002-2019GL086189-ts01.txt"
        )

    catalog = pd.read_csv(
        "Liu2020.txt",
        sep='\s+',
        header=1,
        names=["yr", "mon", "day", "hr", "min", "sec", "lat", "lon", "dep", "mag"],
        dtype={
            "yr": int,
            "mon": int,
            "day": int,
            "hr": int,
            "min": int,
            "sec": float,
            "lat": float,
            "lon": float,
            "dep": float,
            "mag": float,
        },
    )

    catalog["date"] = (
        catalog["yr"].map("{:04d}".format)
        + "-"
        + catalog["mon"].map("{:02d}".format)
        + "-"
        + catalog["day"].map("{:02d}".format)
        + "T"
        + catalog["hr"].map("{:02d}".format)
        + ":"
        + catalog["min"].map("{:02d}".format)
        + ":"
        + catalog["sec"].map("{:06.3f}".format)
    )
    catalog["date"] = catalog["date"].map(datetime.fromisoformat)
    catalog["X"] = (catalog["lon"] - (config.center[1] - config.horizontal)) * config.degree2km
    catalog["Y"] = (catalog["lat"] - (config.center[0] - config.vertical)) * config.degree2km
    catalog["Z"] = catalog['dep']
    catalog["time"] = catalog["date"]
    catalog["magnitude"] = catalog["mag"]
    catalog["latitude"] = catalog["lat"]
    catalog["longitude"] = catalog["lon"]

    return catalog


def load_GaMMA_catalog(fname, config=Config()):

    catalog = pd.read_csv(fname, sep='\t',)

    catalog["date"] = catalog["time"].map(datetime.fromisoformat)
    catalog["X"] = (catalog["longitude"].map(float) - (config.center[1] - config.horizontal)) * config.degree2km
    catalog["Y"] = (catalog["latitude"].map(float) - (config.center[0] - config.vertical)) * config.degree2km
    catalog["Z"] = catalog['depth(m)'].map(float)
    catalog["mag"] = catalog["magnitude"]

    return catalog


def filter_catalog(catalog, start_datetime, end_datetime, xmin, xmax, ymin, ymax, config=Config()):

    selected_catalog = catalog[
        (catalog["date"] >= start_datetime)
        & (catalog["date"] <= end_datetime)
        & (catalog['X'] >= xmin)
        & (catalog['X'] <= xmax)
        & (catalog['Y'] >= ymin)
        & (catalog['Y'] <= ymax)
    ]
    print(f"Filtered catalog {start_datetime}-{end_datetime}: {len(selected_catalog)} events")

    t_event = []
    xyz_event = []
    mag_event = []
    for _, row in selected_catalog.iterrows():
        t_event.append(timestamp(row["date"]))
        xyz_event.append([row['X'], row['Y'], row['Z']])
        if "mag" in row:
            mag_event.append(row["mag"])
    t_event = np.array(t_event)
    xyz_event = np.array(xyz_event)
    mag_event = np.array(mag_event)

    return t_event, xyz_event, mag_event, selected_catalog


def calc_detection_performance(t_pred, t_true, time_accuracy_threshold=3):
    # time_accuracy_threshold = 3 #s
    evaluation_matrix = np.abs(t_pred[np.newaxis, :] - t_true[:, np.newaxis]) < time_accuracy_threshold  # s
    recalls = np.sum(evaluation_matrix, axis=1) > 0
    num_recall = np.sum(recalls)
    num_precision = np.sum(np.sum(evaluation_matrix, axis=0) > 0)
    if (len(t_true) > 0) and (len(t_pred) > 0):
        recall = num_recall / len(t_true)
        precision = num_precision / len(t_pred)
        f1 = 2 * recall * precision / (recall + precision)
    return recall, precision, f1


def calc_time_loc_error(t_pred, xyz_pred, t_true, xyz_true, time_accuracy_threshold):

    evaluation_matrix = np.abs(t_pred[np.newaxis, :] - t_true[:, np.newaxis]) < time_accuracy_threshold  # s
    diff_time = t_pred[np.newaxis, :] - t_true[:, np.newaxis]
    matched_idx = np.argmin(np.abs(diff_time), axis=1)[np.sum(evaluation_matrix, axis=1) > 0]
    recalled_idx = np.arange(xyz_true.shape[0])[np.sum(evaluation_matrix, axis=1) > 0]
    err_time = diff_time[np.arange(diff_time.shape[0]), np.argmin(np.abs(diff_time), axis=1)][
        np.sum(evaluation_matrix, axis=1) > 0
    ]

    err_z = []
    err_xy = []
    err_xyz = []
    err_loc = []
    t = []
    for i in range(len(recalled_idx)):
        tmp_z = np.abs(xyz_pred[matched_idx[i], 2] - xyz_true[recalled_idx[i], 2])
        tmp_xy = np.linalg.norm(xyz_pred[matched_idx[i], 0:2] - xyz_true[recalled_idx[i], 0:2])
        tmp_xyz = xyz_pred[matched_idx[i], :] - xyz_true[recalled_idx[i], :]
        tmp_loc = np.linalg.norm(xyz_pred[matched_idx[i], 0:3] - xyz_true[recalled_idx[i], 0:3])
        err_z.append(tmp_z)
        err_xy.append(tmp_xy)
        err_xyz.append(tmp_xyz)
        err_loc.append(tmp_loc)
        t.append(t_true[recalled_idx[i]])

    return np.array(err_time), np.array(err_xyz), np.array(err_xy), np.array(err_z), np.array(err_loc), np.array(t)


def calc_time_mag_error(t_pred, mag_pred, t_true, mag_true, time_accuracy_threshold):

    evaluation_matrix = np.abs(t_pred[np.newaxis, :] - t_true[:, np.newaxis]) < time_accuracy_threshold  # s
    diff_time = t_pred[np.newaxis, :] - t_true[:, np.newaxis]
    matched_idx = np.argmin(np.abs(diff_time), axis=1)[np.sum(evaluation_matrix, axis=1) > 0]
    recalled_idx = np.arange(mag_true.shape[0])[np.sum(evaluation_matrix, axis=1) > 0]
    err_time = diff_time[np.arange(diff_time.shape[0]), np.argmin(np.abs(diff_time), axis=1)][
        np.sum(evaluation_matrix, axis=1) > 0
    ]

    err_mag = []
    t = []
    mag = []
    for i in range(len(recalled_idx)):
        tmp_mag = mag_pred[matched_idx[i]] - mag_true[recalled_idx[i]]
        err_mag.append(tmp_mag)
        t.append(t_pred[matched_idx[i]])
        mag.append(mag_true[recalled_idx[i]])

    return np.array(err_time), np.array(err_mag), np.array(t), np.array(mag)



def plot_loc_error(
    t_pred, xyz_pred, t_true, xyz_true, time_accuracy_threshold, fig_name, xlim=None, ylim=None, station_locs=None
):

    evaluation_matrix = np.abs(t_pred[np.newaxis, :] - t_true[:, np.newaxis]) < time_accuracy_threshold  # s
    diff_time = t_pred[np.newaxis, :] - t_true[:, np.newaxis]
    matched_idx = np.argmin(np.abs(diff_time), axis=1)[np.sum(evaluation_matrix, axis=1) > 0]
    recalled_idx = np.arange(xyz_true.shape[0])[np.sum(evaluation_matrix, axis=1) > 0]
    # err_time = diff_time[np.arange(diff_time.shape[0]), np.argmin(np.abs(diff_time), axis=1)][
    #     np.sum(evaluation_matrix, axis=1) > 0
    # ]

    plt.figure()
    # plt.scatter(xyz_true[recalled_idx,0], xyz_true[recalled_idx,1], s=2, c="C3", alpha=0.8, label="SCSN")
    # plt.scatter(xyz_pred[matched_idx, 0], xyz_pred[matched_idx, 1], s=2, c="C0", marker="x", alpha=0.8, label="End2End")
    plt.plot(xyz_true[recalled_idx, 0], xyz_true[recalled_idx, 1], ".", color="C3", markersize=2, alpha=0.8)
    plt.plot(xyz_pred[matched_idx, 0], xyz_pred[matched_idx, 1], ".", color="C0", markersize=2, alpha=0.8)

    plt.plot(-100, -100, ".", color="C3", markersize=10, alpha=0.5, label="SCSN")
    plt.plot(-100, -100, ".", color="C0", markersize=10, alpha=0.5, label="End2End")

    if station_locs is not None:
        plt.scatter(station_locs[:, 0], station_locs[:, 1], color="k", marker="^", label="Station")
    plt.axis("scaled")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.legend()
    # plt.title("Earthquake locati")
    # for i in range(len(recalled_idx)):
    #   plt.plot([xyz_true[recalled_idx[i],0], xyz_pred[matched_idx[i], 0]], [xyz_true[recalled_idx[i],1], xyz_pred[matched_idx[i], 1]], '--')
    # plt.plot([10,40], [10, 40], 'r-')
    plt.savefig(fig_name + ".png", bbox_inches="tight")
    # plt.savefig(fig_name + ".pdf", bbox_inches="tight")


def plot_waveform(
    t_plot, xyz_plot, t_pred, t_true, station_locs, waveform, time, fig_dir, num_plot=50, type="pred", vp=6.0
):

    dt = 0.01

    for i in tqdm(range(min(len(t_plot), num_plot))):

        t = [int(t_plot[i]) - 10, int(t_plot[i]) + 35]
        dist = np.linalg.norm(xyz_plot[i] - station_locs, axis=1)

        plt.figure(figsize=(15, 6))
        for j in range(waveform.shape[0]):
            plt.plot(
                time[max([int(t[0] / dt), 0]) : int(t[1] / dt)],
                waveform[j, -1, max([int(t[0] / dt), 0]) : int(t[1] / dt)] * 3 + dist[j],
                linewidth=0.5,
                color="k",
            )
        plt.xlim(t)
        ylim = plt.gca().get_ylim()

        t_selected = t_true[(t[0] - 30 < t_true) & (t_true < t[1] + 30)]
        for j in range(len(t_selected)):
            if j == 0:
                label = "Catalog"
            else:
                label = ""
            (tmp,) = plt.plot([t_selected[j], t_selected[j]], ylim, "--", color="C1", linewidth=2, label=label)
        if type == "true":
            plt.plot(
                time[max([int(t[0] / dt), 0]) : int(t[1] / dt)],
                (time[max([int(t[0] / dt), 0]) : int(t[1] / dt)] - t_true[i]) * vp,
                ":",
                color="C1",
            )

        t_selected = t_pred[(t[0] - 30 < t_pred) & (t_pred < t[1] + 30)]
        for j in range(len(t_selected)):
            if j == 0:
                label = "End2End"
            else:
                label = ""
            (tmp,) = plt.plot([t_selected[j], t_selected[j]], ylim, "-", color="C0", linewidth=2, label=label)
        if type == "pred":
            plt.plot(
                time[max([int(t[0] / dt), 0]) : int(t[1] / dt)],
                (time[max([int(t[0] / dt), 0]) : int(t[1] / dt)] - t_pred[i]) * vp,
                ":",
                color="C0",
            )

        plt.ylim(ylim)
        plt.legend(loc="lower right")
        plt.ylabel("Distance (km)")
        plt.xlabel("Time (s)")
        plt.savefig(os.path.join(fig_dir, f"{i:04d}.png"))
        plt.close()


def plot_true_positive(
    t_pred,
    t_true,
    threshold,
    xyz_pred,
    date,
    fig_dir,
    data_dir=None,
    waveform=None,
    station_locs=None,
    num_plot=50,
    vp=6.0,
):
    """
    delta_time = [[pred1-true1, pred2-true1, pred3-true1, ...]
                  [pred1-true2, pred2-true2, pred3-true2, ...]
                  [pred1-true3, pred2-true3, pred3-true3, ...]
                  ...]
    """
    dt = 0.01

    ## load staion and waveforms
    if (waveform is None) and (data_dir is not None):
        station_locs = torch.load(os.path.join(data_dir, 'stations.pt'))[1]
        waveform = []
        for hour in tqdm(range(24), desc="Hour"):
            tmp = torch.load(os.path.join(data_dir, f"{date}/{hour:02d}.pt"))
            tmp = log_transform(tmp.type(torch.DoubleTensor))
            waveform.append(tmp)
        waveform = np.concatenate(waveform, axis=2)
        np.nan_to_num(waveform, copy=False)
    time = np.arange(waveform.shape[-1]) * dt

    ## find true positive
    diff_time = t_pred[np.newaxis, :] - t_true[:, np.newaxis]
    evaluation_matrix = np.abs(diff_time) < threshold  # s

    tp_idx = np.sum(evaluation_matrix, axis=0) > 0
    t_tp = t_pred[tp_idx]
    xyz_tp = xyz_pred[tp_idx]

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)
    np.seterr("ignore")

    ## plot true positive
    plot_waveform(t_tp, xyz_tp, t_tp, t_true, station_locs, waveform, time, fig_dir, type="pred")


def plot_false_positive(
    t_pred,
    t_true,
    threshold,
    xyz_pred,
    date,
    fig_dir,
    data_dir=None,
    waveform=None,
    station_locs=None,
    num_plot=50,
    vp=6.0,
):
    """
    delta_time = [[pred1-true1, pred2-true1, pred3-true1, ...]
                  [pred1-true2, pred2-true2, pred3-true2, ...]
                  [pred1-true3, pred2-true3, pred3-true3, ...]
                  ...]
    """
    dt = 0.01

    ## load staion and waveforms
    if (waveform is None) and (data_dir is not None):
        station_locs = torch.load(os.path.join(data_dir, 'stations.pt'))[1]
        waveform = []
        for hour in tqdm(range(24), desc="Hour"):
            tmp = torch.load(os.path.join(data_dir, f"{date}/{hour:02d}.pt"))
            tmp = log_transform(tmp.type(torch.DoubleTensor))
            waveform.append(tmp)
        waveform = np.concatenate(waveform, axis=2)
        np.nan_to_num(waveform, copy=False)
    time = np.arange(waveform.shape[-1]) * dt

    ## find false positive
    diff_time = t_pred[np.newaxis, :] - t_true[:, np.newaxis]
    evaluation_matrix = np.abs(diff_time) < threshold  # s

    fp_idx = np.sum(evaluation_matrix, axis=0) == 0
    t_fp = t_pred[fp_idx]
    xyz_fp = xyz_pred[fp_idx]

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)
    np.seterr("ignore")

    ## plot false positive
    plot_waveform(t_fp, xyz_fp, t_fp, t_true, station_locs, waveform, time, fig_dir, type="pred")


def plot_false_negative(
    t_pred,
    t_true,
    threshold,
    xyz_true,
    date,
    fig_dir,
    data_dir=None,
    waveform=None,
    station_locs=None,
    num_plot=50,
    vp=6.0,
):
    """
    delta_time = [[pred1-true1, pred2-true1, pred3-true1, ...]
                  [pred1-true2, pred2-true2, pred3-true2, ...]
                  [pred1-true3, pred2-true3, pred3-true3, ...]
                  ...]
    """
    dt = 0.01

    ## load staion and waveforms
    if (waveform is None) and (data_dir is not None):
        station_locs = torch.load(os.path.join(data_dir, 'stations.pt'))[1]
        waveform = []
        for hour in tqdm(range(24), desc="Hour"):
            tmp = torch.load(os.path.join(data_dir, f"{date}/{hour:02d}.pt"))
            tmp = log_transform(tmp.type(torch.DoubleTensor))
            waveform.append(tmp)
        waveform = np.concatenate(waveform, axis=2)
        np.nan_to_num(waveform, copy=False)
    time = np.arange(waveform.shape[-1]) * dt

    ## find false negative
    diff_time = t_pred[np.newaxis, :] - t_true[:, np.newaxis]
    evaluation_matrix = np.abs(diff_time) < threshold  # s

    fn_idx = np.sum(evaluation_matrix, axis=1) == 0
    t_fn = t_true[fn_idx]
    xyz_fn = xyz_true[fn_idx]

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)
    np.seterr("ignore")

    ## plot false negative
    plot_waveform(t_fn, xyz_fn, t_pred, t_fn, station_locs, waveform, time, fig_dir, type="true")


if __name__ == "__main__":
    # catalog = load_scsn()
    # print(catalog.iloc[0])

    # xmax = 101
    # ymax = 101
    # start_datetime = datetime.fromisoformat("2019-07-05T00:00:00.000")
    # end_datetime = datetime.fromisoformat("2019-07-07T00:00:00.000")
    # t_scsn, xyz_scsn = filter_scsn(load_scsn(), start_datetime, end_datetime, 0, xmax, 0, ymax)
    # pass
    fire.Fire(load_GaMMA_catalog)

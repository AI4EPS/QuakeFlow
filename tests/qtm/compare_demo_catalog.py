# %%
"""
Compare QTM detections on a real region against the catalogues it should reproduce.

The templates are cut from the AdLoc catalogue, so every AdLoc event is one QTM must
re-detect; anything else it returns is either a genuinely new (smaller) event or a false
alarm, and the two are told apart by how much of the network contributed. SCEDC is the
independent reference.

    python tests/qtm/compare_demo_catalog.py --region demo

Writes {region}/qtm/figures/qtm_vs_catalog.png and qtm_vs_catalog.csv.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import DEFAULT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--region", type=str, default="demo")
    parser.add_argument("--tolerance", type=float, default=2.0, help="origin-time tolerance for a match (s)")
    return parser.parse_args()


def match(a, b, tol):
    """greedy nearest-in-time, one-to-one; returns index into b for each row of a (-1 = none)"""
    ta = a["time"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    tb = b["time"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    order = np.argsort(tb)
    tb_sorted = tb[order]
    out = np.full(len(a), -1)
    used = set()
    idx = np.searchsorted(tb_sorted, ta)
    for i, t in enumerate(ta):
        best, bestdt = -1, tol
        for j in range(max(idx[i] - 3, 0), min(idx[i] + 3, len(tb_sorted))):
            k = int(order[j])
            dt = abs(tb_sorted[j] - t)
            if dt < bestdt and k not in used:
                best, bestdt = k, dt
        if best >= 0:
            used.add(best)
        out[i] = best
    return out


# %%
def main():
    args = parse_args()
    base = f"{args.root_path}/{args.region}"
    figdir = f"{base}/qtm/figures"
    os.makedirs(figdir, exist_ok=True)

    qtm = pd.read_csv(f"{base}/qtm/qtm_events.csv")
    qtm["time"] = pd.to_datetime(qtm["time"], utc=True, format="ISO8601")
    qtm = qtm.sort_values("time").reset_index(drop=True)

    adloc = pd.read_csv(f"{base}/adloc/adloc_events.csv")
    adloc["time"] = pd.to_datetime(adloc["time"], utc=True, format="ISO8601")
    adloc = adloc.sort_values("time").reset_index(drop=True)

    scedc = None
    if os.path.exists(f"{base}/obspy/catalog.csv"):
        scedc = pd.read_csv(f"{base}/obspy/catalog.csv")
        scedc["time"] = pd.to_datetime(scedc["time"], utc=True, format="ISO8601")
        scedc = scedc.sort_values("time").reset_index(drop=True)

    ## how many stations backed each QTM detection -- the discriminator the benchmark found
    picks = pd.read_csv(f"{base}/qtm/qtm_picks.csv")
    if "event_index" in picks.columns and "idx_sta" in picks.columns:
        nsta = picks[picks["event_index"] >= 0].groupby("event_index")["idx_sta"].nunique()
        qtm["n_station"] = qtm["event_index"].map(nsta).fillna(0).astype(int)
    else:
        qtm["n_station"] = np.nan

    print(f"QTM        : {len(qtm):>6} events")
    print(f"AdLoc      : {len(adloc):>6} events (the templates)")
    if scedc is not None:
        print(f"SCEDC      : {len(scedc):>6} events")

    ## AdLoc -> QTM: recovery of the template events
    j = match(adloc, qtm, args.tolerance)
    adloc["qtm_row"] = j
    rec = (j >= 0).mean()
    print(f"\ntemplate recovery : {int((j >= 0).sum())}/{len(adloc)} = {rec:.1%} of AdLoc events re-detected")

    ## QTM -> AdLoc: which detections are new
    k = match(qtm, adloc, args.tolerance)
    qtm["adloc_row"] = k
    qtm["is_new"] = k < 0
    print(f"new detections    : {int(qtm['is_new'].sum())}/{len(qtm)} not in the AdLoc catalogue")
    if qtm["n_station"].notna().any():
        for thr in [0, 4, 8, 12, 16, 20]:
            m = qtm["n_station"] >= thr
            print(f"   >= {thr:2d} stations: {int(m.sum()):>6} kept, of which {int((m & qtm['is_new']).sum()):>6} new")

    if scedc is not None:
        s = match(scedc, qtm, args.tolerance)
        print(f"\nSCEDC recovery    : {int((s >= 0).sum())}/{len(scedc)} = {(s >= 0).mean():.1%}")

    qtm.to_csv(f"{base}/qtm/qtm_vs_catalog.csv", index=False)

    # %% figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t0 = min(qtm["time"].min(), adloc["time"].min())

    ax = axes[0, 0]
    for d, lab, c in [(adloc, f"AdLoc ({len(adloc)})", "tab:blue"),
                      (qtm[~qtm["is_new"]], f"QTM, matched ({int((~qtm['is_new']).sum())})", "tab:green"),
                      (qtm[qtm["is_new"]], f"QTM, new ({int(qtm['is_new'].sum())})", "tab:red")]:
        ax.plot((d["time"] - t0).dt.total_seconds() / 3600, np.arange(len(d)), ".", ms=2, color=c, label=lab)
    ax.set_xlabel("hours into the record")
    ax.set_ylabel("cumulative count")
    ax.set_title("cumulative detections", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if qtm["n_station"].notna().any():
        bins = np.arange(0, qtm["n_station"].max() + 2)
        ax.hist([qtm.loc[~qtm["is_new"], "n_station"], qtm.loc[qtm["is_new"], "n_station"]], bins=bins,
                stacked=True, color=["tab:green", "tab:red"], label=["matches AdLoc", "new"])
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    ax.set_xlabel("distinct stations contributing")
    ax.set_ylabel("detections")
    ax.set_title("network support\n(the benchmark's cleanest real/false discriminator)", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    m = adloc["qtm_row"] >= 0
    if m.any():
        dt = (qtm.loc[adloc.loc[m, "qtm_row"], "time"].values - adloc.loc[m, "time"].values) / np.timedelta64(1, "s")
        ax.hist(dt, bins=np.linspace(-args.tolerance, args.tolerance, 61), color="tab:blue")
        ax.set_title(f"origin-time error vs AdLoc\nmedian {np.median(dt):+.3f} s", fontsize=10)
    ax.set_xlabel("QTM - AdLoc (s)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if "magnitude" in adloc.columns:
        bins = np.arange(-2, 5.1, 0.2)
        ax.hist(adloc["magnitude"], bins=bins, histtype="step", lw=2, color="tab:blue", label="AdLoc")
        det = adloc.loc[adloc["qtm_row"] >= 0, "magnitude"]
        ax.hist(det, bins=bins, histtype="step", lw=2, color="tab:green", label="AdLoc re-detected by QTM")
        ax.set_yscale("log")
        ax.set_xlabel("magnitude")
        ax.set_ylabel("events")
        ax.set_title("which magnitudes QTM recovers", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"QTM on {args.region}: {len(qtm)} detections vs {len(adloc)} template events", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{figdir}/qtm_vs_catalog.png", dpi=150)
    print(f"\nwrote {figdir}/qtm_vs_catalog.png")
    print(f"wrote {base}/qtm/qtm_vs_catalog.csv")


if __name__ == "__main__":
    main()

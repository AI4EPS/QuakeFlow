# %%
"""
Step 4: score the QTM output against the benchmark ground truth.

Two levels, because they fail for different reasons:

  correlation   the raw CCTorch output (qtm/ccpairs/TM_*.csv).  Every injection is a copy
                of the waveform its template was cut from, so the matching template must
                peak at cc = 1 at the ground-truth origin time.  Anything less is a
                template/continuous-data mismatch, not a detection problem.
  association   qtm/qtm_events.csv.  Recall, false alarms, origin-time and location error.

    python tests/qtm/evaluate_qtm.py --region synthetic

Figures go to {region}/benchmark/figures/, the scored tables to {region}/benchmark/.
"""

import argparse
import os
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import DEFAULT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--region", type=str, default="synthetic")
    parser.add_argument("--tolerance", type=float, default=2.0, help="origin-time tolerance for a match (s)")
    return parser.parse_args()


# %%
def load(args):
    base = f"{args.root_path}/{args.region}"
    gt = pd.read_csv(f"{base}/benchmark/ground_truth_events.csv")
    gt["time"] = pd.to_datetime(gt["time"], utc=True, format="ISO8601")
    gp = pd.read_csv(f"{base}/benchmark/ground_truth_picks.csv")

    tm = [pd.read_csv(f, parse_dates=["phase_time", "origin_time"])
          for f in sorted(glob(f"{base}/qtm/ccpairs/TM_???_???.csv"))]
    if not tm:
        raise SystemExit(f"no {base}/qtm/ccpairs/TM_*.csv -- run run_qtm.py first")
    tm = pd.concat(tm, ignore_index=True)
    tm["origin_time"] = pd.to_datetime(tm["origin_time"], utc=True)

    ## idx_eve indexes the *template* table template_events.csv, whose event_index is the
    ## benchmark template id -- not qtm_events.csv, which is the output catalogue
    cc_events = pd.read_csv(f"{base}/qtm/template_events.csv")
    tmpl2src = gt.dropna(subset=["template_event_index"]).drop_duplicates("template_event_index")
    tmpl2src = tmpl2src.set_index("template_event_index")["idx_event"]
    tm["template_event_index"] = tm["idx_eve"].map(cc_events.set_index("idx_eve")["event_index"])
    tm["src_idx_event"] = tm["template_event_index"].map(tmpl2src)

    stations = pd.read_csv(f"{base}/qtm/template_stations.csv")
    tm = tm.merge(stations[["idx_sta", "station_id"]], on="idx_sta", how="left")
    tm["instrument"] = tm["station_id"].str.split(".").str[3]

    ev = None
    if os.path.exists(f"{base}/qtm/qtm_events.csv"):
        ev = pd.read_csv(f"{base}/qtm/qtm_events.csv", parse_dates=["time"])
        ev["time"] = pd.to_datetime(ev["time"], utc=True)
    return gt, gp, tm, ev


# %%
def score_correlation(args, gt, gp, tm):
    """For every (injection, template) pair, the best cc within the tolerance window."""
    rows = []
    self_picks = []
    for _, g in gt.iterrows():
        lo = g["time"] - pd.Timedelta(seconds=args.tolerance)
        hi = g["time"] + pd.Timedelta(seconds=args.tolerance)
        near = tm[(tm["origin_time"] >= lo) & (tm["origin_time"] <= hi)]
        self_picks.append(near[near["src_idx_event"] == g["idx_event"]].assign(inj_id=g["inj_id"]))
        for src, h in near.groupby("src_idx_event"):
            best = h.loc[h["cc"].idxmax()]
            rows.append(
                {
                    "inj_id": g["inj_id"],
                    "block": g["block"],
                    "idx_event": g["idx_event"],
                    "template_idx_event": int(src),
                    "self": int(src) == int(g["idx_event"]),
                    "n_picks": len(h),
                    "n_stations": h["idx_sta"].nunique(),
                    "cc_max": float(h["cc"].max()),
                    "cc_median": float(h["cc"].median()),
                    "dt_sec": float((best["origin_time"] - g["time"]).total_seconds()),
                }
            )
    cc = pd.DataFrame(rows)
    if "set_id" in gt.columns:
        cc = cc.merge(gt[["inj_id", "set_id", "amplitude"]], on="inj_id", how="left")

    self_cc = cc[cc["self"]]
    print("\n=== correlation level ===")
    print(f"injections with any matching-template hit : {len(self_cc)}/{len(gt)}")
    if len(self_cc):
        print(f"self-match cc_max   : median {self_cc['cc_max'].median():.3f}  "
              f"range {self_cc['cc_max'].min():.3f}-{self_cc['cc_max'].max():.3f}")
        print(f"self-match cc_median: median {self_cc['cc_median'].median():.3f}")
        print(f"self-match stations : median {self_cc['n_stations'].median():.0f} of 45")
        print(f"origin-time offset  : median {self_cc['dt_sec'].median():+.3f} s, "
              f"std {self_cc['dt_sec'].std():.3f} s")
    sp = pd.concat(self_picks, ignore_index=True) if self_picks else pd.DataFrame()

    ## The template block is the injection each template was cut from, so the correlation
    ## is a waveform against itself: every event x station x phase must come back at 1.0.
    ## Anything short of that is a processing difference, not a detection limit.
    tmpl = gt[gt["block"] == "template"]
    ## only what was actually injected: a cut taken from a dead stretch of the source record
    ## puts no signal in, so there is nothing for the template to find
    expected = int(gp[gp["inj_id"].isin(tmpl["inj_id"]) & gp["injected"]].shape[0])
    ident = sp[sp["inj_id"].isin(tmpl["inj_id"])] if len(sp) else pd.DataFrame()
    if len(ident):
        ident = ident.loc[ident.groupby(["inj_id", "idx_sta", "phase_type"])["cc"].idxmax()]
        print(f"\nidentity check: {expected} injected template phases "
              f"({len(tmpl)} events x stations x 2 phases, dead channels excluded)")
        print(f"  returned {len(ident)} ({len(ident) / max(expected, 1):.1%}), "
              f"cc min {ident['cc'].min():.6f}, median {ident['cc'].median():.6f}")
        if len(ident) < expected or ident["cc"].min() < 0.999:
            gap = ident[ident["cc"] < 0.999]
            print(f"  *** {expected - len(ident)} missing and {len(gap)} below 0.999 -- a waveform is "
                  f"not being correlated with itself")
        else:
            print("  every matched phase is 1.0")
    ## Only the template set is a waveform against itself; the test sets are the same
    ## waveform in different noise at a lower amplitude, so their cc is *supposed* to fall
    ## with SNR.  Scoring them against 1.0 would report the detection limit as a bug.
    tsp = sp[sp["inj_id"].isin(tmpl["inj_id"])] if len(sp) else sp
    if len(tsp):
        tab = tsp.pivot_table(index="instrument", columns="phase_type", values="cc",
                              aggfunc=["count", "median"])
        print(f"\ntemplate-set cc per pick ({len(tsp)} picks) -- every one of these should be ~1.0:")
        print(tab.round(3).to_string())
        print(f"  all picks: median {tsp['cc'].median():.3f}, "
              f"{(tsp['cc'] >= 0.95).mean():.1%} at or above 0.95")

    tmpl_cc = self_cc[self_cc["inj_id"].isin(tmpl["inj_id"])]
    if len(tmpl_cc) and tmpl_cc["cc_max"].median() < 0.9:
        print(f"  *** each template-set injection is an exact copy of its template, so cc should be ~1.0.\n"
              f"      {tmpl_cc['cc_max'].median():.2f} means the template and the continuous data are not\n"
              f"      being treated the same way (cut_templates_cc.py bandpasses the templates\n"
              f"      2-12 Hz; CCTorch's read_mseed only demeans the continuous record).")
    return cc, sp


# %%
def score_by_set(gt, gp, sp, labelled=None):
    """Pick recovery per injection set -- the detection-vs-SNR curve.

    Every set is a full pass over every source, so the injected-pick count is identical
    across sets and the recovered fraction is comparable set to set with no reweighting.
    """
    if "set_id" not in gt.columns:
        return None

    expected = (
        gp[gp["injected"]].merge(gt[["inj_id", "set_id"]], on="inj_id")
        .groupby("set_id").size().rename("injected")
    )
    if len(sp):
        ## one row per station-phase: the best cc that came back at the ground-truth time
        got = sp.loc[sp.groupby(["inj_id", "idx_sta", "phase_type"])["cc"].idxmax()]
        got = got.merge(gt[["inj_id", "set_id"]], on="inj_id")
        found = got.groupby("set_id").size().rename("returned")
        ccmed = got.groupby("set_id")["cc"].median().rename("cc_median")
    else:
        found = pd.Series(0, index=expected.index, name="returned")
        ccmed = pd.Series(np.nan, index=expected.index, name="cc_median")

    t = pd.concat([gt.groupby("set_id")["amplitude"].first().rename("snr"),
                   gt.groupby("set_id")["block"].first(),
                   expected, found, ccmed], axis=1).reset_index()
    t["returned"] = t["returned"].fillna(0).astype(int)
    t["recovery"] = t["returned"] / t["injected"]

    ## true and false positives per set.  wrong_time picks sit on no injection, so they
    ## belong to no set and are reported once, globally, below the table.
    if labelled is not None:
        lab = labelled[labelled["set_id"].notna()]
        piv = lab.pivot_table(index="set_id", columns="label", values="cc", aggfunc="size").fillna(0)
        for k in ["true_positive", "wrong_template", "dead_channel", "duplicate"]:
            t[k] = t["set_id"].map(piv[k]).fillna(0).astype(int) if k in piv else 0
        t["false_positive"] = t["wrong_template"] + t["dead_channel"]
        t["precision"] = t["true_positive"] / (t["true_positive"] + t["false_positive"]).replace(0, np.nan)

    print("\n=== pick recovery per set (the detection-vs-SNR curve) ===")
    head = f"{'set':>4}{'SNR':>9}{'block':>10}{'injected':>10}{'TP':>8}{'recall':>9}{'cc_med':>9}"
    if "false_positive" in t:
        head += f"{'FP':>8}{'precision':>11}"
    print(head)
    for _, r in t.iterrows():
        line = (f"{int(r['set_id']):>4}{r['snr']:>9g}{r['block']:>10}{int(r['injected']):>10}"
                f"{int(r['returned']):>8}{r['recovery']:>8.1%}{r['cc_median']:>9.4f}")
        if "false_positive" in t:
            p = r["precision"]
            line += f"{int(r['false_positive']):>8}" + (f"{p:>10.1%}" if np.isfinite(p) else f"{'-':>11}")
        print(line)
    if labelled is not None:
        nt = int((labelled["label"] == "wrong_time").sum())
        print(f"  + {nt} wrong_time picks on no injection at all (belong to no set)")
    return t


# %%
def score_picks(args, gt, gp, tm):
    """Classify every raw correlation pick as a true or false positive.

    A pick is a true positive only if the template that fired is the source that was
    actually injected, at that injection's ground-truth origin time, on a station-phase that
    really carried signal.  The false positives split into two kinds that have different
    causes and different fixes:

      wrong_template  landed on a real injection but fired the wrong source's template.
                      These are the off-diagonal correlations between sources a few km
                      apart -- duplicates of a real event, not noise triggers.
      wrong_time      fired where nothing was injected at all.

    Extra picks on a station-phase already counted are reported as duplicates rather than
    folded into either, so precision is not distorted by a template peaking twice.
    """
    tm = tm.sort_values("origin_time").reset_index(drop=True)
    tol = pd.Timedelta(seconds=args.tolerance)

    ## which injection (if any) each pick sits on -- windows do not overlap, so nearest wins
    g = gt.sort_values("time").reset_index(drop=True)
    near = pd.merge_asof(tm[["origin_time"]], g[["time", "inj_id", "idx_event", "set_id"]].rename(
        columns={"time": "gt_time", "idx_event": "gt_idx_event"}),
        left_on="origin_time", right_on="gt_time", direction="nearest", tolerance=tol)
    tm = pd.concat([tm, near.drop(columns=["origin_time"])], axis=1)

    ## station-phases that actually carried signal, per injection
    injected = set(
        zip(gp.loc[gp["injected"], "inj_id"], gp.loc[gp["injected"], "station_id"],
            gp.loc[gp["injected"], "phase_type"])
    )
    on_event = tm["inj_id"].notna()
    right_tmpl = on_event & (tm["src_idx_event"] == tm["gt_idx_event"])
    has_signal = pd.Series(
        [(i, s, p) in injected for i, s, p in zip(tm["inj_id"], tm["station_id"], tm["phase_type"])],
        index=tm.index,
    )

    tm["label"] = "wrong_time"
    tm.loc[on_event & ~right_tmpl, "label"] = "wrong_template"
    tm.loc[right_tmpl & has_signal, "label"] = "true_positive"
    tm.loc[right_tmpl & ~has_signal, "label"] = "dead_channel"

    ## only the first hit on a station-phase counts; the rest are duplicate peaks
    tp = tm[tm["label"] == "true_positive"]
    keep = tp.loc[tp.groupby(["inj_id", "idx_sta", "phase_type"])["cc"].idxmax()].index
    tm.loc[tp.index.difference(keep), "label"] = "duplicate"

    n = tm["label"].value_counts()
    n_tp = int(n.get("true_positive", 0))
    n_fp = int(n.get("wrong_template", 0) + n.get("wrong_time", 0) + n.get("dead_channel", 0))
    n_inj = int(gp["injected"].sum())

    print("\n=== pick level: true vs false positives ===")
    print(f"{len(tm)} raw correlation picks (cc >= min_cc), {n_inj} injected station-phases")
    for k in ["true_positive", "duplicate", "wrong_template", "wrong_time", "dead_channel"]:
        v = int(n.get(k, 0))
        if v:
            print(f"  {k:<16} {v:>6}  ({v / len(tm):.1%} of picks)   median cc "
                  f"{tm.loc[tm['label'] == k, 'cc'].median():.3f}")
    print(f"  precision {n_tp}/{n_tp + n_fp} = {n_tp / max(n_tp + n_fp, 1):.1%}   "
          f"recall {n_tp}/{n_inj} = {n_tp / max(n_inj, 1):.1%}")

    ## where a cc threshold would put the trade-off -- the false positives sit low in cc,
    ## so this is the cheapest lever on precision
    is_tp = tm["label"] == "true_positive"
    is_fp = tm["label"].isin(["wrong_template", "wrong_time", "dead_channel"])
    print(f"\n  {'min_cc':>7}{'TP':>8}{'FP':>8}{'precision':>11}{'recall':>9}")
    for thr in [0.5, 0.6, 0.65, 0.7, 0.8, 0.9]:
        keep = tm["cc"] >= thr
        a, b = int((is_tp & keep).sum()), int((is_fp & keep).sum())
        print(f"  {thr:>7.2f}{a:>8}{b:>8}{a / max(a + b, 1):>10.1%}{a / max(n_inj, 1):>9.1%}")
    return tm


# %%
def score_association(args, gt, ev):
    """Greedy nearest-in-time match of catalogue entries to injections."""
    if ev is None or len(ev) == 0:
        print("\n=== association level ===\nno qtm_events.csv")
        return gt.assign(detected=False, dt_sec=np.nan), None

    gt = gt.sort_values("time").reset_index(drop=True)
    ev = ev.sort_values("time").reset_index(drop=True)
    used = set()
    det, dts, idx = [], [], []
    for _, g in gt.iterrows():
        d = (ev["time"] - g["time"]).dt.total_seconds().abs()
        d[list(used)] = np.inf
        k = int(d.idxmin())
        if d[k] <= args.tolerance:
            used.add(k)
            det.append(True)
            dts.append(float((ev.loc[k, "time"] - g["time"]).total_seconds()))
            idx.append(k)
        else:
            det.append(False)
            dts.append(np.nan)
            idx.append(-1)
    gt = gt.assign(detected=det, dt_sec=dts, qtm_row=idx)
    ev = ev.assign(matched=[i in used for i in range(len(ev))])

    n_det = int(gt["detected"].sum())
    print("\n=== association level ===")
    print(f"recall            : {n_det}/{len(gt)} = {n_det / len(gt):.1%}")
    print(f"false alarms      : {int((~ev['matched']).sum())}/{len(ev)} catalogue entries unmatched")
    if n_det:
        print(f"origin-time error : median {gt['dt_sec'].median():+.3f} s, "
              f"90% within {np.nanpercentile(np.abs(gt['dt_sec']), 90):.2f} s")
        m = gt[gt["detected"]]
        km = np.hypot(
            (ev.loc[m["qtm_row"], "longitude"].values - m["longitude"].values)
            * 111.19 * np.cos(np.deg2rad(m["latitude"].values)),
            (ev.loc[m["qtm_row"], "latitude"].values - m["latitude"].values) * 111.19,
        )
        print(f"location error    : median {np.median(km):.1f} km "
              f"(QTM reports the template's location, so this is template-assignment error)")
        gt.loc[gt["detected"], "loc_err_km"] = km
    if "set_id" in gt.columns:
        print("by set:")
        for sid, r in gt.groupby("set_id"):
            print(f"  set {int(sid)}  SNR {r['amplitude'].iloc[0]:<8g} {r['block'].iloc[0]:9s} "
                  f"{int(r['detected'].sum())}/{len(r)}")
    else:
        by_block = gt.groupby("block")["detected"].agg(["sum", "size"])
        print("by block:")
        for b, r in by_block.iterrows():
            print(f"  {b:9s} {int(r['sum'])}/{int(r['size'])}")
    return gt, ev


# %%
def plot(args, gt, cc, ev, by_set, labelled):
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    os.makedirs(figdir, exist_ok=True)
    t0 = gt["time"].min()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    ## 1. timeline
    ax = axes[0, 0]
    mins = (gt["time"] - t0).dt.total_seconds() / 60
    ax.scatter(mins[gt["detected"]], gt.loc[gt["detected"], "idx_event"], marker="o", s=45,
               facecolor="tab:green", edgecolor="k", lw=0.4, label="detected")
    ax.scatter(mins[~gt["detected"]], gt.loc[~gt["detected"], "idx_event"], marker="x", s=55,
               color="tab:red", label="missed")
    if ev is not None:
        fa = ev[~ev["matched"]]
        ax.scatter((fa["time"] - t0).dt.total_seconds() / 60, np.full(len(fa), -1.2), marker="^",
                   s=35, color="tab:orange", label="false alarm")
    ax.set_xlabel("minutes into the record")
    ax.set_ylabel("source event")
    ax.set_title("what QTM found", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ## 2. cc matrix: template vs injected source
    ax = axes[0, 1]
    n = int(gt["idx_event"].max()) + 1
    mat = np.full((n, n), np.nan)
    if len(cc):
        piv = cc.groupby(["template_idx_event", "idx_event"])["cc_max"].median()
        for (i, j), v in piv.items():
            mat[int(i), int(j)] = v
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="magma", origin="lower")
    plt.colorbar(im, ax=ax, label="median cc_max")
    for i in range(n):
        if np.isfinite(mat[i, i]):
            ax.text(i, i, f"{mat[i, i]:.2f}", ha="center", va="center", fontsize=6,
                    color="w" if mat[i, i] < 0.6 else "k")
    ax.set_xlabel("injected source event")
    ax.set_ylabel("template")
    ax.set_title("cc at the ground-truth origin time\n(diagonal is the exact copy, should be 1.0)", fontsize=10)

    ## 3. self-match cc distribution
    ax = axes[0, 2]
    s = cc[cc["self"]]
    if len(s):
        ax.hist(s["cc_max"], bins=np.linspace(0, 1, 41), color="tab:blue", edgecolor="k", lw=0.4,
                label="matching template")
    o = cc[~cc["self"]]
    if len(o):
        ax.hist(o["cc_max"], bins=np.linspace(0, 1, 41), color="tab:orange", alpha=0.6,
                edgecolor="k", lw=0.4, label="other templates")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("cc_max at the ground-truth time")
    ax.set_ylabel("injection-template pairs")
    ax.set_title("self-match should pile up at 1.0", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ## 4. the decay curve: pick recovery and event recall against set SNR
    ax = axes[1, 0]
    if by_set is not None and len(by_set):
        t = by_set.sort_values("snr")
        ax.semilogx(t["snr"], t["recovery"], "o-", color="tab:blue", label="picks recovered")
        rec = gt.groupby("set_id").agg(snr=("amplitude", "first"), r=("detected", "mean")).sort_values("snr")
        ax.semilogx(rec["snr"], rec["r"], "s--", color="tab:green", label="events detected")
        for _, r in t.iterrows():
            ax.annotate(f"{int(r['returned'])}/{int(r['injected'])}", (r["snr"], r["recovery"]),
                        fontsize=6.5, xytext=(0, 7), textcoords="offset points", ha="center")
        ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("injected SNR of the set")
    ax.set_ylabel("fraction recovered")
    ax.set_title("detection decay with SNR\n(one point per injection set)", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    ## 5. origin-time error
    ax = axes[1, 1]
    if gt["detected"].any():
        ax.hist(gt.loc[gt["detected"], "dt_sec"], bins=25, color="tab:green", edgecolor="k", lw=0.4)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("QTM origin time - ground truth (s)")
    ax.set_ylabel("detections")
    ax.set_title("origin-time error", fontsize=10)
    ax.grid(alpha=0.3)

    ## 6. pick-level true vs false positives, by cc
    ax = axes[1, 2]
    if labelled is not None and len(labelled):
        bins = np.linspace(0.5, 1.0, 26)
        order = [("true_positive", "tab:green"), ("wrong_template", "tab:orange"),
                 ("wrong_time", "tab:red"), ("duplicate", "0.6"), ("dead_channel", "tab:purple")]
        keep = [(k, c) for k, c in order if (labelled["label"] == k).any()]
        ax.hist([labelled.loc[labelled["label"] == k, "cc"] for k, _ in keep], bins=bins, stacked=True,
                color=[c for _, c in keep],
                label=[f"{k} ({int((labelled['label'] == k).sum())})" for k, _ in keep])
        ax.legend(fontsize=6.5)
        ax.set_yscale("log")
    ax.set_xlabel("cc of the correlation pick")
    ax.set_ylabel("picks")
    ax.set_title("true vs false positives, by cc\n(a cc threshold cuts vertically here)", fontsize=10)
    ax.grid(alpha=0.3)

    sets = "/".join(f"{v:g}" for v in gt.groupby("set_id")["amplitude"].first()) \
        if "set_id" in gt.columns else f"{gt['template_ratio'].iloc[0]:g}"
    fig.suptitle(
        f"QTM on the benchmark: set SNR {sets}, "
        f"{len(gt)} injections, {(gt['time'].diff().dt.total_seconds()).median():.0f} s apart",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(f"{figdir}/eval_summary.png", dpi=160)
    plt.close(fig)
    print(f"\n  {figdir}/eval_summary.png")


# %%
def plot_recovery(args, gt, gp, labelled):
    """Event detection over the record, and every pick's fate as a station x injection raster.

    One cell per (station, injection, phase), so the whole experiment is on one page: which
    station lost which pick, at which SNR, and whether the loss was a miss or the wrong
    template firing.  Stations are ordered by how many picks they recovered, which puts the
    weak ones at the bottom where they read as rows rather than scattered cells.
    """
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    os.makedirs(figdir, exist_ok=True)

    gt = gt.sort_values("time").reset_index(drop=True)
    order = list(gt["inj_id"])
    col = {v: i for i, v in enumerate(order)}
    has_sets = "set_id" in gt.columns

    ## status per (station, injection, phase): 0 missed, 1 found, 2 wrong template, 3 not injected
    MISSED, FOUND, WRONG, DEAD = 0, 1, 2, 3
    tp = labelled[labelled["label"] == "true_positive"]
    wt = labelled[labelled["label"] == "wrong_template"]
    stations = sorted(gp["station_id"].unique())
    row = {s: i for i, s in enumerate(stations)}

    mats = {}
    for ph in ["P", "S"]:
        m = np.full((len(stations), len(order)), DEAD, dtype=int)
        g = gp[(gp["phase_type"] == ph) & gp["injected"]]
        m[[row[s] for s in g["station_id"]], [col[i] for i in g["inj_id"]]] = MISSED
        w = wt[wt["phase_type"] == ph]
        if len(w):
            m[[row[s] for s in w["station_id"]], [col[i] for i in w["inj_id"]]] = WRONG
        t = tp[tp["phase_type"] == ph]
        m[[row[s] for s in t["station_id"]], [col[i] for i in t["inj_id"]]] = FOUND
        mats[ph] = m

    ## best-recovering stations at the top
    score = (mats["P"] == FOUND).sum(axis=1) + (mats["S"] == FOUND).sum(axis=1)
    idx = np.argsort(-score)
    stations = [stations[i] for i in idx]
    for ph in mats:
        mats[ph] = mats[ph][idx]
    score = score[idx]

    cmap = matplotlib.colors.ListedColormap(["#f0f0f0", "#1a9850", "#f46d43", "#ffffff"])
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 3, 3], width_ratios=[20, 1.6], hspace=0.16, wspace=0.02)

    ## ---- top: event detection over the record
    ax = fig.add_subplot(gs[0, 0])
    mins = (gt["time"] - gt["time"].min()).dt.total_seconds() / 60
    det = gt["detected"].values if "detected" in gt.columns else np.zeros(len(gt), bool)
    ax.scatter(np.arange(len(gt))[det], gt["idx_event"].values[det], marker="o", s=55,
               facecolor="#1a9850", edgecolor="k", lw=0.5, label="event detected", zorder=3)
    ax.scatter(np.arange(len(gt))[~det], gt["idx_event"].values[~det], marker="X", s=70,
               color="#d73027", label="event missed", zorder=3)
    ax.set_ylabel("source event")
    fig.suptitle(f"QTM on the benchmark: {int(det.sum())}/{len(gt)} events detected, "
                 f"{int((labelled['label'] == 'true_positive').sum())}/{int(gp['injected'].sum())} picks recovered",
                 fontsize=13, y=0.995)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_ylim(-1, gt["idx_event"].max() + 1)

    ## ---- the two rasters
    for k, ph in enumerate(["P", "S"]):
        a = fig.add_subplot(gs[k + 1, 0])
        a.imshow(mats[ph], aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        a.set_yticks(range(len(stations)))
        a.set_yticklabels(stations, fontsize=5.5)
        a.set_ylabel(f"{ph} phase   (station, ordered by picks recovered)", fontsize=9)
        n_f = int((mats[ph] == FOUND).sum())
        n_i = int((mats[ph] != DEAD).sum())
        a.text(0.005, 1.012, f"{ph}: {n_f}/{n_i} recovered", transform=a.transAxes, fontsize=9)
        for x in np.arange(-0.5, len(order), 1):
            a.axvline(x, color="w", lw=0.15)

        ## right margin: per-station recovery
        b = fig.add_subplot(gs[k + 1, 1])
        v = (mats[ph] == FOUND).sum(axis=1)
        b.barh(range(len(stations)), v, color="#1a9850", height=0.8)
        b.set_ylim(len(stations) - 0.5, -0.5)
        b.set_xlim(0, len(order))
        b.set_yticks([])
        b.tick_params(labelsize=6)
        b.set_xlabel("picks", fontsize=7)

    ## set boundaries and SNR labels on every panel
    axes_all = [fig.axes[0], fig.axes[1], fig.axes[3]]
    if has_sets:
        bounds, labels = [], []
        for sid, g in gt.groupby("set_id"):
            lo, hi = col[g["inj_id"].iloc[0]], col[g["inj_id"].iloc[-1]]
            bounds.append(hi + 0.5)
            labels.append(((lo + hi) / 2, f"set {int(sid)}\nSNR {g['amplitude'].iloc[0]:g}\n"
                                          f"{g['block'].iloc[0]}"))
        for a in axes_all:
            for x in bounds[:-1]:
                a.axvline(x, color="k", lw=1.4, ls="-", alpha=0.8)
        tr = fig.axes[0].get_xaxis_transform()
        for x, t in labels:
            fig.axes[0].text(x, 1.04, t, transform=tr, ha="center", va="bottom", fontsize=8.5)
    for a in axes_all:
        a.set_xlim(-0.5, len(order) - 0.5)
    fig.axes[0].set_xticks([])
    fig.axes[1].set_xticks([])
    fig.axes[3].set_xticks(range(0, len(order), max(len(order) // 25, 1)))
    fig.axes[3].set_xticklabels([f"{int(i)}" for i in range(0, len(order), max(len(order) // 25, 1))], fontsize=6)
    fig.axes[3].set_xlabel(f"injection (in time order, {mins.iloc[-1]:.0f} min of record)", fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.5", lw=0.4) for c in
               ["#1a9850", "#f0f0f0", "#f46d43", "#ffffff"]]
    fig.legend(handles, ["pick recovered", "pick missed", "wrong template fired",
                         "not injected (dead channel)"],
               loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(top=0.905, bottom=0.055, left=0.075, right=0.98)
    fig.savefig(f"{figdir}/eval_recovery.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/eval_recovery.png")


# %%
def plot_cc_vs_snr(args, gt, gp, labelled):
    """cc against the injected SNR, with the two-noise prediction on top.

    For a template s+n1 correlated against a copy a*s+n2 in independent noise, the
    normalised correlation is

        cc = 1 / sqrt( (1 + 1/SNR_template^2) * (1 + 1/SNR_data^2) )

    so cc ~ SNR at low SNR, saturates at the template's own SNR, and passes 0.7 at SNR 1.
    Points below min_cc never enter the file, so the low-SNR medians are censored upward --
    the recall panel is what to read there instead.
    """
    figdir = f"{args.root_path}/{args.region}/benchmark/figures"
    tp = labelled[labelled["label"] == "true_positive"].drop(columns=["set_id"], errors="ignore")
    tp = tp.merge(gp[["inj_id", "station_id", "phase_type", "snr"]],
                  on=["inj_id", "station_id", "phase_type"], how="left")
    tp = tp.merge(gt[["inj_id", "set_id", "amplitude"]], on="inj_id", how="left")
    if not len(tp) or tp["snr"].isna().all():
        return
    snr_t = float(gt.loc[gt["block"] == "template", "amplitude"].iloc[0])
    theory = lambda sd: 1 / np.sqrt((1 + 1 / snr_t ** 2) * (1 + 1 / np.asarray(sd, float) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    x = np.logspace(-1.3, 1.5, 200)

    ax = axes[0]
    ax.plot(x, theory(x), "k-", lw=2, zorder=5,
            label=f"$1/\\sqrt{{(1+1/{snr_t:g}^2)(1+1/SNR^2)}}$")
    for ph, c, off in [("P", "tab:red", 0.93), ("S", "tab:blue", 1.07)]:
        g = tp[tp["phase_type"] == ph]
        if not len(g):
            continue
        pos, dat = [], []
        for sid, h in g.groupby("set_id"):
            pos.append(float(h["amplitude"].iloc[0]) * off)
            dat.append(h["cc"].values)
        bp = ax.boxplot(dat, positions=pos, widths=[p * 0.13 for p in pos], patch_artist=True,
                        showfliers=False, manage_ticks=False)
        for b in bp["boxes"]:
            b.set(facecolor=c, alpha=0.55, lw=0.6)
        for k in ["medians", "whiskers", "caps"]:
            for b in bp[k]:
                b.set(color="k", lw=0.7)
        ax.plot([], [], "s", color=c, label=f"{ph} (observed)")
    ax.axhline(0.5, color="tab:green", ls="--", lw=1.2)
    snr_at = 1 / np.sqrt((1 / 0.5 ** 2) / (1 + 1 / snr_t ** 2) - 1)
    ax.axvline(snr_at, color="tab:green", ls=":", lw=1.2)
    ax.text(snr_at * 1.06, 0.06, f"min_cc 0.5\n= SNR {snr_at:.2f}", fontsize=8, color="tab:green")
    ax.set_xscale("log")
    ax.set_xlabel("injected SNR of the pick")
    ax.set_ylabel("cc of the recovered pick")
    ax.set_ylim(0, 1.05)
    ax.set_title("cc follows the two-noise law\n(low-SNR boxes are censored by min_cc)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    inj = gp[gp["injected"]].merge(gt[["inj_id", "set_id"]], on="inj_id").groupby(
        ["set_id", "phase_type"]).size()
    for ph, c in [("P", "tab:red"), ("S", "tab:blue")]:
        xs, ys = [], []
        for sid, h in tp[tp["phase_type"] == ph].groupby("set_id"):
            if (sid, ph) not in inj.index:
                continue
            xs.append(float(h["amplitude"].iloc[0]))
            ys.append(len(h) / inj[(sid, ph)])
        ax.semilogx(xs, ys, "o-", color=c, label=f"{ph}")
    ax.axvline(snr_at, color="tab:green", ls=":", lw=1.2)
    ax.set_xlabel("injected SNR of the set")
    ax.set_ylabel("fraction of injected picks recovered")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("recall vs SNR, per phase\n(the gap is the P/S template asymmetry)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(f"{figdir}/eval_cc_snr.png", dpi=160)
    plt.close(fig)
    print(f"  {figdir}/eval_cc_snr.png")


# %%
if __name__ == "__main__":
    args = parse_args()
    base = f"{args.root_path}/{args.region}"
    gt, gp, tm, ev = load(args)
    print(f"{len(gt)} injections, {len(tm)} raw TM picks, "
          f"{0 if ev is None else len(ev)} associated events")
    cc, sp = score_correlation(args, gt, gp, tm)
    labelled = score_picks(args, gt, gp, tm)
    by_set = score_by_set(gt, gp, sp, labelled)
    gt, ev = score_association(args, gt, ev)
    plot(args, gt, cc, ev, by_set, labelled)
    plot_recovery(args, gt, gp, labelled)
    plot_cc_vs_snr(args, gt, gp, labelled)
    cc.to_csv(f"{base}/benchmark/eval_correlation.csv", index=False)
    gt.to_csv(f"{base}/benchmark/eval_detections.csv", index=False)
    print(f"  {base}/benchmark/eval_correlation.csv")
    print(f"  {base}/benchmark/eval_detections.csv")
    if by_set is not None:
        by_set.to_csv(f"{base}/benchmark/eval_by_set.csv", index=False)
        print(f"  {base}/benchmark/eval_by_set.csv")
    labelled.to_csv(f"{base}/benchmark/eval_picks.csv", index=False)
    print(f"  {base}/benchmark/eval_picks.csv")

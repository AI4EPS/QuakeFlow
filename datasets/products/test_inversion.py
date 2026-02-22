# %%
"""Test inversion on all available datasets and report polarity accuracy."""
import numpy as np
import pandas as pd
from utils import load_dataset, load_waveforms
from inversion import (
    build_focal_mechanism_grid, _build_moment_tensors,
    invert_focal_mechanism, prepare_observations,
)
from polarity import calc_radiation_pattern

datasets = [
    ("NCEDC", "../NCEDC/dataset/2023/001.h5"),
    ("NCEDC", "../NCEDC/dataset/2024/001.h5"),
    ("SCEDC", "../SCEDC/dataset/2023/001.h5"),
    ("SCEDC", "../SCEDC/dataset/2024/001.h5"),
]

# Pre-build grid once
grid = build_focal_mechanism_grid(5.0, 5.0, 5.0)
M_batch = _build_moment_tensors(grid)
print(f"Grid: {len(grid)} trial mechanisms\n")

all_summaries = []

for region, data_path in datasets:
    year = data_path.split("/")[-2]
    label = f"{region} {year}/001"
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    try:
        df = load_dataset(data_path)
    except Exception as e:
        print(f"  SKIP: {e}")
        continue

    waveforms = load_waveforms(data_path)

    has_fm = df["strike"].notna() & df["dip"].notna() & df["rake"].notna()
    has_pol = df["p_phase_polarity"].isin(["U", "D"])
    fm_pol_events = df[has_fm & has_pol]["event_id"].unique()

    print(f"  Total events: {df['event_id'].nunique()}")
    print(f"  Events with FM: {df[has_fm]['event_id'].nunique()}")
    print(f"  Events with FM + polarity: {len(fm_pol_events)}")

    if len(fm_pol_events) == 0:
        print("  No events with FM + polarity, skipping.")
        continue

    # Run inversion
    results = []
    for eid in fm_pol_events:
        event_data = df[df["event_id"] == eid]
        result = invert_focal_mechanism(
            event_data, waveforms=waveforms, grid=grid, M_batch=M_batch,
            alpha=0.5, min_n_pol=6, takeoff_method="uniform",
        )
        result["catalog_strike"] = event_data["strike"].iloc[0]
        result["catalog_dip"] = event_data["dip"].iloc[0]
        result["catalog_rake"] = event_data["rake"].iloc[0]
        results.append(result)

    results_df = pd.DataFrame(results)
    ok = results_df["status"] == "OK"

    print(f"\n  Succeeded: {ok.sum()}/{len(results_df)}")

    # Per-event table
    print(f"\n  {'Event ID':<16} {'Stat':>4} {'Q':>1} | {'Catalog S/D/R':>16} | {'Inverted S/D/R':>16} | {'nP':>3} {'nSP':>3} {'mfP':>5} {'mfSP':>5}")
    print(f"  {'-'*85}")
    for _, r in results_df.iterrows():
        cat = f"{r['catalog_strike']:.0f}/{r['catalog_dip']:.0f}/{r['catalog_rake']:.0f}"
        if r["status"] == "OK":
            inv = f"{r['strike']:.0f}/{r['dip']:.0f}/{r['rake']:.0f}"
            mfp = f"{r['misfit_pol']:.3f}"
            mfs = f"{r['misfit_sp']:.3f}" if not np.isnan(r["misfit_sp"]) else "  N/A"
        else:
            inv = "N/A"
            mfp = " N/A"
            mfs = " N/A"
        q = r.get("quality", "")
        st = "OK" if r["status"] == "OK" else "FAIL"
        print(f"  {r['event_id']:<16} {st:>4} {q:>1} | {cat:>16} | {inv:>16} | {int(r['n_pol']):>3} {int(r['n_sp']):>3} {mfp:>5} {mfs:>5}")

    # Polarity accuracy (raw = all stations; null-zone = matching optimizer)
    total_inv = 0
    total_cat = 0
    total_n = 0
    total_inv_nz = 0
    total_cat_nz = 0
    total_n_nz = 0
    null_zone_fraction = 0.1

    for _, r in results_df[ok].iterrows():
        eid = r["event_id"]
        event_data = df[df["event_id"] == eid]
        obs = prepare_observations(event_data, waveforms, takeoff_method="uniform")
        if obs is None:
            continue
        pol_mask = obs["polarity_obs"] != 0
        if pol_mask.sum() == 0:
            continue
        takeoff = obs["takeoff"][pol_mask]
        azimuth = obs["azimuth"][pol_mask]
        pol_obs = obs["polarity_obs"][pol_mask]

        rad_inv = calc_radiation_pattern(r["strike"], r["dip"], r["rake"], takeoff, azimuth)
        rad_cat = calc_radiation_pattern(r["catalog_strike"], r["catalog_dip"], r["catalog_rake"], takeoff, azimuth)

        # Raw accuracy (all stations)
        correct_inv = np.sum(np.sign(rad_inv["P"]) == pol_obs)
        correct_cat = np.sum(np.sign(rad_cat["P"]) == pol_obs)
        n = len(pol_obs)
        total_inv += correct_inv
        total_cat += correct_cat
        total_n += n

        # Null-zone accuracy (exclude stations near inverted nodal planes)
        abs_P_inv = np.abs(rad_inv["P"])
        max_P_inv = np.max(abs_P_inv) if len(abs_P_inv) > 0 else 1.0
        not_nodal = abs_P_inv >= null_zone_fraction * max_P_inv
        if not_nodal.sum() > 0:
            total_inv_nz += np.sum(np.sign(rad_inv["P"][not_nodal]) == pol_obs[not_nodal])
            total_cat_nz += np.sum(np.sign(rad_cat["P"][not_nodal]) == pol_obs[not_nodal])
            total_n_nz += not_nodal.sum()

    if total_n > 0:
        inv_acc = total_inv / total_n * 100
        cat_acc = total_cat / total_n * 100
    else:
        inv_acc = cat_acc = 0.0

    if total_n_nz > 0:
        inv_acc_nz = total_inv_nz / total_n_nz * 100
        cat_acc_nz = total_cat_nz / total_n_nz * 100
    else:
        inv_acc_nz = cat_acc_nz = 0.0

    # Quality distribution
    qual_counts = {}
    for q in ["A", "B", "C", "D"]:
        n = (results_df.loc[ok, "quality"] == q).sum()
        if n > 0:
            qual_counts[q] = n

    print(f"\n  Polarity accuracy (all stations):  Inverted {total_inv}/{total_n} ({inv_acc:.1f}%) | Catalog {total_cat}/{total_n} ({cat_acc:.1f}%)")
    print(f"  Polarity accuracy (excl. nodal):   Inverted {total_inv_nz}/{total_n_nz} ({inv_acc_nz:.1f}%) | Catalog {total_cat_nz}/{total_n_nz} ({cat_acc_nz:.1f}%)")
    print(f"  Quality: {qual_counts}")

    all_summaries.append({
        "dataset": label,
        "n_events": len(fm_pol_events),
        "n_succeeded": int(ok.sum()),
        "n_pol_obs": total_n,
        "inv_accuracy": inv_acc,
        "cat_accuracy": cat_acc,
        "n_pol_nz": total_n_nz,
        "inv_accuracy_nz": inv_acc_nz,
        "cat_accuracy_nz": cat_acc_nz,
        "quality": qual_counts,
    })

# Final summary table
print(f"\n\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")
print(f"  {'Dataset':<16} {'Events':>6} {'OK':>4} | {'nPol':>5} {'Inv':>7} {'Cat':>7} | {'nPol*':>5} {'Inv*':>7} {'Cat*':>7}")
print(f"  {' '*40}  (all stations)       (excl. nodal)")
print(f"  {'-'*80}")
for s in all_summaries:
    print(f"  {s['dataset']:<16} {s['n_events']:>6} {s['n_succeeded']:>4} | {s['n_pol_obs']:>5} {s['inv_accuracy']:>6.1f}% {s['cat_accuracy']:>6.1f}% | {s['n_pol_nz']:>5} {s['inv_accuracy_nz']:>6.1f}% {s['cat_accuracy_nz']:>6.1f}%")

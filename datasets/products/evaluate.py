# %%
"""
Evaluate accuracy of predicted (FM radiation pattern) and detected (waveform first motion)
polarities against catalog polarities.

Usage:
    python evaluate.py --data_path ../NCEDC/dataset/2024/001.parquet \
        --predictions_csv ../NCEDC/dataset/2024/001.polarities.csv \
        --output_dir figures/NCEDC
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import SAMPLING_RATE, detect_first_motion, load_waveforms


# %%
def build_comparison_table(data_path, polarity_df):
    """
    Build a comparison table with predicted, detected, and catalog polarities.

    Args:
        data_path: Path to HDF5 or Parquet file (for waveform-based detection)
        polarity_df: DataFrame from predict.py output

    Returns:
        DataFrame with columns: event_id, network, station, catalog, predicted, detected,
        det_z_amp, pred_z_amp, snr, distance_km, log_sp_ratio,
        takeoff_simple, takeoff_eikonal, pred_match, det_match, pred_det_match
    """
    # Filter to entries with catalog polarity
    df = polarity_df[polarity_df["p_phase_polarity"].isin(["U", "D"])].copy()
    if len(df) == 0:
        print("No entries with catalog polarity found.")
        return pd.DataFrame()

    # Predicted label from Z-component polarity
    df["predicted"] = np.where(df["p_polarity_z"] > 0, "U", "D")
    df["catalog"] = df["p_phase_polarity"]

    # Load waveforms for detection
    print(f"Loading waveforms from {data_path}")
    waveform_dict = load_waveforms(data_path)

    # Load metadata for p_phase_index
    from utils import load_dataset
    meta = load_dataset(data_path)

    # Build index for p_phase_index lookup
    meta_key_cols = ["event_id", "network", "station", "location", "instrument"]
    meta_indexed = meta.set_index(meta_key_cols)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting from waveforms"):
        event_id = row["event_id"]
        network = row["network"]
        station = row["station"]
        location = row.get("location", "")
        instrument = row.get("instrument", "")

        # Find waveform
        waveform = None
        for key, wf in waveform_dict.items():
            if key[0] == event_id and key[1] == network and key[2] == station:
                waveform = wf
                location = key[3]
                instrument = key[4]
                break

        if waveform is None:
            continue

        # Get p_phase_index from metadata
        try:
            meta_row = meta_indexed.loc[(event_id, network, station, location, instrument)]
            if isinstance(meta_row, pd.DataFrame):
                meta_row = meta_row.iloc[0]
            p_index = meta_row.get("p_phase_index", None)
            snr = meta_row.get("snr", np.nan)
            distance_km = meta_row.get("distance_km", np.nan)
        except KeyError:
            continue

        det = detect_first_motion(waveform, p_index)

        records.append({
            "event_id": event_id,
            "network": network,
            "station": station,
            "catalog": row["catalog"],
            "predicted": row["predicted"],
            "detected": det["label"] if det and det["label"] else None,
            "det_z_amp": det["z"] if det else None,
            "pred_z_amp": row["p_polarity_z"],
            "snr": float(snr) if not np.isnan(snr) else 0,
            "distance_km": float(distance_km) if not np.isnan(distance_km) else 0,
            "log_sp_ratio": row.get("log_sp_ratio", np.nan),
            "takeoff_simple": row.get("takeoff_simple", np.nan),
            "takeoff_eikonal": row.get("takeoff_eikonal", np.nan),
        })

    result = pd.DataFrame(records)
    if len(result) > 0:
        result["pred_match"] = result["predicted"] == result["catalog"]
        result["det_match"] = result["detected"] == result["catalog"]
        result["pred_det_match"] = result["predicted"] == result["detected"]

    return result


# %%
def print_summary(comp):
    """Print accuracy summary statistics."""
    print("=" * 60)
    print("POLARITY ACCURACY SUMMARY")
    print("=" * 60)
    print(f"Total entries with catalog polarity: {len(comp)}")

    # Overall match rates
    print(f"\nPredicted (FM) vs Catalog:")
    n = comp["pred_match"].sum()
    print(f"  Accuracy: {n}/{len(comp)} = {n/len(comp)*100:.1f}%")

    valid_det = comp["detected"].notna()
    if valid_det.sum() > 0:
        n = comp.loc[valid_det, "det_match"].sum()
        total = valid_det.sum()
        print(f"\nDetected (WF) vs Catalog:")
        print(f"  Accuracy: {n}/{total} = {n/total*100:.1f}%")

        n = comp.loc[valid_det, "pred_det_match"].sum()
        print(f"\nPredicted vs Detected:")
        print(f"  Agreement: {n}/{total} = {n/total*100:.1f}%")

    # Breakdown by catalog polarity
    print(f"\nBreakdown by catalog polarity:")
    for label in ["U", "D"]:
        subset = comp[comp["catalog"] == label]
        if len(subset) == 0:
            continue
        pred_acc = subset["pred_match"].mean() * 100
        det_acc = subset.loc[subset["detected"].notna(), "det_match"].mean() * 100 if subset["detected"].notna().any() else 0
        print(f"  {label}: n={len(subset)}, pred={pred_acc:.1f}%, det={det_acc:.1f}%")

    # Accuracy by SNR bins
    print(f"\nAccuracy by SNR:")
    snr_bins = [(0, 2), (2, 5), (5, 10), (10, 50), (50, 1000)]
    for lo, hi in snr_bins:
        subset = comp[(comp["snr"] >= lo) & (comp["snr"] < hi)]
        if len(subset) == 0:
            continue
        pred_acc = subset["pred_match"].mean() * 100
        det_acc = subset.loc[subset["detected"].notna(), "det_match"].mean() * 100 if subset["detected"].notna().any() else 0
        print(f"  SNR [{lo:>3},{hi:>4}): n={len(subset):>4}, pred={pred_acc:.1f}%, det={det_acc:.1f}%")

    # Accuracy by distance bins
    print(f"\nAccuracy by distance:")
    dist_bins = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500)]
    for lo, hi in dist_bins:
        subset = comp[(comp["distance_km"] >= lo) & (comp["distance_km"] < hi)]
        if len(subset) == 0:
            continue
        pred_acc = subset["pred_match"].mean() * 100
        det_acc = subset.loc[subset["detected"].notna(), "det_match"].mean() * 100 if subset["detected"].notna().any() else 0
        print(f"  Dist [{lo:>3},{hi:>3}) km: n={len(subset):>4}, pred={pred_acc:.1f}%, det={det_acc:.1f}%")

    # Takeoff angle comparison
    if "takeoff_simple" in comp.columns and "takeoff_eikonal" in comp.columns:
        valid_both = comp["takeoff_simple"].notna() & comp["takeoff_eikonal"].notna()
        if valid_both.any():
            diff = comp.loc[valid_both, "takeoff_simple"] - comp.loc[valid_both, "takeoff_eikonal"]
            print(f"\nTakeoff angle (simple - eikonal):")
            print(f"  Mean diff: {diff.mean():.2f} deg, Std: {diff.std():.2f} deg, Max: {diff.abs().max():.2f} deg")

    print("=" * 60)


# %%
def plot_analysis(comp, data_path, output_dir):
    """Generate analysis plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Match rates per event ---
    event_stats = []
    for eid, grp in comp.groupby("event_id"):
        pred_acc = grp["pred_match"].mean()
        det_valid = grp["detected"].notna()
        det_acc = grp.loc[det_valid, "det_match"].mean() if det_valid.any() else np.nan
        event_stats.append({"event_id": eid, "n": len(grp), "pred_acc": pred_acc, "det_acc": det_acc})
    stats_df = pd.DataFrame(event_stats).sort_values("pred_acc", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = range(len(stats_df))
    labels = [f"{e[:12]}\n(n={n})" for e, n in zip(stats_df["event_id"], stats_df["n"])]

    for ax, col, title in zip(axes, ["pred_acc", "det_acc"],
                               ["Predicted (FM) vs Catalog", "Detected (WF) vs Catalog"]):
        vals = stats_df[col].fillna(0)
        colors = ["green" if v > 0.7 else "orange" if v > 0.5 else "red" for v in vals]
        ax.bar(x, vals * 100, color=colors, edgecolor="k", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Match Rate (%)")
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("P-wave Polarity Match Rate per Event", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/match_rates_per_event.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir}/match_rates_per_event.png")

    # --- Figure 2: Accuracy vs SNR ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    valid = comp["detected"].notna()
    for ax, match_col, title in zip(axes, ["pred_match", "det_match"],
                                     ["Predicted vs Catalog", "Detected vs Catalog"]):
        subset = comp[valid] if match_col == "det_match" else comp
        ax.scatter(subset["snr"], subset[match_col].astype(float) + np.random.uniform(-0.05, 0.05, len(subset)),
                  c=subset[match_col].map({True: "green", False: "red"}), alpha=0.3, s=15)
        snr_edges = [0, 2, 5, 10, 20, 50, 100, 500]
        bin_centers = []
        bin_accs = []
        for i in range(len(snr_edges) - 1):
            mask = (subset["snr"] >= snr_edges[i]) & (subset["snr"] < snr_edges[i+1])
            if mask.sum() > 0:
                bin_centers.append((snr_edges[i] + snr_edges[i+1]) / 2)
                bin_accs.append(subset.loc[mask, match_col].mean())
        ax.plot(bin_centers, bin_accs, "b-o", lw=2, markersize=6)
        ax.set_xlabel("SNR")
        ax.set_ylabel("Match (1=correct, 0=wrong)")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir}/accuracy_vs_snr.png")

    # --- Figure 3: Accuracy vs Distance ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, match_col, title in zip(axes, ["pred_match", "det_match"],
                                     ["Predicted vs Catalog", "Detected vs Catalog"]):
        subset = comp[valid] if match_col == "det_match" else comp
        ax.scatter(subset["distance_km"], subset[match_col].astype(float) + np.random.uniform(-0.05, 0.05, len(subset)),
                  c=subset[match_col].map({True: "green", False: "red"}), alpha=0.3, s=15)
        dist_edges = [0, 20, 50, 100, 150, 200, 300, 500]
        bin_centers = []
        bin_accs = []
        for i in range(len(dist_edges) - 1):
            mask = (subset["distance_km"] >= dist_edges[i]) & (subset["distance_km"] < dist_edges[i+1])
            if mask.sum() > 0:
                bin_centers.append((dist_edges[i] + dist_edges[i+1]) / 2)
                bin_accs.append(subset.loc[mask, match_col].mean())
        ax.plot(bin_centers, bin_accs, "b-o", lw=2, markersize=6)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Match (1=correct, 0=wrong)")
        ax.set_title(title)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_vs_distance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir}/accuracy_vs_distance.png")

    # --- Figure 4: Confusion matrices ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    pairs = [
        ("predicted", "catalog", "Predicted vs Catalog"),
        ("detected", "catalog", "Detected vs Catalog"),
        ("predicted", "detected", "Predicted vs Detected"),
    ]
    for ax, (col_a, col_b, title) in zip(axes, pairs):
        subset = comp.dropna(subset=[col_a, col_b])
        labels = ["U", "D"]
        matrix = np.zeros((2, 2), dtype=int)
        for i, la in enumerate(labels):
            for j, lb in enumerate(labels):
                matrix[i, j] = ((subset[col_a] == la) & (subset[col_b] == lb)).sum()

        im = ax.imshow(matrix, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14,
                       color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels)
        ax.set_xlabel(col_b.capitalize())
        ax.set_ylabel(col_a.capitalize())
        ax.set_title(f"{title}\n(acc={subset[col_a].eq(subset[col_b]).mean()*100:.1f}%)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir}/confusion_matrices.png")

    # --- Figure 5: Waveform examples (matches and mismatches) ---
    plot_waveform_examples(comp, data_path, output_dir)

    # --- Figure 6: Takeoff angle comparison ---
    if "takeoff_simple" in comp.columns and "takeoff_eikonal" in comp.columns:
        valid_both = comp["takeoff_simple"].notna() & comp["takeoff_eikonal"].notna()
        if valid_both.sum() > 10:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sub = comp[valid_both]

            axes[0].scatter(sub["takeoff_simple"], sub["takeoff_eikonal"],
                          c=sub["pred_match"].map({True: "green", False: "red"}), alpha=0.3, s=15)
            axes[0].plot([0, 180], [0, 180], "k--", alpha=0.5)
            axes[0].set_xlabel("Takeoff Simple (deg)")
            axes[0].set_ylabel("Takeoff Eikonal (deg)")
            axes[0].set_title("Takeoff Angle Comparison")
            axes[0].set_aspect("equal")

            diff = sub["takeoff_simple"] - sub["takeoff_eikonal"]
            axes[1].hist(diff, bins=50, edgecolor="k", alpha=0.7)
            axes[1].set_xlabel("Simple - Eikonal (deg)")
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"Takeoff Diff (mean={diff.mean():.1f}, std={diff.std():.1f})")
            axes[1].axvline(0, color="red", ls="--")

            plt.tight_layout()
            plt.savefig(f"{output_dir}/takeoff_comparison.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved {output_dir}/takeoff_comparison.png")


# %%
def plot_waveform_examples(comp, data_path, output_dir, n_examples=12):
    """Plot waveform examples: matches and mismatches with catalog."""
    valid = comp["detected"].notna()
    subset = comp[valid].copy()
    if len(subset) == 0:
        return

    # Pick some matches and mismatches
    matches = subset[subset["pred_match"]].sort_values("snr", ascending=False).head(n_examples // 2)
    mismatches = subset[~subset["pred_match"]].sort_values("snr", ascending=False).head(n_examples // 2)
    examples = pd.concat([matches, mismatches])

    n_plot = min(len(examples), n_examples)
    if n_plot == 0:
        return

    # Load waveforms
    pairs = set((r["event_id"], r["network"], r["station"]) for _, r in examples.iterrows())
    waveform_dict = load_waveforms(data_path, pairs)

    # Get p_phase_index from metadata
    from utils import load_dataset
    meta = load_dataset(data_path)
    meta_key_cols = ["event_id", "network", "station", "location", "instrument"]
    meta_indexed = meta.set_index(meta_key_cols)

    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 2.5 * n_plot))
    if n_plot == 1:
        axes = [axes]

    plot_idx = 0
    for _, row in examples.head(n_plot).iterrows():
        ax = axes[plot_idx]
        event_id = row["event_id"]
        network = row["network"]
        station = row["station"]

        # Find waveform
        wf = None
        p_idx = None
        for key, w in waveform_dict.items():
            if key[0] == event_id and key[1] == network and key[2] == station:
                wf = w
                try:
                    meta_row = meta_indexed.loc[(key[0], key[1], key[2], key[3], key[4])]
                    if isinstance(meta_row, pd.DataFrame):
                        meta_row = meta_row.iloc[0]
                    p_idx = meta_row.get("p_phase_index", None)
                except KeyError:
                    pass
                break

        if wf is None or p_idx is None:
            plot_idx += 1
            continue

        p_idx = int(p_idx)
        t_before, t_after = 1.0, 3.0
        idx_start = max(0, int(p_idx - t_before * SAMPLING_RATE))
        idx_end = min(wf.shape[1], int(p_idx + t_after * SAMPLING_RATE))
        time_arr = (np.arange(idx_start, idx_end) - p_idx) / SAMPLING_RATE

        components = ["E", "N", "Z"]
        for c_idx, (cmp, color) in enumerate(zip(components, ["#1f77b4", "#ff7f0e", "#2ca02c"])):
            trace = wf[c_idx, idx_start:idx_end]
            max_val = np.max(np.abs(trace)) if np.max(np.abs(trace)) > 0 else 1
            ax.plot(time_arr, trace / max_val + c_idx * 2.5, color=color, lw=0.8, label=cmp)

        ax.axvline(0, color="red", linestyle="--", lw=1, alpha=0.7)

        cat = row["catalog"]
        pred = row["predicted"]
        det = row["detected"]
        title = f"{event_id} | {network}.{station} | SNR={row['snr']:.1f} | dist={row['distance_km']:.0f}km"
        title += f" | Cat={cat} Pred={pred} Det={det}"

        if pred == cat == det:
            title += " [ALL AGREE]"
            ax.set_facecolor("#e6ffe6")
        elif pred == cat != det:
            title += " [pred=cat, det wrong]"
            ax.set_facecolor("#fff3e6")
        elif det == cat != pred:
            title += " [det=cat, pred wrong]"
            ax.set_facecolor("#fff3e6")
        elif pred != cat and det != cat:
            title += " [both wrong]"
            ax.set_facecolor("#ffe6e6")
        else:
            ax.set_facecolor("#ffe6e6")

        ax.set_title(title, fontsize=9)
        ax.set_yticks([0, 2.5, 5.0])
        ax.set_yticklabels(components)
        ax.set_xlim(-t_before, t_after)
        if plot_idx == 0:
            ax.legend(loc="upper right", fontsize=7)
        if plot_idx == n_plot - 1:
            ax.set_xlabel("Time relative to P arrival (s)")

        plot_idx += 1

    plt.tight_layout()
    plt.savefig(f"{output_dir}/waveform_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir}/waveform_examples.png")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate polarity prediction accuracy")
    parser.add_argument("--data_path", type=str, required=True, help="Path to HDF5 or Parquet dataset file")
    parser.add_argument("--predictions_csv", type=str, required=True, help="Path to predictions CSV from predict.py")
    parser.add_argument("--output_dir", type=str, default="figures", help="Output directory for plots")
    parser.add_argument("--output_csv", type=str, default=None, help="Save comparison table to CSV")
    args = parser.parse_args()

    data_path = Path(args.data_path)

    # Load predictions
    polarity_df = pd.read_csv(args.predictions_csv)
    print(f"Loaded {len(polarity_df)} polarity predictions")

    # Build comparison table
    comp = build_comparison_table(data_path, polarity_df)
    if len(comp) == 0:
        print("No data to analyze.")
        exit()

    # Print summary
    print_summary(comp)

    # Save comparison CSV
    csv_path = args.output_csv or str(data_path.with_suffix(".comparison.csv"))
    comp.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to {csv_path}")

    # Generate plots
    print("\n--- Generating plots ---")
    plot_analysis(comp, data_path, args.output_dir)
    print(f"\nAll plots saved to {args.output_dir}/")

# %%

# %%
"""
Evaluate prediction vs detection consistency per component (E, N, Z)
for both P and S wave arrivals.

Uses inverted focal mechanisms (focal_mechanisms.csv) for FM predictions.
Loads the full dataset from parquet so we evaluate ALL records.

For each station:
  - Prediction: sign of radiation pattern amplitude per component (from inverted FM)
  - Detection: sign of waveform first-motion per component
  - Catalog: phase polarity label from the dataset (Z-component only, P-wave)

Reports P-wave and S-wave results separately, E/N/Z per component.

Usage:
    python evaluate_components.py --data_path ../NCEDC/dataset/2024/001.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from eikonal2d import calc_ray_param
from polarity import calc_radiation_pattern
from tqdm import tqdm
from utils import load_dataset, load_waveforms  # noqa: F401


SAMPLING_RATE = 100.0


def compute_snr(waveform, phase_index, noise_window_s=1.0, signal_window_s=1.0):
    """Compute SNR for a phase arrival using the Z component.

    SNR = max(|signal|) / rms(noise), where noise is before the arrival
    and signal is after.
    """
    if phase_index is None or (isinstance(phase_index, float) and np.isnan(phase_index)):
        return np.nan
    phase_index = int(phase_index)
    noise_samples = int(noise_window_s * SAMPLING_RATE)
    signal_samples = int(signal_window_s * SAMPLING_RATE)
    if phase_index < noise_samples or phase_index + signal_samples >= waveform.shape[1]:
        return np.nan
    # Use vector norm across 3 components
    noise = waveform[:, phase_index - noise_samples : phase_index]
    signal = waveform[:, phase_index : phase_index + signal_samples]
    noise_rms = np.sqrt(np.mean(np.sum(noise**2, axis=0)))
    signal_max = np.max(np.sqrt(np.sum(signal**2, axis=0)))
    if noise_rms < 1e-15:
        return np.nan
    return float(signal_max / noise_rms)


def detect_component_motion(waveform, phase_index, pre_window=3, post_window=10):
    """Detect polarity from waveform at a phase arrival, per component.

    Uses the same first-extremum logic as detect_first_motion() in utils.py,
    applied independently to each component.

    Returns:
        dict with 'e', 'n', 'z' amplitude values, or None if not possible.
    """
    if phase_index is None or (isinstance(phase_index, float) and np.isnan(phase_index)):
        return None
    phase_index = int(phase_index)
    if phase_index < pre_window or phase_index + post_window >= waveform.shape[1]:
        return None

    result = {}
    for i, comp in enumerate(["e", "n", "z"]):
        pre_mean = np.mean(waveform[i, phase_index - pre_window : phase_index])
        post = waveform[i, phase_index : phase_index + post_window]
        diff = post - pre_mean
        # First-extremum search (same logic as utils.detect_first_motion)
        amplitude = 0.0
        for ii in range(len(diff) - 1):
            s = diff[ii]
            if s <= 0:
                if s < diff[ii + 1] or ii == len(diff) - 2:
                    if ii == 0:
                        break
                    amplitude = float(s)
                    break
            else:
                if s > diff[ii + 1] or ii == len(diff) - 2:
                    if ii == 0:
                        break
                    amplitude = float(s)
                    break
        result[comp] = amplitude
    return result


def resolve_out_dir(data_path):
    """Resolve output directory from data path."""
    resolved = Path(data_path).resolve()
    parts = resolved.parts
    try:
        ds_idx = parts.index("dataset")
        region = parts[ds_idx - 1]
        sub_path = Path(*parts[ds_idx + 1 :]).parent / Path(data_path).stem
        return Path(__file__).parent / region / sub_path
    except ValueError:
        return Path(__file__).parent / Path(data_path).stem


def evaluate_dataset(data_path, fm_path=None):
    """Evaluate prediction vs detection consistency for one dataset.

    Args:
        data_path: Path to parquet/HDF5 dataset.
        fm_path: Path to inverted focal_mechanisms.csv. Auto-detected if None.

    Returns DataFrame with per-record, per-component, per-phase results.
    """
    phases = load_dataset(data_path)
    waveforms = load_waveforms(data_path)

    # Build waveform lookup by (event_id, network, station)
    wf_lookup = {}
    for key, wf in waveforms.items():
        short_key = (key[0], key[1], key[2])
        wf_lookup[short_key] = (wf, key)

    # Load inverted FM
    if fm_path is None:
        fm_path = resolve_out_dir(data_path) / "focal_mechanisms.csv"
    fm_path = Path(fm_path)

    if fm_path.exists():
        fm_df = pd.read_csv(fm_path)[["event_id", "strike", "dip", "rake"]]
        print(f"Loaded {len(fm_df)} inverted FMs from {fm_path}")
        # Replace catalog FM with inverted FM
        phases = phases.drop(columns=["strike", "dip", "rake"], errors="ignore")
        phases = phases.merge(fm_df, on="event_id", how="left")
    else:
        print(f"WARNING: No inverted FM file at {fm_path}, using catalog FM")

    # Compute FM predictions for records with strike/dip/rake
    fm_mask = phases["strike"].notna() & phases["dip"].notna() & phases["rake"].notna()
    has_fm = phases[fm_mask].copy()

    # Compute ray parameters
    ray = calc_ray_param(
        has_fm["event_longitude"].values,
        has_fm["event_latitude"].values,
        has_fm["event_depth_km"].values,
        has_fm["station_longitude"].values,
        has_fm["station_latitude"].values,
        has_fm["station_depth_km"].values,
        np.zeros(len(has_fm), dtype=int),
        None,
    )
    has_fm["azimuth_calc"] = ray["azimuth"]
    has_fm["takeoff_calc"] = ray["takeoff"]

    # Compute radiation patterns per event
    pred_map = {}  # (event_id, network, station) -> {p_e, p_n, p_z, s_e, s_n, s_z}
    for event_id, group in has_fm.groupby("event_id"):
        strike = group["strike"].iloc[0]
        dip = group["dip"].iloc[0]
        rake = group["rake"].iloc[0]

        radiation = calc_radiation_pattern(
            strike, dip, rake,
            group["takeoff_calc"].values, group["azimuth_calc"].values,
        )

        P_ENZ = radiation["P_ENZ"]
        S_ENZ = radiation["S_ENZ"]
        for i, (_, row) in enumerate(group.iterrows()):
            key = (row["event_id"], row["network"], row["station"])
            pred_map[key] = {
                "p_e": P_ENZ[i, 0], "p_n": P_ENZ[i, 1], "p_z": P_ENZ[i, 2],
                "s_e": S_ENZ[i, 0], "s_n": S_ENZ[i, 1], "s_z": S_ENZ[i, 2],
            }

    n_fm_events = has_fm["event_id"].nunique()
    n_cat_pol = phases["p_phase_polarity"].isin(["U", "D"]).sum()
    n_with_s = phases["s_phase_index"].notna().sum()
    print(f"FM predictions: {len(pred_map)} stations from {n_fm_events} events")
    print(f"Total records: {len(phases)}, catalog polarity: {n_cat_pol}, with S pick: {n_with_s}")

    # Filter to records with catalog polarity labels
    phases = phases[phases["p_phase_polarity"].isin(["U", "D"])].copy()
    print(f"After filtering to labeled records: {len(phases)}")

    records = []
    for _, row in tqdm(phases.iterrows(), total=len(phases), desc="Evaluating"):
        short_key = (row["event_id"], row["network"], row["station"])
        entry = wf_lookup.get(short_key)
        if entry is None:
            continue
        wf, full_key = entry

        p_idx = row.get("p_phase_index")
        s_idx = row.get("s_phase_index")
        pred = pred_map.get(short_key)
        catalog_pol = row["p_phase_polarity"]

        base = {
            "event_id": row["event_id"],
            "network": row["network"],
            "station": row["station"],
            "catalog_polarity": catalog_pol,
            "distance_km": row.get("distance_km", np.nan),
            "has_fm": pred is not None,
        }

        # --- P wave ---
        if p_idx is not None and not (isinstance(p_idx, float) and np.isnan(p_idx)):
            det = detect_component_motion(wf, p_idx)
            if det is not None:
                p_snr = compute_snr(wf, p_idx)
                for comp in ["e", "n", "z"]:
                    det_val = det[comp]
                    det_sign = "U" if det_val > 0 else "D" if det_val < 0 else ""
                    pred_val = pred[f"p_{comp}"] if pred else np.nan
                    pred_sign = ("U" if pred_val > 0 else "D" if pred_val < 0 else "") if pred else ""
                    records.append({
                        **base,
                        "phase": "P",
                        "component": comp.upper(),
                        "snr": p_snr,
                        "pred_amplitude": pred_val if pred else np.nan,
                        "det_amplitude": det_val,
                        "pred_sign": pred_sign,
                        "det_sign": det_sign,
                    })

        # --- S wave ---
        if s_idx is not None and not (isinstance(s_idx, float) and np.isnan(s_idx)):
            det_s = detect_component_motion(wf, s_idx, pre_window=3, post_window=15)
            if det_s is not None:
                s_snr = compute_snr(wf, s_idx)
                for comp in ["e", "n", "z"]:
                    det_val = det_s[comp]
                    det_sign = "U" if det_val > 0 else "D" if det_val < 0 else ""
                    pred_val = pred[f"s_{comp}"] if pred else np.nan
                    pred_sign = ("U" if pred_val > 0 else "D" if pred_val < 0 else "") if pred else ""
                    records.append({
                        **base,
                        "phase": "S",
                        "component": comp.upper(),
                        "snr": s_snr,
                        "pred_amplitude": pred_val if pred else np.nan,
                        "det_amplitude": det_val,
                        "pred_sign": pred_sign,
                        "det_sign": det_sign,
                    })

    return pd.DataFrame(records)


def _print_phase_section(results, phase, label):
    """Print a report section for one phase (P or S)."""
    sub = results[results["phase"] == phase]
    if len(sub) == 0:
        print(f"\n  No {phase}-wave records")
        return

    fm = sub[sub["has_fm"]].copy()
    n_total = len(sub) // 3  # 3 components per record
    n_fm = len(fm) // 3

    print(f"\n  ── {phase}-WAVE ({n_total} stations, {n_fm} with inverted FM) ──")

    # Prediction vs Detection per component (FM subset)
    if len(fm) > 0:
        fm["pred_det_match"] = fm["pred_sign"] == fm["det_sign"]
        print(f"\n  Prediction vs Detection (inverted FM):")
        print(f"  {'Comp':<5} {'N':>6} {'Match':>6} {'Rate':>8}")
        print(f"  {'-'*28}")
        for comp in ["E", "N", "Z"]:
            c = fm[fm["component"] == comp]
            if len(c) == 0:
                continue
            n_m = c["pred_det_match"].sum()
            print(f"  {comp:<5} {len(c):>6} {n_m:>6} {n_m/len(c)*100:>7.1f}%")
        n_m = fm["pred_det_match"].sum()
        print(f"  {'ALL':<5} {len(fm):>6} {n_m:>6} {n_m/len(fm)*100:>7.1f}%")

    # Detection vs Catalog (P-wave Z only)
    if phase == "P":
        cat_z = sub[(sub["catalog_polarity"].isin(["U", "D"])) & (sub["component"] == "Z")].copy()
        if len(cat_z) > 0:
            det_valid = cat_z["det_sign"].isin(["U", "D"])
            det_z = cat_z[det_valid]
            print(f"\n  Detection Z vs Catalog Label ({len(cat_z)} with label, {len(det_z)} detected):")
            if len(det_z) > 0:
                det_match = (det_z["det_sign"] == det_z["catalog_polarity"]).sum()
                print(f"  Det Z vs Catalog:  {det_match}/{len(det_z)} = {det_match/len(det_z)*100:.1f}%")

                # Also pred vs catalog for FM subset
                fm_det = det_z[det_z["has_fm"]]
                if len(fm_det) > 0:
                    pred_match = (fm_det["pred_sign"] == fm_det["catalog_polarity"]).sum()
                    print(f"  Pred Z vs Catalog: {pred_match}/{len(fm_det)} = {pred_match/len(fm_det)*100:.1f}%")

                # Breakdown by label
                for lab in ["U", "D"]:
                    s = det_z[det_z["catalog_polarity"] == lab]
                    if len(s) > 0:
                        n_m = (s["det_sign"] == lab).sum()
                        print(f"    {lab}: {n_m}/{len(s)} = {n_m/len(s)*100:.1f}%")

        # Prediction vs Catalog per component (FM subset)
        fm_cat = sub[(sub["catalog_polarity"].isin(["U", "D"])) & sub["has_fm"]].copy()
        if len(fm_cat) > 0:
            print(f"\n  Prediction vs Catalog per component (FM subset):")
            print(f"  {'Comp':<5} {'N':>6} {'Pred-Cat':>10} {'Det-Cat':>10} {'Pred-Det':>10}")
            print(f"  {'-'*47}")
            for comp in ["E", "N", "Z"]:
                c = fm_cat[fm_cat["component"] == comp]
                if len(c) == 0:
                    continue
                det_ok = c["det_sign"].isin(["U", "D"])
                pc = (c["pred_sign"] == c["catalog_polarity"]).mean() * 100
                dc = (c.loc[det_ok, "det_sign"] == c.loc[det_ok, "catalog_polarity"]).mean() * 100 if det_ok.any() else float("nan")
                pd_ = (c["pred_sign"] == c["det_sign"]).mean() * 100
                dc_s = f"{dc:.1f}%" if not np.isnan(dc) else "  n/a"
                print(f"  {comp:<5} {len(c):>6} {pc:>9.1f}% {dc_s:>10} {pd_:>9.1f}%")

    # SNR breakdown (for FM subset, per component)
    fm_sub = sub[sub["has_fm"]].copy()
    if len(fm_sub) > 0 and "snr" in fm_sub.columns:
        fm_sub["pred_det_match"] = fm_sub["pred_sign"] == fm_sub["det_sign"]
        # For P-wave: catalog polarity is valid; for S-wave: only Pred-Det
        show_cat = phase == "P"
        snr_bins = [(0, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 1000)]

        for comp in ["E", "N", "Z"]:
            fm_c = fm_sub[fm_sub["component"] == comp]
            if len(fm_c) == 0:
                continue
            print(f"\n  Accuracy by SNR ({phase}-wave {comp}, FM subset):")
            header = f"  {'SNR':>12} {'N':>5}"
            if show_cat:
                header += f" {'Pred-Cat':>10} {'Det-Cat':>10}"
            header += f" {'Pred-Det':>10}"
            print(header)
            print(f"  {'-'*len(header)}")
            for lo, hi in snr_bins:
                mask = (fm_c["snr"] >= lo) & (fm_c["snr"] < hi)
                s = fm_c[mask]
                if len(s) == 0:
                    continue
                pd_rate = s["pred_det_match"].mean() * 100
                line = f"  {'['+str(lo)+','+str(hi)+')':>12} {len(s):>5}"
                if show_cat:
                    sc = s[s["catalog_polarity"].isin(["U", "D"])]
                    if len(sc) > 0:
                        pc = (sc["pred_sign"] == sc["catalog_polarity"]).mean() * 100
                        det_ok = sc["det_sign"].isin(["U", "D"])
                        dc = (sc.loc[det_ok, "det_sign"] == sc.loc[det_ok, "catalog_polarity"]).mean() * 100 if det_ok.any() else float("nan")
                        dc_s = f"{dc:.1f}%" if not np.isnan(dc) else "  n/a"
                        line += f" {pc:>9.1f}% {dc_s:>10}"
                    else:
                        line += f" {'n/a':>10} {'n/a':>10}"
                line += f" {pd_rate:>9.1f}%"
                print(line)


def print_report(results, label=""):
    """Print consistency report with P and S sections."""
    if len(results) == 0:
        print(f"No results for {label}")
        return

    header = f"  {label}" if label else ""
    print(f"\n{'='*70}")
    print(f"  PREDICTION vs DETECTION CONSISTENCY{header}")
    print(f"{'='*70}")

    _print_phase_section(results, "P", label)
    _print_phase_section(results, "S", label)

    print(f"\n{'='*70}")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction vs detection per component")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--fm_path", type=str, default=None,
                        help="Path to inverted focal_mechanisms.csv (auto-detected if not given)")
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir = resolve_out_dir(data_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_path}")
    results = evaluate_dataset(data_path, fm_path=args.fm_path)

    label = f"{data_path.parts[-3]}/{data_path.parts[-2]}/{data_path.stem}" if len(data_path.parts) >= 3 else str(data_path)
    print_report(results, label)

    out_csv = args.output_csv or str(out_dir / "component_consistency.csv")
    if len(results) > 0:
        results.to_csv(out_csv, index=False)
        print(f"\nSaved {len(results)} records to {out_csv}")

# %%

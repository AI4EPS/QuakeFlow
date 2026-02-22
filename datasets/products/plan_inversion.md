# Plan: Add `inversion.py` — Focal Mechanism Grid Search Inversion

## Context

The `datasets/products/` pipeline currently supports **forward modeling** (given a known focal mechanism, predict polarities and S/P ratios at stations via `predict.py` + `polarity.py`) and **evaluation** (`evaluate.py`). The missing piece is the **inverse problem**: given observed P-wave polarities and S/P amplitude ratios, find the best-fit focal mechanism (strike/dip/rake) via grid search.

The approach follows HASH (Hardebeck & Shearer 2002) for polarity handling (null-zone exclusion near nodal planes, fractional misfit) and extends it with S/P ratio constraints. The key challenge is balancing the two data types and handling outlier (false) polarity picks robustly.

## Files to Create

- [datasets/products/inversion.py](datasets/products/inversion.py) — new file (~400 lines)

## Files to Reuse (read-only)

- [polarity.py](datasets/products/polarity.py) — `calc_radiation_pattern()` for forward model (moment tensor formulation at lines 46-62)
- [utils.py](datasets/products/utils.py) — `load_dataset()`, `load_waveforms()`, `detect_first_motion()`, `SAMPLING_RATE`
- [eikonal2d.py](datasets/products/eikonal2d.py) — `calc_ray_param()` for takeoff/azimuth recomputation
- [predict.py](datasets/products/predict.py) — reference for TauP velocity model and per-event loop pattern

## Implementation

### 1. `build_focal_mechanism_grid(strike_step=5, dip_step=5, rake_step=5)`

Enumerate all trial (strike, dip, rake) combinations:
- strike: [0, 360) step 5 → 72 values
- dip: [0, 90] step 5 → 19 values
- rake: [-180, 180) step 5 → 72 values
- Total: ~98,496 mechanisms → (N_mech, 3) array, ~1 MB

### 2. `measure_sp_ratio(waveform, p_index, s_index, ...)`

Measure observed log10(S/P) from 3-component waveforms:
- **P amplitude**: max of vector norm `sqrt(E^2+N^2+Z^2)` in window [p_index, p_index + 0.5s]
- **S amplitude**: max of vector norm in window [s_index, s_index + 1.0s]
- **SNR gate**: require P SNR > 3.0 (vs pre-P noise) to return valid measurement
- Returns `log_SP_obs` (float) or `None` if invalid

Using vector norm (L2 across 3 components) avoids sensitivity to unknown instrument orientation and captures full S energy regardless of SV/SH partition.

### 3. `polarity_misfit(P_predicted, polarity_obs, weights, null_zone_fraction=0.1)`

HASH-style weighted fractional misfit with null-zone exclusion:

```
For each mechanism k, each station s:
  1. If |P_pred[k,s]| < null_zone_fraction * max_s(|P_pred[k,s]|):
     → station is near a nodal plane, EXCLUDE from scoring
  2. If sign(P_pred[k,s]) != polarity_obs[s]:
     → mismatch, add weight[s] to penalty
  3. misfit[k] = sum(mismatch_weights) / sum(valid_weights)
```

**Null zone** is the primary outlier robustness mechanism: stations near nodal planes have ambiguous predicted polarity due to small errors in takeoff angle, so excluding them prevents false penalties. This naturally tolerates ~10-20% false picks since the misfit is a fraction, not a count.

**Weights**: `clip(snr / 10, 0.1, 3.0)` — high-SNR stations are more trustworthy, but capped at 3x to prevent single stations from dominating.

Input/output shapes: `P_predicted` (N_mech, N_sta), `polarity_obs` (N_sta,) encoded +1/-1/0, returns (N_mech,).

### 4. `sp_ratio_misfit(log_SP_predicted, log_SP_obs, weights, robust=True)`

Log-space absolute deviation (L1, more outlier-robust than L2):

```
valid = ~isnan(log_SP_obs)
residuals = clip(log_SP_pred[:, valid], -2, 3) - log_SP_obs[valid]
misfit[k] = weighted_mean(|residuals[k]|, weights[valid])
```

**Clipping** predicted log_SP to [-2, 3]: near P nodal planes, theoretical S/P → infinity, which is both numerically problematic and physically unreliable. Clipping prevents these from dominating.

**Same null-zone exclusion** as polarity: stations where |P_pred| is small get zero weight for S/P scoring too, since both numerator measurement and denominator prediction are unreliable near nodal planes.

Input/output shapes: `log_SP_predicted` (N_mech, N_sta), `log_SP_obs` (N_sta,), returns (N_mech,).

### 5. `combined_misfit(misfit_pol, misfit_sp, alpha=0.5)`

Weighted combination with percentile normalization:

```
pol_norm = (misfit_pol - p5_pol) / (p95_pol - p5_pol + eps)
sp_norm  = (misfit_sp - p5_sp) / (p95_sp - p5_sp + eps)
total = alpha * pol_norm + (1 - alpha) * sp_norm
```

- Polarity misfit is in [0,1] (fraction); S/P misfit is in log10 units (typically 0.1-1.0)
- 5th-95th percentile normalization makes them scale-comparable without being sensitive to extreme outliers
- When `n_sp == 0`: automatically `alpha = 1.0` (polarity only)
- When `n_pol == 0`: automatically `alpha = 0.0` (S/P only)
- Default `alpha=0.5` balances both equally

### 6. Vectorized Forward Model (core performance)

Instead of calling `calc_radiation_pattern` per mechanism in a loop, **vectorize over the entire grid at once**:

```python
# Pre-compute moment tensors for ALL grid mechanisms: (N_mech, 3, 3)
M_batch[k] = outer(d_vec[k], n_vec[k]) + outer(n_vec[k], d_vec[k])

# Pre-compute ray vectors for all stations: (N_sta, 3)
gamma, e_sv, e_sh  # from takeoff & azimuth (same formulas as polarity.py lines 69-87)

# Single einsum for all mechanisms × all stations:
P_all  = einsum('si, kij, sj -> ks', gamma, M_batch, gamma)   # (N_mech, N_sta)
SV_all = einsum('si, kij, sj -> ks', e_sv,  M_batch, gamma)
SH_all = einsum('si, kij, sj -> ks', e_sh,  M_batch, gamma)

log_SP_pred = log10(sqrt(SV^2 + SH^2) / (|P| + eps))
```

This replaces ~100K Python-level calls with 3 einsum operations. For memory management, chunk over mechanisms (default 10K per chunk) when N_stations is large.

### 7. `grid_search_event(takeoff, azimuth, polarity_obs, log_SP_obs, weights_pol, weights_sp, grid, ...)`

Single-event grid search core combining steps 3-6:
1. Compute ray geometry vectors from takeoff/azimuth arrays
2. Compute M_batch from grid (chunked if needed)
3. Vectorized forward model → P_all, SV_all, SH_all, log_SP_pred
4. polarity_misfit() → (N_mech,)
5. sp_ratio_misfit() → (N_mech,)
6. combined_misfit() → (N_mech,)
7. argmin → best mechanism

Returns best strike/dip/rake + misfit values + full misfit surface.

### 8. `estimate_uncertainty(misfit_surface, grid, confidence_level=0.1)`

HASH-style threshold method:
- All mechanisms with `misfit <= min_misfit + confidence_level` are "acceptable"
- Compute circular std of strike and rake (angles wrap), regular std of dip
- Assign quality grade: A (strike_unc < 25°, dip_unc < 15°), B (<35°, <25°), C (<45°, <35°), D (worse)

### 9. `prepare_observations(event_df, waveforms=None, ...)`

Extract observations for one event:
- Encode polarity: "U"→+1, "D"→-1, ""→0
- If waveforms provided: call `measure_sp_ratio()` for each station with both P and S picks
- Compute weights from SNR
- Use stored takeoff_angle/azimuth from dataset (fall back to recompute if missing)

### 10. `invert_focal_mechanism(event_df, waveforms=None, grid=None, alpha=0.5, ...)`

Main public API for one event:
1. `prepare_observations()` → observations dict
2. Check minimum data requirements (default: ≥8 polarities)
3. Build grid if not provided
4. `grid_search_event()` → best solution + misfit surface
5. `estimate_uncertainty()` → quality grade + uncertainties
6. Return result dict with: event_id, strike, dip, rake, misfit_total, misfit_pol, misfit_sp, n_pol, n_sp, strike_unc, dip_unc, rake_unc, quality

### 11. `invert_focal_mechanisms_batch(dataset_path, ...)`

Batch over all events in a dataset file:
- Load dataset via `utils.load_dataset()`
- Optionally load waveforms for S/P measurement
- Pre-build grid once, share across events
- Group by event_id, call `invert_focal_mechanism()` per event
- Collect results into DataFrame, optionally save as CSV

### 12. CLI (`__main__`)

```
python inversion.py --data_path ../NCEDC/dataset/2024/001.h5 \
    --alpha 0.5 --strike_step 5 --min_n_pol 8 --output results.csv
```

Follow same argparse pattern as `predict.py`.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Outlier robustness | Null-zone exclusion (HASH-style) | Stations near nodal planes are excluded rather than down-weighted — naturally tolerates ~10-20% false picks without iterative reweighting |
| S/P measurement | Vector norm (L2 across ENZ) | Component-orientation independent; captures full S energy |
| Misfit combination | Percentile-normalized weighted sum | Makes polarity (fraction) and S/P (log units) scale-comparable |
| S/P misfit norm | L1 (absolute deviation) | More robust to outlier S/P measurements than L2 |
| Vectorization | Batch einsum over full grid | 3 einsum calls replace ~100K Python loops; chunked for memory |
| Null zone threshold | 0.1 (configurable) | HASH default; excludes stations where |P_pred|/max < 10% |

## Verification

1. **Unit test**: Forward-inverse consistency — create synthetic event with known strike/dip/rake, generate synthetic polarities via `calc_radiation_pattern()`, run `invert_focal_mechanism()`, verify recovered mechanism matches (within grid spacing)
2. **Comparison with catalog**: Run on events that have catalog focal mechanisms (strike/dip/rake in HDF5 attrs), compare inverted vs catalog solutions
3. **Quick smoke test**: `python inversion.py --data_path ../NCEDC/dataset/2023/001.h5 --alpha 1.0` (polarity-only first, then add S/P)

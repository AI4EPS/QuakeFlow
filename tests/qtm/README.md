# QTM benchmark

A synthetic region with known ground truth, in the same format as `local/demo`, so
`cut_templates_cc.py`, `run_qtm.py` and `run_qtm_association.py` run on it unmodified.

Real events are cut out of the real Ridgecrest waveforms and spliced into a real noise
window at known times.  Every injection is a byte-for-byte copy of the waveform its
template was cut from, so a correct pipeline must recover **cc = 1 at the ground-truth
origin time** for the matching template.  Anything less is a bug, not a detection limit.

## Running it

These live in `tests/` because they verify the workflow in `scripts/`, not because they are
part of it.  They resolve `scripts/` and `scripts/local/` from their own location, so they
run from any working directory; only the workflow scripts themselves need `cd scripts`.

```bash
python tests/qtm/extract_noise.py   --src_region demo --region synthetic
python tests/qtm/extract_events.py  --src_region demo --region synthetic
python tests/qtm/make_benchmark.py  --region synthetic --test_ratio 3 1 0.3 0.1

(cd scripts && python cut_templates_cc.py    --region synthetic --mode TM)
(cd scripts && python run_qtm.py             local synthetic)
(cd scripts && python run_qtm_association.py --region synthetic)

python tests/qtm/evaluate_qtm.py    --region synthetic
python tests/qtm/plot_detections.py --region synthetic
```

One template set plus one test set per `--test_ratio`, every set a full pass over all 10
sources, so the injected-pick count is identical across sets (894 each) and recovery is
comparable set to set with no reweighting.  That is the detection-vs-SNR curve, from a
single pipeline run instead of one run per point.

### Where things live

Each CCTorch mode gets its own folder and its own template prefix, from `args.MODE_LAYOUT`,
so modes cannot overwrite each other's templates and adding one is a single entry there.
The repo convention is that a folder's `<stage>_*` files are that stage's **output**
(`adloc/adloc_events.csv`, `gamma/gamma_events.csv`), so inputs are `template_*`:

```
{region}/cctorch/     relocation  (the exception: legacy names the relocation chain reads)
    cctorch_{picks,events,stations}.csv   templates
    template.dat, traveltime*.dat, config.json, event_pairs.txt
    ccpairs/ -> dt.cc, dtcc.csv

{region}/qtm/         template matching
    template_{picks,events,stations}.csv  in  -- what you match WITH
    template.dat, traveltime*.dat, config.json, event_pairs.txt
    mseed_list.csv, pairs.txt, ccpairs/        (run_qtm.py's mseed x template pairs)
    qtm_{events,picks}.csv                out -- what you FOUND
```

`event_pairs.txt` (event pairs, from `cut_templates_cc.py`) and `pairs.txt` (mseed x
template, from `run_qtm.py`) are both "pairs" and both land in `qtm/`, hence the two names.

Detection defaults to **cc + MAD**: `--nmad` is 9 for `--mode TM`, 0 for `--mode CC`, and an
explicit `--nmad` always wins. Verified that none of this moves `dt.cc` -- it is byte-identical
to the pre-change baseline.

`evaluate_qtm.py` prints the numbers and writes `eval_summary.png` and `eval_by_set.csv`;
`plot_detections.py` writes the two figures to read the detections off:
`detect_01_timeline` (ground truth, what each candidate is made of, and a raster of every
correlation pick) and `detect_02_threshold` (where the threshold belongs).

Only `make_benchmark.py` needs re-running for a sweep; steps 1 and 2 are fixed.

Per-source and per-injection record sections are ~1 MB each and largely alike, so only a
couple are written: `cut_summary.png` and `record_gallery.png` carry the full set. Pass
`--max_event_figures -1` (`extract_events.py`) or `--max_sections -1` (`plot_events.py`) to
get one per event. Note that `record_inj_NN_{template,test}.png` embeds the block in the
filename, so stale figures survive a rebuild that reassigns blocks and then contradict the
new ones — clear the directory when the schedule changes.

## The regression test

```bash
python tests/qtm/test_identity.py --region synthetic                          # 1.0 and 0.1
python tests/qtm/test_identity.py --region synthetic --template_ratio 1 .3 .1 .03
python tests/qtm/test_identity.py --region synthetic --skip_pipeline          # check what is on disk
```

Rebuilds the benchmark at each amplitude, runs `cut_templates_cc.py` and `run_qtm.py`, and
asserts that **every injected template-block phase comes back at cc >= 0.999**.  Exit code
0 or 1, so it can be wired into CI.

The invariant is amplitude-independent by construction: a template is cut from the same
samples that were injected, so scaling the injection scales the template with it and the
normalised correlation is unchanged.  A drop at low amplitude therefore means the template
side and the continuous side are doing different things to the waveform — never a detection
limit.  Test-block recovery *is* SNR-dependent, so the test reports it but does not assert
it.

Current result:

```
 amplitude  result          returned     min cc      test-block recovery
         1    PASS           894/894   1.000000        2/894  (test SNR 0.01)
        10    PASS           894/894   1.000000        5/894  (test SNR 0.1)
       100    PASS           894/894   1.000000      728/894  (test SNR 1.0)
```

The identity check does not move across four orders of magnitude in amplitude, while
test-block recovery goes from nothing to almost everything between test SNR 0.1 and 1.0.
That contrast is the point: it separates "the correlation is wrong" from "the signal is too
weak", which was impossible to tell apart before.

The test deletes `qtm/ccpairs` before each run and insists the files come back:
`run_qtm.py` shells out to `torchrun` and returns 0 even when that dies, so without the
check a crashed run scores as a pass against the previous run's output.  That happened
during development and produced a "fix" that had never executed.

It leaves `{region}` built at the last amplitude in the list; re-run `make_benchmark.py`
to put it back where you want it.

## The knobs

| flag | default | what it controls |
|---|---|---|
| `--template_ratio` | 1.0 | SNR of the one template set — how clean the templates are |
| `--test_ratio` | 0.1 | SNR of each test set, **one value per set**. `--test_ratio 1 0.3 0.1 0.03` builds four test sets in one record |
| `--spacing_sec` | auto | seconds between injections; default spreads them evenly over the record |
| `--normalize` | `station` | divide the noise by its own band rms and each event by its own signal rms, both over all of a station's components together. Instrument gains here span three orders of magnitude, so without this one multiplier means a different SNR at every station. `network` keeps the falloff across the network; `none` injects raw counts. |

With `--normalize station` the ratios **are** the SNR, not a multiplier on counts.  Verified
from the written mseed rather than the bookkeeping: measured rms(signal window)/rms(pre-P
noise) is 1.363 against 1.414 predicted for SNR 1, and 0.984 against 1.00005 for SNR 0.01.

**SNR here means window rms, not peak.**  Signal is the rms of the 2–12 Hz cut from P to the
end of the window; noise is the rms of the whole noise channel.  That is the quantity a
matched filter integrates, so it is the right one for this benchmark — but the window is
`4*(ts-tp)` long (11 s at 20 km, 35 s at 60 km) and its tail is decayed coda, so a distant
station needs a larger *peak* to reach the same nominal SNR.  Note `--min_snr` in
`extract_events.py` is a different measure again (peak envelope over median background, used
only to screen candidates); the two numbers are not comparable.

**The measured decay** (2026-08-07, `--template_ratio 1 --test_ratio 1 0.3 0.1 0.03`, 894
injected picks per set):

| set | SNR | picks recovered | cc median | events detected |
|---|---|---|---|---|
| 0 template | 1 | 894/894 (100%) | 1.0000 | 10/10 |
| 1 test | 1 | 514/894 (57.5%) | 0.695 | 10/10 |
| 2 test | 0.3 | 91/894 (10.2%) | 0.545 | 10/10 |
| 3 test | 0.1 | 5/894 (0.6%) | 0.513 | 4/10 |
| 4 test | 0.03 | 3/894 (0.3%) | 0.525 | 4/10 |

**Pick-level true vs false positives** (same run, all 3984 raw correlation picks). A pick is
a true positive only if the template that fired is the source that was injected, at that
injection's origin time, on a station-phase that carried signal:

| label | picks | median cc | meaning |
|---|---|---|---|
| `true_positive` | 1507 (37.8%) | 1.000 | correct template, correct time |
| `wrong_template` | 136 (3.4%) | 0.526 | real injection, wrong source's template — a duplicate, not a noise trigger |
| `wrong_time` | 2338 (58.7%) | 0.521 | fired where nothing was injected |
| `duplicate` | 3 | 0.529 | extra peak on a station-phase already counted |

Overall precision 37.9%. **`min_cc_score = 0.5` ([cut_templates_cc.py:343](../cut_templates_cc.py#L343)) is
far too permissive** — the false positives sit in a narrow band just above it, and moving the
threshold is nearly free:

| min_cc | TP | FP | precision | recall |
|---|---|---|---|---|
| 0.50 | 1507 | 2474 | 37.9% | 33.7% |
| **0.60** | 1311 | 56 | **95.9%** | 29.3% |
| 0.65 | 1233 | 3 | 99.8% | 27.6% |
| 0.70 | 1145 | 0 | 100.0% | 25.6% |

0.5 → 0.6 removes 2418 of 2474 false positives and costs 196 of 1507 true positives.

**MAD threshold.** `--nmad N` (on `cut_templates_cc.py`, written into `cctorch/config.json`)
additionally requires `cc > N * median(|cc|)` of that template's own correlation trace --
the adaptive threshold of Shelly et al. (2007), who use 8-9x recomputed per template per
day. `--nmad 0` disables it. The absolute floor still applies, as it does in Shelly's work.
`compare_thresholds.py` draws them side by side in the `eval_recovery.png` layout:

| | absolute cc > 0.5 | 9 x MAD only | cc > 0.5 AND 9 x MAD |
|---|---|---|---|
| picks | 4943 | 3602 | 2996 |
| true positive | 2454 | **2468** | 2343 |
| wrong_template | 834 | 682 | **438** |
| wrong_time | 1652 | 452 | **215** |
| pick precision | 49.7% | 68.5% | **78.2%** |
| pick recall | 58.4% | **58.8%** | 55.8% |
| event recall | 44/50 | 44/50 | 41/50 |
| association false alarms | 82/126 | 14/58 | **6/47** |
| P recovered | 1053 | 970 | 960 |
| S recovered | 1401 | **1498** | 1383 |

**MAD alone dominates the absolute threshold on every measure**: more true positives, 19
points more precision, the same event recall, and a sixth of the association false alarms.
The reason is visible in the last two rows -- it does not simply tighten, it *redistributes*.
9 x MAD is ~0.535 for a short P template and ~0.42 for a long S one, so it raises the bar on
P (1053 -> 970, and the noisy single-component rows stop firing) and lowers it on S
(1401 -> 1498, recovering weak S picks the 0.5 floor had rejected). A single absolute number
cannot do both. The P/S false-positive ratio falls from 6.8x to 2.2x.

Adding the absolute floor on top buys another 10 points of precision and cuts association
false alarms to 6, at the cost of 3 events and ~125 true picks. Which to prefer depends on
whether the downstream step cares more about catalogue purity or completeness.

Caveat: Shelly thresholds a *network-stacked* correlation trace, ours is per station-phase,
so the constant 9 does not carry over from the literature -- it was picked to match his and
should be swept on this benchmark. At equal recall a plain absolute 0.6 is competitive
(86.1% precision at 53.5% recall); MAD's advantage is that it needs no per-phase tuning.
Per-set precision at the current 0.5 is 95.9% / 95.9% / 84.3% / 13.9% / 9.7% down the SNR
ladder, so nearly all of the damage is in the sets that are below the detection limit anyway.

`eval_recovery.png` puts every pick's fate on one page — event detection over the record on
top, then a station x injection raster for P and for S — and two things fall straight out of
it:

**Almost every false positive is on P.**

| phase | true_positive | wrong_template | wrong_time |
|---|---|---|---|
| P | 566 | 121 | **2289** |
| S | 941 | 15 | 49 |

97% of the false positives are P. The P window has to stop at the S arrival, so P templates
are much shorter than S templates and correlate with noise far more easily — the short
template is the cost of the window definition, not a bug in the correlation.

**Three single-component stations produce most of the noise.** `CI.WNM..EH`,
`CI.WRV2..EH` and `CI.WVP2..EH` record Z only, so `reduce_c` averages one channel instead of
three and there is no cross-component averaging to suppress noise:

| station | picks | TP | FP |
|---|---|---|---|
| CI.WNM..EH | 842 | 35 | **807** |
| CI.WRV2..EH | 445 | 41 | 404 |
| CI.WVP2..EH | 247 | 35 | 212 |

3 of 45 station-instrument entries, **38.5% of all picks and 57.4% of all false positives**,
for 111 of 1507 true positives. They read as solid orange rows near the top of the P raster.
A per-station cc threshold that scales with channel count, or simply excluding them, is the
other cheap lever.

Two things worth reading off the decay table. **Picks decay far faster than events**: at SNR 0.3 only a
tenth of the picks survive but every event is still found, because association stacks across
45 stations. The pick cliff is between SNR 1 and 0.3, the event cliff between 0.3 and 0.1.
And **sets 0 and 1 are at the same SNR** — 100% vs 57.5% — so that gap is purely the identical
noise realisation, which is what the identity check trades on and what nothing on real data
ever gets.

## How it is built

1. **Noise** — 12:00–13:00 on 2019-07-04, inside the 11:52–15:21 gap where the AdLoc
   catalogue lists no event at all and 5.5 h before the M6.4 foreshock.  All 129 channel
   files are trimmed, demeaned and rewritten with their original headers, so sampling
   rates (40 Hz BH, 100 Hz HH/HN/EH) and start times are the real ones.

2. **Events** — 10 events from the AdLoc catalogue, filtered for pick count, isolation in
   time and SNR against the local background, then chosen by farthest-point sampling so
   they are as far apart as the aftershock zone allows (~6 km minimum; the zone is only
   ~30 km across, so that is the ceiling for 10 events).  Windows are
   `[tp - win_lead_sec, tp + win_sp_factor*(ts - tp) + win_tail_sec]` (3 s, 4x, 0 s by default) from straight-ray `vp = 6`, `vs = vp/1.73`, tapered
   0.25 s at each end.  All 45 stations and all 129 channels are cut for every event.

   The hypocentre and origin time come straight from `adloc_events.csv` (`--hypocenter
   adloc`).  `--hypocenter fit` (refit each event from its own AdLoc picks under the same
   vp/vs) was the default while that catalogue was 21 km off; that is fixed (see below) and
   the refit now moves events only 0.4 km, so the catalogue value is used directly.  `fit`
   and `scedc` remain for cross-checks.

3. **Injection** — one template set plus one test set per `--test_ratio`, each set a full
   pass over all 10 sources, laid out one after another and uniformly spaced.  The template
   set is written to `adloc_events.csv` / `adloc_picks.csv` as the templates the pipeline
   gets; every injection, those included, is a detection it has to make.  Equal source
   counts per set are what make recovery comparable across SNR without reweighting.

   Set 0 is the template set, so it occupies the opening stretch of the record and each test
   set follows in turn.  "Template" therefore also means "early", and so does "high SNR"
   if the test ratios descend — which was measured (2026-08-06) rather than assumed:
   across 6 stations, in 20 s windows with the injections masked out, the network-median
   noise level differs between the two halves of the record by **0.22%**.  Spreading the
   templates over the record removes that confound but buys nothing on recall, and it makes
   the schedule harder to read, so it was tried and reverted.

   Individual stations are a different story — CI.WMF gets 51%/hr louder and CI.SLA 55%/hr
   quieter — so **per-station** test-block statistics do carry a first-half/second-half
   bias.  Keep that in mind before attributing a station's poor recovery to the station.

## Ground truth

- `{region}/benchmark/ground_truth_events.csv` — one row per injection: `inj_id`, `block`
  (`template` / `test`), `idx_event` (which source), `time`, location, `snr_z_median`,
  `set_id`, `amplitude` (the set's SNR), `template_event_index`.
- `{region}/benchmark/ground_truth_picks.csv` — one row per injection/station/phase:
  sample-exact `phase_time`, `tp_tt`, `ts_tt`, `win_pre`, `win_post`, `snr`, `injected`.
- `{region}/benchmark/eval_correlation.csv`, `eval_detections.csv` — written by
  `evaluate_qtm.py`.

Ground-truth P/S are *theoretical* straight-ray times, not picks.  At close stations the
real arrival can lead them by up to ~1 s; that is harmless because the same theoretical
time positions both the template and every injection, so they stay aligned.

## What the first run found (2026-07-28)

Self-match cc was 0.78 instead of 1.0.  Four differences between what the template side
does to a waveform and what the continuous side does, fixed one at a time and measured
here:

| fix | median cc, all picks | % picks >= 0.95 |
|---|---|---|
| baseline | 0.778 | 4.7% |
| + CCTorch bandpasses the continuous data to the template band | 0.879 | 33.7% |
| + the zero padding in truncated P templates is masked out | 0.984 | 60.1% |
| + accelerometer integration off (templates are not integrated) | 0.999 | 92.2% |
| + same resampling route on both sides (40 -> 100 Hz) | 1.000 | 92.5% |
| + channels and travel times read past empty components | **1.000** | **92.1%** |

**The identity check.** The template block is the injection each template was cut from, so
that correlation is a waveform against itself and must return exactly 1.  It now does, for
every phase that had signal injected:

    identity check: 894 injected template phases
      returned 894 (100.0%), cc min 1.000000, median 1.000000

Over all 20 injections, 1776 of the 1788 injected station-phases come back.  The 12 that do
not are all SNR <= 1.2 (median 0.22, against a median of 62 for the ones that do), nine of
them on CI.WBM..HN — the noisiest accelerometer in the network.  That is the benchmark
measuring the detection limit, not a defect: the test-block injections are the same
waveform in *different* noise, so cc falls off with SNR exactly as it should (median 0.71
below SNR 2, 1.0000 above SNR 100).

**One station had to be excluded, and finding out why is what the check is for.**
CI.MPM flat-lines for 5.3 minutes (constant value 790, 20:56:48–21:02:07) across the origin
of source event 4; 14% of that station's day is flat, because obspy fills gaps with the
last sample.  Demeaning a constant gives exactly zero, so nothing was injected there —
`extract_events.py` now reports flat cuts and `make_benchmark.py` marks them `injected =
False`, so the ground truth stops claiming 45 stations where only 42 carry signal.

The changes:

- `CCTorch/cctorch/data.py::read_mseed` — bandpass to `config.freqmin`/`freqmax`; the
  accelerometer integration is now behind `config.integrate_accelerometer` (default off);
  resampling uses decimate/resample instead of linear interpolation.  `template_mask` is
  loaded next to the templates and stacked into the info batch.
- `scripts/cut_templates_cc.py` — writes `freqmin`/`freqmax` into `cctorch/config.json`
  instead of hardcoding 2-12 Hz, writes `template_mask.dat` alongside `template.dat`, and
  takes `--mode {CC,TM}`.
- `CCTorch/cctorch/model.py` — the mask is used for the template's own normalization, as
  the kernel of the same grouped `conv1d` for the running mean/std of the continuous data,
  and to count how many channels `reduce_c` should average over.
- `CCTorch/cctorch/transforms.py::DetectPeaksTM` — each channel keeps its own travel time
  and broadcasts against its own correlation series, instead of channel 0's value standing
  in for all three.  `traveltime_mask` says which components were actually written, and the
  times are averaged over those only when `reduce_c` has already collapsed the channels
  into one series.  Component traces can start at slightly different times, so nothing
  assumes they agree.

**Single-component stations were silently dropped.**  Three of the 45 (CI.WNM..EH,
CI.WRV2..EH, CI.WVP2..EH) record Z only.  `reduce_c` averaged over three channels
regardless, capping them at cc = 1/3, below `min_cc = 0.5`; and `DetectPeaksTM` read the
travel time from channel 0, which for those stations is the 0.0 left in an unwritten slot,
putting the origin time a whole travel time late.  Both now consult the masks.

**Why a mask and not a longer P window.**  The P window has to stop at the S arrival, so P
and S templates are genuinely different lengths while sharing one `nt = max(P, S)` array.
The old code correlated the leftover zeros as if they were data: a P slot filled 250 of 400
samples correlated with a copy of itself at 0.63.  Lengthening P would put the S arrival
inside the P template; the fix is to leave the windows alone and exclude the padding from
the normalization.  `template_mask.dat` is written by the same loop that writes
`template.dat`, so the two cannot disagree, and it mirrors the existing
`traveltime_mask.dat`.  `cctorch_picks.csv` is unchanged.

**Still open: the association step.**  It returns 162 entries for 20 injections, and it got
*worse* at every correlation fix above — 25, 38, 79, 205 — because as cc rose more picks
cleared `min_cc` and more wrong-template clusters survived.  `plot_detections.py` shows the two
populations are not close to each other:

| | real | false alarm |
|---|---|---|
| count | 20 | 142 |
| distinct stations | 40-45 | 1-15 |
| median cc | ~1.0 | ~0.53 |
| origin-time error | < 0.001 s | 3-26 s, at offsets that repeat across events |

Requiring 16 distinct stations gives recall 20/20 with zero false alarms; `min_stations`
against recall and precision is tabulated in `benchmark/detection_threshold.csv`.  The
extras are real injections matched by the *wrong* template (off-diagonal cc 0.5-0.67
between sources ~6 km apart), so they are duplicates rather than noise triggers.
`run_qtm_association.py` filters on neither measure: `MIN_STATION = 3` counts picks in a
2 s bin, not distinct stations, and no threshold is applied to the output events.

## The AdLoc locations in local/demo (fixed 2026-08-05, except depth)

`adloc_events.csv` used to put the whole sequence ~21 km ESE and ~6 km too deep.  The
horizontal error is gone; the depth bias is not.  Time-matched against `obspy/catalog.csv`:

| | before (2026-07-28) | now (2026-08-05 rerun) |
|---|---|---|
| matched to SCEDC | 582 | 622 / 1232 |
| horizontal offset | +0.198° lon, −0.097° lat (~21 km) | +0.0038° lon, +0.0030° lat (**0.5 km**) |
| depth offset | +5.98 km | **+5.89 km** |
| refit moves the event | 20.6 km | **0.4 km** |
| pick residual vs `vp = 6` | −1.95 ± 1.47 s | −0.03 ± 0.28 s (P), +0.04 ± 0.51 s (S) |

The depth bias is a velocity-model specification problem, not a depth/origin-time
trade-off: `init_eikonal2d` runs `np.interp` over the layer nodes, so the layered model in
`run_adloc_v2.py` (`zz=[0,5.5,16,32]`, `vp=[5.5,5.5,6.7,7.8]`) becomes a linear gradient
that is only 5.79/6.01/6.24 km/s at 8/10/12 km.  Fixing the epicentre at SCEDC and scanning
depth, that model's misfit bottoms out at 12.0 km against SCEDC's 4.9 km — so ADLoc is
finding the right answer for the model it is handed.  Until a proper 1D model is chosen,
benchmark **depths run ~6 km deep**; epicentres and origin times do not.

Station coordinates in `adloc_stations.csv` were always correct.

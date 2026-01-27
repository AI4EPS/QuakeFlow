# Systematic Testing of NCEDC and SCEDC Dataset Building

Test the complete data processing pipeline for both NCEDC and SCEDC regions.

**Test Data:** 2024/001 (already processed via `parse_event.py`, `parse_phase.py`, `parse_fm.py`, `cut_event.py`, `cut_event_parquet.py`)

---

## 1. Validate CSV Catalog Files

### Prerequisites
Verify processed files exist on GCS:
- `events.csv` — event metadata
- `phases.csv` — phase picks 
- `focal_mechanisms.csv` — fault plane solutions

If files are missing, warn and provide the command to regenerate.

### Validation Steps

1. **Understand raw data formats** by reading source parsing scripts of events (parse_event.py), phases (parse_phase.py), and focal mechanisms (parse_fm.py).

2. **Download raw catalog data** for the selected date. Be careful about the different sources for event, phase, focal mechinism and the differences between NCEDC and SCEDC. 
!!!Don't forget to download raw phase and focal_mechanism files.


3. **Compare raw vs processed data**:
   - Verify no events, phases, or focal mechanisms are missing (count match)
   - If missing, analyze how the data is filtered out.
   - Verify all extracted fields are correctly parsed
   !!!Don't forget to check all extracted attributes.
   !!!Don't forget to check the extracted values all match between raw sources and generated csvs.

---

## 2. Validate HDF5 and Parquet Waveform Files

### Prerequisites
Verify processed files exist on GCS:
- `001/waveform.h5` — HDF5 hierarchical format
- `001.parquet` — columnar format for ML pipelines

If files are missing, warn and provide the command to regenerate.

### Visualization Tests (20 events)

1. **Phase moveout plots** — all stations for a single event:
   - Plot Z-component waveforms sorted by distance
   - Overlay P and S arrival markers
   - Verify moveout pattern is physically consistent (arrivals later at farther stations)

2. **Single-station waveform plots** — zoom in on each phase:
   - Plot P-wave window (e.g., ±2 seconds around pick)
   - Plot S-wave window (e.g., ±2 seconds around pick)
   - Verify phase alignment and polarity labels match waveform first motion
   - Display all metadata attributes (SNR, distance, azimuth, weight) on the side.

### Attribute Validation

Cross-check waveform file attributes against source CSV files:
- Event metadata
- Phase picks
- Station info
- Focal mechanism
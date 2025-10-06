# Data Directory Guide

> This repo **does not** track large/raw datasets in Git.  
> This document explains the local layout, what each folder contains, where the data comes from,  
> and how to (re)create the derived artifacts.

## 📁 Layout (local, not pushed)

```
data/
├─ raw/                      # IGNORE in git (all large/original datasets live here)
│  ├─ NASABatteryAging/      # NASA Li-ion aging datasets (.mat + README.txt)
│  │  ├─ 1. BatteryAgingARC-FY08Q4/
│  │  ├─ 2. BatteryAgingARC_25_26_27_28_P1/
│  │  ├─ 3. BatteryAgingARC_25-44/
│  │  ├─ 4. BatteryAgingARC_45_46_47_48/
│  │  ├─ 5. BatteryAgingARC_49_50_51_52/
│  │  └─ 6. BatteryAgingARC_53_54_55_56/
│  ├─ CALCEBatteryDatasets/  # (optional) CALCE packs/cycles, zips/CSVs, etc.
│  └─ EVIoT-PredictiveMaint/ # (optional) IoT maintenance dataset, CSVs/zips
│
├─ processed/                # Derived, machine-readable artifacts
│  ├─ metadata/              # Small provenance/config files (tracked in git)
│  │  └─ nasa_training_context.json
│  ├─ clean_nasa_charge_*.csv        # (ignored) resampled/feature CSVs
│  └─ nasa_manifest.csv              # (ignored) inventory/summary of .mat files
│
└─ README (this file)
```

### Why raw data isn’t in Git
- Large binaries bloat history and slow down cloning.
- Different users may keep different slices locally.
- Reproducibility comes from code + lightweight **metadata sidecars**, not storing the raw bytes.

---

## 📦 Datasets (sources & contents)

### 1) NASA Lithium-Ion Battery Aging (ARC) – **primary for this project**
- **Format:** MATLAB `.mat` files + `README.txt` per subset.
- **Structure:** Six subset folders (FY08Q4; 25_26_27_28_P1; 25–44; 45–48; 49–52; 53–56).
- **Signals (typical):** `Time`, `Voltage_measured`, `Current_measured`, `Temperature_measured`, `Capacity` (usually during discharge), plus impedance records in some cycles.
- **Charge protocol (common):** CC 1.5 A → CV 4.2 V, terminate ≈ 20 mA.
- **Why used:** Well-structured cycle data suitable for fast-charge policy exploration and benchmarking.

> Put the unpacked NASA subset folders under:  
> `data/raw/NASABatteryAging/…`

### 2) CALCE Battery Datasets (optional, exploratory)
- **Format:** mix of CSV/ZIP.
- **Status:** Present for potential cross-checks; not part of the default pipeline.
- **Location:** `data/raw/CALCEBatteryDatasets/…`

### 3) EV IoT Predictive Maintenance (optional, exploratory)
- **Format:** CSVs, archives.
- **Status:** Not part of the default pipeline.
- **Location:** `data/raw/EVIoT-PredictiveMaint/…`

---

## 🔁 How to prepare data locally (after cloning)

1. **Create the local folders** (if they don’t exist):
   - Windows (PowerShell):  
     ```powershell
     mkdir data\raw, data\processed\metadata -ErrorAction SilentlyContinue
     ```
   - macOS/Linux:  
     ```bash
     mkdir -p data/raw data/processed/metadata
     ```

2. **Place datasets**:
   - Copy/unzip the NASA folders into `data/raw/NASABatteryAging/` exactly as shown above.
   - (Optional) Add CALCE / EVIoT under their respective folders if you plan to use them.

3. **Verify the raw inventory** (lightweight check, no heavy loads):
   ```bash
   # From repo root, in the project's conda env:
   python -m src.nasa_data_extract.build_manifest
   ```
   This writes `data/processed/nasa_manifest.csv` (ignored by Git)  
   with cycle counts and quick size proxies.

---

## 🧪 Recreating derived artifacts

The codebase ships with small, reproducible runners:

- **Small default slice (room-temp B0005 only):**
  ```bash
  python -m src.nasa_data_extract.run_nasa_pipeline
  ```
  This writes a resampled/feature CSV like:  
  `data/processed/clean_nasa_charge_SINGLEFILE.csv` (ignored by Git)  
  and a sidecar provenance file:  
  `data/processed/clean_nasa_charge_SINGLEFILE.meta.json` (tracked if you move/copy into `processed/metadata`).

- **Specific file or subset (scale intentionally):**
  ```bash
  # Single file
  python -m src.nasa_data_extract.run_nasa_pipeline --file "1. BatteryAgingARC-FY08Q4/B0005.mat"

  # One subset (limit first N files)
  python -m src.nasa_data_extract.run_nasa_pipeline --subset "1. BatteryAgingARC-FY08Q4" --limit 2
  ```

---

## 🧾 Provenance & registry

- Each pipeline run can emit a **sidecar JSON** (provenance) next to the CSV with:
  - dataset family & subsets,
  - input file list + hashes (if enabled),
  - processing config (e.g., resample rate, capacity assumptions),
  - environment versions.
- A rolling **registry CSV** can be maintained at `data/processed/datasets_registry.csv` to track all outputs produced locally.

> See: `src/nasa_data_extract/metadata.py` and `run_nasa_pipeline.py` for implementation details.

---

## 🔐 Licensing & usage

- Check the original dataset licenses/terms before redistribution.
- This repo **does not** redistribute raw data; users fetch datasets themselves and place them under `data/raw/`.
- Derived CSVs are ignored by default; share them only if your data license permits.

---

## 🧰 Troubleshooting

- **Raw data shows up in Git status?**  
  Ensure `.gitignore` exists at repo root with:
  ```
  /data/raw/**
  /data/processed/**
  !/data/processed/metadata/**
  ```
  If files were accidentally committed earlier, run `git rm -r --cached data/raw`.

- **Memory pressure during processing?**  
  Use `--file` or `--subset --limit` to keep slices small,  
  or switch to an append/chunked mode (ask us to enable).

---

## ❓FAQ

- **Can I put sample data in the repo?**  
  Yes — place tiny examples under `data/samples/` (tracked), not under `data/raw/`.

- **Where do I change nominal capacity or resample rate?**  
  See `engineer_features.py` and the metadata config under `data/processed/metadata/`.

---

*Last updated automatically when committed.*

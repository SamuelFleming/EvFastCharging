# src/nasa_data_extract/run_nasa_pipeline.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from read import get_root
from extract_charge_cycles import build_charge_dataframe
from engineer_features import resample_and_engineer

DEFAULT_SUBSET = "1. BatteryAgingARC-FY08Q4"
DEFAULT_FILE   = "1. BatteryAgingARC-FY08Q4/B0005.mat"

def main():
    ap = argparse.ArgumentParser(description="NASA charge-cycle pipeline (file/subset/whole)")
    ap.add_argument("--subset", nargs="+", help="Subset folder name(s) exactly as seen under data/raw/NASABatteryAging")
    ap.add_argument("--file", nargs="+", help="Specific .mat file(s), relative to RAW root (or absolute paths)")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N files after filtering")
    ap.add_argument("--out", default=None, help="Output CSV path")
    args = ap.parse_args()

    raw_root = get_root()
    data_dir = raw_root.parents[2]   # .../data
    out_dir = data_dir / "processed" # -> .../data/processed
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default behavior: smallest, simplest slice
    subset = args.subset
    files = args.file
    if subset is None and files is None:
        print(f"No args provided → defaulting to 1 file: {DEFAULT_FILE}")
        files = [DEFAULT_FILE]

    # Determine output
    if args.out:
        out_csv = Path(args.out)
    elif files:
        out_csv = out_dir / "clean_nasa_charge_SINGLEFILE.csv"
    elif subset:
        suffix = "_".join(s.replace(" ", "").replace(".", "") for s in subset)
        out_csv = out_dir / f"clean_nasa_charge_{suffix}.csv"
    else:
        out_csv = out_dir / "clean_nasa_charge.csv"

    # Resolve file list (if --file provided)
    mat_paths = None
    if files:
        paths = []
        for f in files:
            p = Path(f)
            if not p.is_absolute():
                p = raw_root / p
            paths.append(p)
        mat_paths = paths[: args.limit] if args.limit else paths

    # Extract
    print("[1/3] Extracting charge cycles…")
    charge_df = build_charge_dataframe(mat_paths=mat_paths, subset=subset, limit=args.limit)
    if charge_df.empty:
        print("No data matched your filters. Check names/paths.")
        return
    print(f"   rows={len(charge_df):,}, batteries={charge_df['battery_id'].nunique()}, subsets={charge_df['subset'].nunique()}")

    # Resample + features
    print("[2/3] Resampling to 1 Hz and engineering features…")
    feat_df = resample_and_engineer(charge_df)
    print(f"   rows={len(feat_df):,} after resampling")

    # Write
    print("[3/3] Writing CSV…")
    feat_df.to_csv(out_csv, index=False)
    print("   wrote:", out_csv)

    with pd.option_context("display.max_columns", 12, "display.width", 140):
        print("\nPreview:\n", feat_df.head())

if __name__ == "__main__":
    main()
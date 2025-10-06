# src/nasa_data_extract/test_nasa_pipeline.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from read import get_root
from extract_charge_cycles import build_charge_dataframe
from engineer_features import resample_and_engineer

def main():
    raw_root = get_root()
    out_dir = raw_root.parent / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "clean_nasa_charge.csv"

    print("[1/3] Extracting charge cycles…")
    charge_df = build_charge_dataframe()
    if charge_df.empty:
        print("No .mat data found under:", raw_root)
        return

    print(f"   rows={len(charge_df):,}, batteries={charge_df['battery_id'].nunique()}, "
          f"subsets={charge_df['subset'].nunique()}")

    print("[2/3] Resampling to 1 Hz and engineering features…")
    feat_df = resample_and_engineer(charge_df)
    print(f"   rows={len(feat_df):,} after resampling")

    print("[3/3] Writing CSV…")
    feat_df.to_csv(out_csv, index=False)
    print("   wrote:", out_csv)

    # Preview
    with pd.option_context("display.max_columns", 12, "display.width", 120):
        print("\nPreview:\n", feat_df.head())

if __name__ == "__main__":
    main()

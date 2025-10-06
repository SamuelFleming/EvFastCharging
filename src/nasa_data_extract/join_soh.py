# src/nasa_data_extract/join_soh.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from .extract_discharge_cycles import build_discharge_capacity_table
from .read import get_root

def main():
    print("[join_soh] Starting SoH construction pipeline")
    root = get_root()
    print(f"[join_soh] Using RAW ROOT: {root}")

    # --- Step 1: build the discharge capacity table ---
    df_cap = build_discharge_capacity_table()
    if df_cap.empty:
        print("[join_soh] No discharge capacity data extracted — exiting early.")
        return

    print(f"[join_soh] Discharge capacity table → {len(df_cap):,} rows")
    print(df_cap.head())

    # --- Step 2: compute SoH relative to the first cycle per battery ---
    df_cap = (
        df_cap.sort_values(["battery_id", "cycle_id"])
              .groupby("battery_id")
              .apply(lambda g: g.assign(
                  capacity_ref_Ah=g["capacity_discharge_Ah"].iloc[0],
                  SoH=g["capacity_discharge_Ah"] / g["capacity_discharge_Ah"].iloc[0]
              ))
              .reset_index(drop=True)
    )
    print("[join_soh] SoH computed successfully.")
    print(df_cap.head())

    # --- Step 3: write output ---
    out_path = Path("data/processed/TblSoH_discharge_capacity.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_cap.to_csv(out_path, index=False)

    print(f"[join_soh] Wrote SoH table to: {out_path.resolve()}")
    print(f"[join_soh] Batteries processed: {df_cap['battery_id'].nunique()}")
    print(f"[join_soh] Cycle range: {df_cap['cycle_id'].min()}–{df_cap['cycle_id'].max()}")
    print("[join_soh] Done ✅")

if __name__ == "__main__":
    main()


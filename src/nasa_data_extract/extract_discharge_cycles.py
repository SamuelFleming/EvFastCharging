# src/nasa_data_extract/extract_discharge_cycles.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from .read import get_root, load_mat_safely, iter_cycles, get_field, list_mat_files

def extract_discharge_capacity_from_file(path: Path) -> pd.DataFrame:
    print(f"[extract_discharge_capacity_from_file] Loading: {path.name}")
    d = load_mat_safely(path)
    records = []

    # Look for the top-level struct (e.g., 'B0005')
    top_keys = [k for k in d.keys() if not k.startswith("__")]
    for k in top_keys:
        battery = d[k]
        cycles = iter_cycles(battery)
        print(f"  Found {len(cycles)} cycles in {k}")
        for i, cyc in enumerate(cycles, 1):
            cyc_type = getattr(cyc, "type", "").lower()
            if cyc_type != "discharge":
                continue  # only process discharge cycles

            data = getattr(cyc, "data", None)
            if data is None:
                continue

            # Extract discharge data
            V = get_field(data, "Voltage_measured", "voltage_measured", "V")
            I = get_field(data, "Current_measured", "current_measured", "I")
            T = get_field(data, "Temperature_measured", "temperature_measured", "T")
            t = get_field(data, "Time", "time")
            C = get_field(data, "Capacity", "capacity")

            # Use provided capacity (in Ah) if available
            if C is not None and np.isscalar(C):
                cap_Ah = float(C)
            elif C is not None and isinstance(C, (np.ndarray, list)):
                cap_Ah = float(np.max(C))
            else:
                # Compute from current integration
                cap_Ah = np.trapz(np.abs(I), t) / 3600.0

            records.append(dict(
                battery_id=path.stem,
                cycle_id=i,
                capacity_discharge_Ah=cap_Ah,
                ambient_C=get_field(cyc, "ambient_temperature", "ambient_T", "ambient"),
                type=cyc_type,
            ))

        print(f"  → Extracted {len(records)} discharge cycles from {path.name}")

    return pd.DataFrame(records)

def build_discharge_capacity_table(files=None) -> pd.DataFrame:
    root = get_root()
    print(f"[build_discharge_capacity_table] Root = {root}")
    mats = files or list_mat_files(limit=5)
    print(f"Found {len(mats)} candidate .mat files:")
    for p in mats:
        print(" •", p)

    dfs = []
    for p in mats:
        try:
            df = extract_discharge_capacity_from_file(p)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"!! Error on {p.name}: {e}")

    if not dfs:
        print("No discharge cycles extracted.")
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    print(f"[build_discharge_capacity_table] Combined → {len(out)} rows total")
    return out

if __name__ == "__main__":
    df = build_discharge_capacity_table()
    out = Path("data/processed/TblSoH_discharge_capacity_DEBUG.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote debug discharge table: {out}")



# src/nasa_data_extract/build_manifest.py

"""
Reminder: subset specs (so you can choose wisely)
    1. FY08Q4 → Room temp; CC-CV charge (1.5 A → 4.2 V, ~20 mA cutoff), mostly CC 2 A discharges, clean runs.
Best starting point (e.g., B0005).
    2. 25_26_27_28_P1 → ~24 °C; pulsed discharge (0.05 Hz, 4 A, 50% duty); varying discharge cut-offs (2.0/2.2/2.5/2.7 V).
    3. 25–44 → Hot (~43 °C); CC 4 A discharge; same cut-off variants (cells 29–32).
    4. 45–48 → Cold (~4 °C); mixed 4 A / 1 A; some very low-capacity outliers.
    5. 49–52 → Cold (~4 °C); CC 2 A; experiment crash → incomplete/odd runs; apply QA if using.
    6. 53–56 → Cold (~4 °C); CC 2 A; generally clean completion.

All share the same charge recipe (CC 1.5 A, CV 4.2 V, cutoff ~20 mA).
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any

from read import get_root, list_mat_files, load_mat_safely, iter_cycles, get_field

def summarize_mat(path: Path) -> Dict[str, Any]:
    d = load_mat_safely(path)
    keys = [k for k in d.keys() if not str(k).startswith("__")]

    counts = {"charge":0, "discharge":0, "impedance":0, "other":0}
    total_cycles = 0
    approx_points_charge = 0  # rough proxy from the first charge cycle's time length

    def handle_struct(s):
        nonlocal counts, total_cycles, approx_points_charge
        cycles = iter_cycles(s)
        for c in cycles:
            ctype = get_field(c, "type", "Type", "operation", "name")
            ctype = (str(ctype).lower() if ctype is not None else "")
            if "charge" in ctype:
                counts["charge"] += 1
                # try to estimate number of samples
                data = get_field(c, "data", "Data", "measurements")
                if data is not None and approx_points_charge == 0:
                    t = get_field(data, "Time", "time", "t")
                    if t is not None:
                        arr = np.squeeze(np.asarray(t))
                        if arr.ndim == 0:
                            approx_points_charge = 1
                        else:
                            approx_points_charge = int(arr.shape[0])
            elif "discharge" in ctype:
                counts["discharge"] += 1
            elif "imped" in ctype:
                counts["impedance"] += 1
            else:
                counts["other"] += 1
            total_cycles += 1

    if keys:
        for k in keys:
            handle_struct(d[k])
    else:
        handle_struct(d)

    subset = path.relative_to(get_root()).parts[0]
    return {
        "subset": subset,
        "file": str(path.relative_to(get_root())),
        "battery_id": path.stem,
        "cycles_total": total_cycles,
        "cycles_charge": counts["charge"],
        "cycles_discharge": counts["discharge"],
        "cycles_impedance": counts["impedance"],
        "cycles_other": counts["other"],
        "approx_points_first_charge": approx_points_charge
    }

def main():
    raw_root = get_root()
    out_csv = raw_root.parent / "processed" / "nasa_manifest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in list_mat_files():
        try:
            rows.append(summarize_mat(p))
        except Exception as e:
            rows.append({
                "subset": p.relative_to(raw_root).parts[0],
                "file": str(p.relative_to(raw_root)),
                "battery_id": p.stem,
                "cycles_total": np.nan,
                "cycles_charge": np.nan,
                "cycles_discharge": np.nan,
                "cycles_impedance": np.nan,
                "cycles_other": np.nan,
                "approx_points_first_charge": np.nan,
                "error": str(e)
            })

    df = pd.DataFrame(rows).sort_values(["subset", "file"])
    df.to_csv(out_csv, index=False)
    with pd.option_context("display.max_rows", 50, "display.width", 160):
        print(df.head(20))
    print("\nWrote manifest:", out_csv)

if __name__ == "__main__":
    main()

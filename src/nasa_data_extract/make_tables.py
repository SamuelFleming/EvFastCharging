# src/nasa_data_extract/make_tables.py

'''
Usage

# Default (uses data/processed/clean_nasa_charge_SINGLEFILE.csv)
python -m src.nasa_data_extract.make_tables

# Or specify a CSV and output dir
python -m src.nasa_data_extract.make_tables --csv "data/processed/clean_nasa_charge_FY08Q4.csv" --outdir "data/processed/fy08q4_tables"
'''

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

DEFAULT_CSV = Path("data/processed/clean_nasa_charge_SINGLEFILE.csv")

def make_tbl1(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["cell_id"] = df["battery_id"]
    df["phase"] = "charge"
    df["overV"] = (df["V"] > 4.2).astype(int)
    T_MAX = 45.0
    df["overT"] = (df["T_cell"] > T_MAX).astype(int)
    cols = [
        "cell_id","battery_id","subset","cycle_id",
        "t","V","I","T_cell","SoC","dSoC","C_rate","capacity",
        "overV","overT","phase"
    ]
    return df[cols]

def make_tbl2(tbl1: pd.DataFrame, soc_target: float = 0.8) -> pd.DataFrame:
    grp = tbl1.groupby(["cell_id","cycle_id"], sort=False)
    rows = []
    eid = 0
    for (cell, cyc), g in grp:
        g = g.sort_values("t")
        soc_init = float(g["SoC"].iloc[0]) if not g.empty else np.nan
        T_init = float(g["T_cell"].iloc[0]) if not g.empty else np.nan
        rows.append({
            "episode_id": eid,
            "cell_id": cell,
            "cycle_id": cyc,
            "subset": g["subset"].iloc[0] if "subset" in g.columns else "",
            "SoC_init": soc_init,
            "T_init": T_init,
            "SoC_target": soc_target,
            "len_sec": float(g["t"].iloc[-1]) if not g.empty else np.nan
        })
        eid += 1
    return pd.DataFrame(rows)

def make_tbl3_from_defaults(tbl1: pd.DataFrame) -> pd.DataFrame:
    uniq = tbl1[["cell_id","subset"]].drop_duplicates()
    V_MAX = 4.2
    T_MAX = 45.0
    C_MAX_CRATE = 4.0
    uniq = uniq.copy()
    uniq["protocol_id"] = uniq["subset"].astype("category").cat.codes
    uniq["V_max"] = V_MAX
    uniq["T_max"] = T_MAX
    uniq["C_max_Crate"] = C_MAX_CRATE
    uniq["chemistry"] = pd.NA
    return uniq[["cell_id","subset","protocol_id","chemistry","V_max","T_max","C_max_Crate"]]

def main():
    ap = argparse.ArgumentParser(description="Materialize RL-ready tables from a processed NASA CSV.")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to processed CSV (default: %(default)s)")
    ap.add_argument("--outdir", default=None, help="Output directory (default: same folder as CSV)")
    ap.add_argument("--soc_target", type=float, default=0.8, help="Target SoC for episodes (default: %(default)s)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    tbl1 = make_tbl1(csv_path)
    tbl1.to_csv(outdir / "Tbl1_signals.csv", index=False)

    tbl2 = make_tbl2(tbl1, soc_target=args.soc_target)
    tbl2.to_csv(outdir / "Tbl2_episodes.csv", index=False)

    tbl3 = make_tbl3_from_defaults(tbl1)
    tbl3.to_csv(outdir / "Tbl3_metadata.csv", index=False)

    print("Wrote:", outdir / "Tbl1_signals.csv")
    print("Wrote:", outdir / "Tbl2_episodes.csv")
    print("Wrote:", outdir / "Tbl3_metadata.csv")

if __name__ == "__main__":
    main()


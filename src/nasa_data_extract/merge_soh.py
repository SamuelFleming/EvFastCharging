# src/nasa_data_extract/merge_soh.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _read_csv(p: Path) -> pd.DataFrame:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def _ensure_battery_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Ensure a DataFrame has 'battery_id'; if missing but 'cell_id' exists, alias it."""
    if "battery_id" not in df.columns:
        if "cell_id" in df.columns:
            df = df.copy()
            df["battery_id"] = df["cell_id"]
            print(f"[merge_soh] ({name}) Added 'battery_id' from 'cell_id'")
        else:
            raise ValueError(f"({name}) requires 'battery_id' or 'cell_id'")
    return df

def merge_soh_into_tbls(
    soh_csv: Path,
    tbl1_csv: Path,
    tbl2_csv: Path,
    out_tbl1_csv: Path | None = None,
    out_tbl2_csv: Path | None = None,
) -> tuple[Path, Path]:
    """
    Left-join SoH (battery_id, cycle_id → SoH) into Tbl2 (episodes) and Tbl1 (signals).
    Returns paths to written files.
    """
    soh_csv = Path(soh_csv)
    tbl1_csv = Path(tbl1_csv)
    tbl2_csv = Path(tbl2_csv)
    out_tbl1_csv = Path(out_tbl1_csv or tbl1_csv.with_name("Tbl1_signals_with_SoH.csv"))
    out_tbl2_csv = Path(out_tbl2_csv or tbl2_csv.with_name("Tbl2_episodes_with_SoH.csv"))

    df_soh = _read_csv(soh_csv)
    df1 = _ensure_battery_id(_read_csv(tbl1_csv), "Tbl1")
    df2 = _ensure_battery_id(_read_csv(tbl2_csv), "Tbl2")

    # Validate keys
    if "cycle_id" not in df1.columns:
        raise ValueError("Tbl1 must contain cycle_id")
    if "cycle_id" not in df2.columns:
        raise ValueError("Tbl2 must contain cycle_id")
    if not {"battery_id","cycle_id","SoH"}.issubset(df_soh.columns):
        raise ValueError("SoH csv must contain battery_id, cycle_id, SoH")

    # Merge
    keys = ["battery_id","cycle_id"]
    df2m = df2.merge(df_soh[keys + ["SoH"]], on=keys, how="left")
    df1m = df1.merge(df_soh[keys + ["SoH"]], on=keys, how="left")

    out_tbl2_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tbl1_csv.parent.mkdir(parents=True, exist_ok=True)
    df2m.to_csv(out_tbl2_csv, index=False)
    df1m.to_csv(out_tbl1_csv, index=False)

    print(f"[merge_soh] wrote: {out_tbl2_csv}  (rows={len(df2m):,})  SoH_nulls={df2m['SoH'].isna().sum():,}")
    print(f"[merge_soh] wrote: {out_tbl1_csv}  (rows={len(df1m):,})  SoH_nulls={df1m['SoH'].isna().sum():,}")
    return out_tbl1_csv, out_tbl2_csv


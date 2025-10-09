# src/utils/data_interface.py
from pathlib import Path
from typing import Tuple, Dict, Any, Literal
import pandas as pd
from .schema import REQUIRED_TBL1, REQUIRED_TBL2, REQUIRED_TBL3, ensure_columns

class DataContractError(Exception):
    pass

def load_mvp_tables(processed_dir: Path | str, with_soh: Literal["none","tbl1","tbl2","both"]="none"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    processed_dir = Path(processed_dir)

    tbl1_name = "Tbl1_signals_with_SoH.csv" if with_soh in ("tbl1","both") else "Tbl1_signals.csv"
    tbl2_name = "Tbl2_episodes_with_SoH.csv" if with_soh in ("tbl2","both") else "Tbl2_episodes.csv"

    tbl1 = pd.read_csv(processed_dir / tbl1_name)
    tbl2 = pd.read_csv(processed_dir / tbl2_name)

    # Tbl3 is optional for now — if you don’t have it yet, you can stub meta.
    tbl3_path = processed_dir / "Tbl3_metadata.csv"
    tbl3 = pd.read_csv(tbl3_path) if tbl3_path.exists() else pd.DataFrame()

    ensure_columns(tbl1, REQUIRED_TBL1, "Tbl1")
    ensure_columns(tbl2, REQUIRED_TBL2, "Tbl2")
    if not tbl3.empty:
        ensure_columns(tbl3, REQUIRED_TBL3, "Tbl3")

    # Build meta (fallbacks if Tbl3 not present)
    if tbl3.empty:
        meta = {
            "cell_id": tbl1["cell_id"].iloc[0],
            "subset": tbl1["subset"].iloc[0],
            # MVP defaults aligned with your Phase-4 plan:
            "V_max": 4.2,
            "T_max": 55.0,      # use your chosen ceiling; plan mentions 55 °C for metrics
            "C_max": 4.0,
            "protocol_id": None,
            "chemistry": None,
        }
    else:
        row = tbl3.iloc[0]
        meta = {
            "cell_id": row["cell_id"],
            "subset": row["subset"],
            "V_max": float(row["V_max"]),
            "T_max": float(row["T_max"]),
            "C_max": float(row["C_max_Crate"]),
            "protocol_id": int(row["protocol_id"]) if pd.notna(row["protocol_id"]) else None,
            "chemistry": str(row["chemistry"]) if pd.notna(row["chemistry"]) else None,
        }

    return tbl1, tbl2, meta

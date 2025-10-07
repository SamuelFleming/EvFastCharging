# src/utils/schema.py
from typing import Iterable
import pandas as pd

REQUIRED_TBL1 = ["battery_id", "cycle_id", "t", "V", "I", "T_cell", "SoC", "dSoC", "C_rate"]
REQUIRED_TBL2 = ["episode_id", "battery_id", "cycle_id", "subset", "SoC_init", "T_init", "SoC_target", "len_sec"]
REQUIRED_TBL3 = ["battery_id", "protocol_id", "V_max", "T_max", "C_max"]

def ensure_columns(df: pd.DataFrame, required: Iterable[str], name="table"):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    return df

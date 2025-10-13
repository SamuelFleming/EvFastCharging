# src/utils/schema.py
from typing import Iterable
import pandas as pd

# === MVP-required columns derived from your processed CSVs ===
# Tbl1_signals.csv (time-series per cycle)
REQUIRED_TBL1 = [
    "cell_id", "battery_id", "subset",
    "cycle_id", "t", "V", "I", "T_cell",
    "SoC", "dSoC", "C_rate", "capacity",
    "overV", "overT", "phase",
]

# Tbl2_episodes.csv (per-episode summary)
REQUIRED_TBL2 = [
    "episode_id", "cell_id", "cycle_id", "subset",
    "SoC_init", "T_init", "SoC_target", "len_sec",
]

# Tbl3_metadata.csv (per-dataset constraints) — if/when present
# If you don’t have Tbl3 yet, keep this list here for forward-compat.
REQUIRED_TBL3 = [
    "cell_id", "subset", "protocol_id", "chemistry",
    "V_max", "T_max", "C_max_Crate",
]

def ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str = "table"):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    return df

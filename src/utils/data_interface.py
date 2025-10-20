# src/utils/data_interface.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any, Literal
import pandas as pd
import json, random
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


def _safe_read_json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_dataset_meta_from_processed(processed_dir: Path | str) -> dict:
    """
    Tries to read <processed_dir>/metadata/live_dataset.json; returns {} if missing/bad.
    """
    processed_dir = Path(processed_dir)
    cand = processed_dir / "metadata" / "live_dataset.json"
    return _safe_read_json(cand) if cand.exists() else {}

def get_limits(processed_dir: Path | str) -> dict:
    """
    Unified limits/capacity getter for RL/baselines:
      - Prefer Tbl3 via load_mvp_tables()
      - Fill missing fields from live_dataset.json if present
      - Fallback to safe MVP defaults
    Returns keys: V_max, T_max, capacity_Ah, i_cut_c
    """
    _, _, meta_tbl = load_mvp_tables(processed_dir, with_soh="none")
    meta_json = load_dataset_meta_from_processed(processed_dir)

    V_max = float(meta_tbl.get("V_max", meta_json.get("V_max", 3.65)))
    T_max = float(meta_tbl.get("T_max", meta_json.get("T_max", 55.0)))

    # Capacity & i_cut_c aren’t always in Tbl3; pull from JSON if available
    capacity_Ah = float(meta_json.get("capacity_Ah", 3.0))
    i_cut_c = float(meta_json.get("i_cut_c", 0.05))

    return {
        "V_max": V_max,
        "T_max": T_max,
        "capacity_Ah": capacity_Ah,
        "i_cut_c": i_cut_c,
    }

def get_init_distribution(dataset_meta: dict | None) -> tuple[tuple[float,float], list[float], tuple[float,float]]:
    """
    Extracts init distributions from dataset_meta (likely live_dataset.json contents).
    Returns: (soc_low, soc_high), temperature_set_C, (soh_low, soh_high)
    Defaults: SoC∈[0.1,0.3], T∈{20,35}, SoH∈[0.85,1.0]
    """
    meta = dataset_meta or {}
    soc_low, soc_high = 0.1, 0.3
    temps = [20.0, 35.0]
    soh_low, soh_high = 0.85, 1.0

    soc_cfg = meta.get("init_soc_range")
    if isinstance(soc_cfg, (list, tuple)) and len(soc_cfg) == 2:
        soc_low, soc_high = float(soc_cfg[0]), float(soc_cfg[1])

    temps_cfg = meta.get("episode_temperatures_C")
    if isinstance(temps_cfg, (list, tuple)) and len(temps_cfg) > 0:
        temps = [float(t) for t in temps_cfg]

    soh_cfg = meta.get("soh_range")
    if isinstance(soh_cfg, (list, tuple)) and len(soh_cfg) == 2:
        soh_low, soh_high = float(soh_cfg[0]), float(soh_cfg[1])

    return (soc_low, soc_high), temps, (soh_low, soh_high)

def sample_episode_context(dataset_meta: dict | None) -> dict:
    """
    Samples a single episode context consistent with dataset_meta.
    Returns {"soc0","temp_C","soh"} using get_init_distribution defaults if needed.
    """
    (soc_low, soc_high), temps, (soh_low, soh_high) = get_init_distribution(dataset_meta or {})
    return {
        "soc0": random.uniform(float(soc_low), float(soc_high)),
        "temp_C": random.choice(temps),
        "soh": random.uniform(float(soh_low), float(soh_high)),
    }
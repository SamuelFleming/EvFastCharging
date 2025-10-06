# src/nasa_data_extract/extract_charge_cycles.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Union

from read import (
    get_root, get_subsetMeta, list_mat_files,
    load_mat_safely, get_field, iter_cycles, to_1d
)

def _ensure_1d_or_none(x):
    if x is None:
        return None
    return np.atleast_1d(np.asarray(x))

def extract_charge_cycles_from_file(path: Path) -> pd.DataFrame:
    """Return tidy charge-cycle rows from a single .mat file."""
    d = load_mat_safely(path)
    keys = [k for k in d.keys() if not str(k).startswith("__")]
    group_label = path.relative_to(get_root()).parts[0]
    meta = get_subsetMeta().get(group_label, {})
    dfs = []

    def process_struct(s, battery_id: str):
        for idx, c in enumerate(iter_cycles(s), start=1):
            ctype = get_field(c, "type", "Type", "operation", "name")
            ctype = (str(ctype).lower() if ctype is not None else "")
            if "charge" not in ctype:
                continue

            data = get_field(c, "data", "Data", "measurements")
            if data is None:
                continue

            t = _ensure_1d_or_none(to_1d(get_field(data, "Time", "time", "t")))
            V = _ensure_1d_or_none(to_1d(get_field(data, "Voltage_measured", "voltage_measured", "voltage", "V")))
            I = _ensure_1d_or_none(to_1d(get_field(data, "Current_measured", "current_measured", "current", "I")))
            T = _ensure_1d_or_none(to_1d(get_field(data, "Temperature_measured", "temperature_measured", "temperature", "Temp", "T")))
            Q = _ensure_1d_or_none(to_1d(get_field(data, "Capacity", "capacity", "Q")))

            cols = [x for x in (t, V, I, T, Q) if x is not None and x.size > 0]
            if not cols:
                continue
            n = max(x.shape[0] for x in cols)

            def pad(x):
                if x is None:
                    return np.full(n, np.nan, dtype=float)
                x = np.asarray(x, dtype=float).reshape(-1)
                if x.shape[0] < n:
                    return np.pad(x, (0, n - x.shape[0]), constant_values=np.nan)
                if x.shape[0] > n:
                    return x[:n]
                return x

            t, V, I, T, Q = map(pad, (t, V, I, T, Q))

            if np.isfinite(t).any():
                t = t - np.nanmin(t)
            else:
                t = np.arange(n, dtype=float)

            row = pd.DataFrame({
                "subset": group_label,
                "battery_id": path.stem,
                "cycle_id": idx,
                "t": t, "V": V, "I": I, "T_cell": T, "capacity": Q
            }).dropna(subset=["t", "V", "I"])

            for k, v in meta.items():
                row[k] = str(v)
            dfs.append(row)

    if keys:
        for k in keys:
            process_struct(d[k], battery_id=path.stem)
    else:
        process_struct(d, battery_id=path.stem)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def build_charge_dataframe(
    mat_paths: Optional[Iterable[Path]] = None,
    subset: Optional[Union[str, Iterable[str]]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Extract and concatenate charge cycles from paths or a subset filter."""
    if mat_paths is None:
        mat_paths = list_mat_files(subset=subset, limit=limit)
    all_charge = []
    for p in mat_paths:
        dfp = extract_charge_cycles_from_file(p)
        if not dfp.empty:
            all_charge.append(dfp)
    return pd.concat(all_charge, ignore_index=True) if all_charge else pd.DataFrame()




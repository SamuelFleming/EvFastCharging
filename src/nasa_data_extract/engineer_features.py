# src/nasa_data_extract/engineer_features.py
from __future__ import annotations
import numpy as np
import pandas as pd

NOMINAL_Q_AH_DEFAULT = 2.0

def resample_and_engineer(df: pd.DataFrame, nominal_q_ah: float = NOMINAL_Q_AH_DEFAULT) -> pd.DataFrame:
    if df.empty:
        return df

    out = []
    numeric_cols = ["t","V","I","T_cell","capacity"]  # we aggregate these
    for (subset, b, c), g in df.groupby(["subset", "battery_id", "cycle_id"], sort=False):
        g = g.sort_values("t").copy()
        g.index = pd.to_datetime(g["t"].astype(float), unit="s", origin="unix")

        # mean only on numeric columns, then re-attach IDs
        gr = g[numeric_cols].resample("1s").mean().interpolate(method="time", limit=5)

        gr["t"] = (gr.index - gr.index[0]).total_seconds()
        gr["subset"], gr["battery_id"], gr["cycle_id"] = subset, b, c

        gr["C_rate"] = gr["I"] / nominal_q_ah
        qn = nominal_q_ah * 3600.0
        i = np.nan_to_num(gr["I"].values, nan=0.0)
        soc = np.cumsum(i * 1.0) / qn
        gr["SoC"]  = np.clip(soc, 0.0, 1.2)
        gr["dSoC"] = np.concatenate([[0.0], np.diff(gr["SoC"])])

        out.append(gr.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


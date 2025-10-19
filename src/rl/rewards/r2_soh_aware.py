# src/rl/rewards/r2_soh_aware.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .base_reward import BaseReward
from .soh_aware_vmax import SoHAwareVmax

@dataclass
class R2SoHAware(BaseReward):
    Vmax_nominal: float
    soh: float = 1.0
    k_drop_perc: float = 0.10
    lambda_V: float = 10.0
    lambda_T: float = 10.0
    lambda_N: float = 10.0

    def __post_init__(self):
        self._vmax_fn = SoHAwareVmax(self.Vmax_nominal, self.k_drop_perc)

    def __call__(self, *, dt_s: float, state: Dict, action: float, limits: Dict):
        V = float(state.get("V", 0.0))
        T = float(state.get("T", 25.0))
        Nep = state.get("Nep", None)
        Tmax = float(limits.get("T_max", 55.0))

        Vmax_eff = self._vmax_fn(float(state.get("SoH", self.soh)))
        overV = 1 if V > Vmax_eff else 0
        overT = 1 if T > Tmax else 0
        nep_zero = 1 if (Nep is not None and Nep <= 0.0) else 0

        penalty = self.lambda_V * overV + self.lambda_T * overT + self.lambda_N * nep_zero
        reward = -dt_s - penalty

        return reward, {
            "overV": overV,
            "overT": overT,
            "nep_zero_event": nep_zero,
            "Vmax_eff": Vmax_eff,
        }

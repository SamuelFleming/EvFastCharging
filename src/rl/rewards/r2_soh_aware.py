# src/rl/rewards/r2_soh_aware.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .base_reward import BaseReward
from .soh_aware_vmax import SoHAwareVmax

@dataclass
class R2SoHAware(BaseReward):
    # --- configuration ---
    # effective Vmax(SoH) = Vmax_nominal * (1 - k_drop_perc * (1 - SoH))
    Vmax_nominal: float = 4.2
    k_drop_perc: float = 0.10
    soh: float = 0.75

    # penalty weights (same semantics as R1)
    lambda_V: float = 10.0
    lambda_T: float = 10.0
    lambda_N: float = 10.0

    # progress shaping (encourage positive dSoC)
    k_progress: float = 200.0

    # internal
    def __post_init__(self):
        # build the SoH-aware ceiling function once
        self._vmax_fn = SoHAwareVmax(Vmax_nominal=self.Vmax_nominal,
                                     k_drop_perc=self.k_drop_perc)

    def __call__(self, *, dt_s: float, state: Dict, action: float, limits: Dict):
        V = float(state.get("V", 0.0))
        T = float(state.get("T", 25.0))
        Nep = state.get("Nep", None)
        dSoC = float(state.get("dSoC", 0.0))
        Tmax = float(limits.get("T_max", 55.0))

        # SoH-aware voltage ceiling
        Vmax_eff = float(self._vmax_fn(float(state.get("SoH", self.soh))))

        overV = 1 if V > Vmax_eff else 0
        overT = 1 if T > Tmax else 0
        nep_zero = 1 if (Nep is not None and Nep <= 0.0) else 0

        penalty = self.lambda_V * overV + self.lambda_T * overT + self.lambda_N * nep_zero
        reward = self.k_progress * max(0.0, dSoC) - dt_s - penalty

        return reward, {
            "overV": overV,
            "overT": overT,
            "nep_zero_event": nep_zero,
            "Vmax_eff": Vmax_eff
        }

# src/rl/rewards/r1_penalty.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .base_reward import BaseReward

@dataclass
class R1Penalty(BaseReward):
    lambda_V: float = 10.0
    lambda_T: float = 10.0
    lambda_N: float = 10.0
    k_progress: float = 200.0  # NEW

    def __call__(self, *, dt_s: float, state: Dict, action: float, limits: Dict):
        V = float(state.get("V", 0.0))
        T = float(state.get("T", 25.0))
        Nep = state.get("Nep", None)
        dSoC = float(state.get("dSoC", 0.0))  # NEW

        Vmax_eff = float(limits.get("Vmax_eff", limits.get("V_max_nominal", 3.65)))
        Tmax = float(limits.get("T_max", 55.0))

        overV = 1 if V > Vmax_eff else 0
        overT = 1 if T > Tmax else 0
        nep_zero = 1 if (Nep is not None and Nep <= 0.0) else 0

        penalty = self.lambda_V * overV + self.lambda_T * overT + self.lambda_N * nep_zero
        # encourage positive SoC progress, still pay -1 per second
        reward = self.k_progress * max(0.0, dSoC) - dt_s - penalty

        return reward, {"overV": overV, "overT": overT, "nep_zero_event": nep_zero, "Vmax_eff": Vmax_eff}


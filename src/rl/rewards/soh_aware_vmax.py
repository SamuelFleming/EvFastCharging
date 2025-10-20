# src/rl/rewards/soh_aware_vmax.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SoHAwareVmax:
    Vmax_nominal: float
    k_drop_perc: float = 0.10  # 10% drop at SoH=0 (linear)

    def __call__(self, soh: float) -> float:
        s = max(0.0, min(1.0, float(soh)))
        return self.Vmax_nominal * (1.0 - self.k_drop_perc * (1.0 - s))

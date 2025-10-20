# src/rl/rewards/base_reward.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Tuple

class BaseReward(ABC):
    """Return (reward, info_updates) per step."""
    @abstractmethod
    def __call__(self, *, dt_s: float, state: Dict, action: float, limits: Dict) -> Tuple[float, Dict]:
        ...

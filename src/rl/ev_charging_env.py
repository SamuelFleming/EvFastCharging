# src/rl/ev_charging_env.py
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.utils.data_interface import (
    load_mvp_tables,
    load_dataset_meta_from_processed,
    get_limits,
    sample_episode_context,
)
from src.rl.rewards.base_reward import BaseReward
from src.rl.rewards.r1_penalty import R1Penalty

# ---------------------------
# Backend interface
# ---------------------------
class SimulatorBackend:
    def reset(self, seed: Optional[int] = None) -> Dict[str, float]:
        raise NotImplementedError
    def step(self, action_C: float, dt_s: float) -> Tuple[Dict[str,float], bool, Dict[str,Any]]:
        """Advance by dt_s with action in C-rate (positive = charging). Returns (obs, terminated, info)."""
        raise NotImplementedError

# ---------------------------
# MVP Surrogate backend
# ---------------------------
@dataclass
class SurrogateParams:
    cap_Ah: float = 3.0       # nominal capacity
    R_ohm: float = 0.025      # internal resistance
    Tamb_C: float = 25.0
    m_kg: float = 0.045       # mass of 18650-ish cell
    Cp_J_per_kgK: float = 900 # heat capacity
    h_W_per_K: float = 0.35   # lumped cooling
    ocv_floor: float = 3.0
    ocv_clip_min: float = 3.1
    ocv_clip_max: float = 4.3

class SurrogateBackend(SimulatorBackend):
    """
    Single-node OCV-R model + simple thermal.
    I convention: charging current is NEGATIVE (like PyBaMM).
    Action (C-rate) is positive and converted to negative amps.
    """
    def __init__(self, V_max: float, T_max: float, target_soc: float,
                 soc_init_range=(0.1, 0.3), params: Optional[SurrogateParams] = None):
        self.V_max = float(V_max)
        self.T_max = float(T_max)
        self.target_soc = float(target_soc)
        self.soc_init_range = soc_init_range
        self.p = params or SurrogateParams()
        self.state = {"soc": 0.2, "T": self.p.Tamb_C, "V": 3.6}
        self.t = 0.0

    def set_ambient(self, Tamb_C: float):
        self.p.Tamb_C = float(Tamb_C)

    def _ocv(self, soc: float) -> float:
        s = min(max(soc, 1e-6), 1 - 1e-6)
        ocv = 3.2 + 0.9*s + 0.25*math.log(s) - 0.2*math.log(1 - s)
        return float(np.clip(ocv, self.p.ocv_clip_min, self.p.ocv_clip_max))

    def reset(self, seed: Optional[int] = None) -> Dict[str, float]:
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        s0 = random.uniform(*self.soc_init_range)
        self.state = {"soc": s0, "T": self.p.Tamb_C, "V": self._ocv(s0)}
        self.t = 0.0
        return {"SoC": self.state["soc"], "V": self.state["V"], "T": self.state["T"]}

    def step(self, action_C: float, dt_s: float) -> Tuple[Dict[str,float], bool, Dict[str,Any]]:
        # Convert action to current (A); charge is negative current
        I_A = -float(action_C) * self.p.cap_Ah
        # SoC update by coulomb counting
        dsoc = -(I_A * dt_s) / (3600.0 * self.p.cap_Ah)  # minus because I<0 increases SoC
        soc = float(np.clip(self.state["soc"] + dsoc, 0.0, 1.0))
        # Electrical model
        V_oc = self._ocv(soc)
        V = V_oc - I_A * self.p.R_ohm
        # Thermal
        T = self.state["T"]
        q_gen = (I_A**2) * self.p.R_ohm             # W
        q_loss = self.p.h_W_per_K * (T - self.p.Tamb_C)
        dT = (q_gen - q_loss) * dt_s / (self.p.m_kg * self.p.Cp_J_per_kgK)
        T = float(T + dT)

        self.state.update({"soc": soc, "V": float(V), "T": T})
        self.t += dt_s

        overV = 1 if V > self.V_max else 0
        overT = 1 if T > self.T_max else 0
        reached = 1 if soc >= self.target_soc else 0
        terminated = bool(overV or overT or reached or soc >= 0.999)

        obs = {"SoC": soc, "V": float(V), "T": T}
        info = {"t_s": self.t, "overV": overV, "overT": overT, "reached": reached, "I_A": I_A}
        return obs, terminated, info

# ---------------------------
# Gym environment wrapper
# ---------------------------
class EVChargingEnv(gym.Env):
    """
    Observation: [SoC, dSoC, V, T]
    Action: C-rate in [action_low, action_high]
    Reward: provided by a Reward implementation (R1/R2)
    """
    metadata = {"render.modes": []}

    def __init__(self,
                 processed_dir: str = "data/processed",
                 dt_s: float = 1.0,
                 target_soc: float = 0.8,
                 action_bounds_C: Tuple[float,float] = (-0.05, 4.0),
                 reward_impl: Optional[BaseReward] = None,
                 backend: Optional[SimulatorBackend] = None,
                 soh: Optional[float] = None,
                 max_steps: int = 1800,            # NEW: episode horizon (seconds if dt_s=1)
                 reach_bonus: float = 1000.0):     # NEW: terminal bonus
        super().__init__()
        # Load meta (V_max, T_max)
        # Load shared limits (Tbl3 and/or live_dataset.json)
        lims = get_limits(processed_dir)
        self.V_max = float(lims["V_max"])
        self.T_max = float(lims["T_max"])
        self.dt_s = float(dt_s)
        self.target_soc = float(target_soc)
        self.max_steps = int(max_steps)
        self.reach_bonus = float(reach_bonus)
        self._steps = 0

        # Keep dataset meta around for episode sampling
        self.dataset_meta = load_dataset_meta_from_processed(processed_dir)
        self.soh = float(self.dataset_meta.get("SoH", 1.0)) if soh is None else float(soh)

        #Add counters
        self._overV_count = 0
        self._overT_count = 0
        self._viol_term_thresh = 3  # terminate only if sustained for 3 ticks

        # Backend
        cap = float(lims.get("capacity_Ah", 3.0))
        if backend is None:
            params = SurrogateParams(cap_Ah=cap)
            self.backend = SurrogateBackend(self.V_max, self.T_max, self.target_soc, params=params)
        else:
            self.backend = backend

        # Reward
        self.reward_impl: BaseReward = reward_impl or R1Penalty()

        # Spaces
        self.action_low, self.action_high = action_bounds_C
        self.action_space = spaces.Box(low=np.array([self.action_low], dtype=np.float32),
                                       high=np.array([self.action_high], dtype=np.float32))
        # SoC, dSoC, V, T
        high = np.array([1.0, 1.0, 6.0, 120.0], dtype=np.float32)
        low  = np.array([0.0, -1.0, 0.0, -40.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._last_soc = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        ctx = sample_episode_context(self.dataset_meta)  # {"soc0","temp_C","soh"}
        self.backend.set_ambient(ctx["temp_C"])
        obs_dict = self.backend.reset(seed=seed)

        # Overwrite initial SoC to match sampled soc0
        obs_dict["SoC"] = float(ctx["soc0"])
        self.backend.state["soc"] = float(ctx["soc0"])
        self.backend.state["V"] = self.backend._ocv(float(ctx["soc0"]))

        # Use sampled SoH for this episode unless user forced a value
        if not hasattr(self, "soh") or self.soh is None:
            self.soh = float(ctx["soh"])

        self._last_soc = float(obs_dict["SoC"])
        obs = np.array([obs_dict["SoC"], 0.0, obs_dict["V"], obs_dict["T"]], dtype=np.float32)
        self._steps = 0
        return obs, {}

    def step(self, action):
        # 1) Action clamp
        a = float(np.clip(action[0], self.action_low, self.action_high))

        # 2) Advance backend by dt
        obs_d, _, info_backend = self.backend.step(a, self.dt_s)

        # 3) Compute dSoC and keep for rewards/metrics
        dSoC = float(obs_d["SoC"] - (self._last_soc if self._last_soc is not None else obs_d["SoC"]))
        self._last_soc = float(obs_d["SoC"])

        # 4) Build state for reward
        state = {
            "SoC": obs_d["SoC"],
            "dSoC": dSoC,
            "V":   obs_d["V"],
            "T":   obs_d["T"],
            "Nep": None,      # surrogate doesn't model plating yet
            "SoH": self.soh,
        }
        limits = {
            "V_max_nominal": self.V_max,
            "T_max": self.T_max,
        }

        # 5) Reward
        reward, info_upd = self.reward_impl(dt_s=self.dt_s, state=state, action=a, limits=limits)

        # 6) Effective limits/flags returned by the reward (e.g., R2 has SoH-aware Vmax)
        Vmax_eff = float(info_upd.get("Vmax_eff", self.V_max))
        overV_eff = int(obs_d["V"] > Vmax_eff)
        overT_eff = int(obs_d["T"] > self.T_max)

        self._overV_count = (self._overV_count + 1) if overV_eff else 0
        self._overT_count = (self._overT_count + 1) if overT_eff else 0

        # 7) Termination & truncation (true terminals: any over-limit or reached target)
        reached = 1 if obs_d["SoC"] >= self.target_soc else 0
        self._steps += 1
        timeout = int(self._steps >= self.max_steps)

        sustained_violation = (self._overV_count >= self._viol_term_thresh) or (self._overT_count >= self._viol_term_thresh)
        terminated = bool(sustained_violation or reached or obs_d["SoC"] >= 0.999)
        truncated  = bool(timeout and not terminated)

        # 8) Terminal bonus (only when we truly reached the target)
        if reached:
            reward += self.reach_bonus
            # After computing reward (and possibly adding reach_bonus), sanitize:
        reward = float(np.nan_to_num(reward, nan=-1e6, posinf=-1e6, neginf=-1e6))

        # 9) Observation vector and info
        # Build obs then sanitize:
        obs = np.array([obs_d["SoC"], dSoC, obs_d["V"], obs_d["T"]], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        info = {
            "t_s": info_backend["t_s"],
            "overV": overV_eff,
            "overT": overT_eff,
            "reached": reached,
            "I_A": info_backend["I_A"],
            "Vmax_eff": Vmax_eff,
            "SoC": obs_d["SoC"],
            "V":   obs_d["V"],
            "T":   obs_d["T"],
            "dSoC": dSoC,
            "nep_zero_event": int(info_upd.get("nep_zero_event", 0)),
        }
        return obs, reward, terminated, truncated, info

    # Episode metrics aggregator
    @staticmethod
    def compute_metrics(episode_infos: list[dict], target_soc: float) -> Dict[str, Any]:
        t_last = episode_infos[-1]["t_s"] if episode_infos else 0.0
        reached = any(info.get("reached", 0) for info in episode_infos)
        v_viol = sum(info.get("overV", 0) for info in episode_infos)
        t_viol = sum(info.get("overT", 0) for info in episode_infos)
        n_viol = sum(info.get("nep_zero_event", 0) for info in episode_infos)
        return {
            "time_s": t_last,
            "reached": int(reached),
            "overV_events": v_viol,
            "overT_events": t_viol,
            "nep_zero_events": n_viol,
            "delta_soh_proxy": float("nan"),
        }

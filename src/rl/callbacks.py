# src/rl/callbacks.py
from __future__ import annotations
from typing import List
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLogger(BaseCallback):
    """
    Minimal per-episode logger for off-policy algos (TD3/SAC/DDPG).
    Accumulates episode returns + metrics using the env's compute_metrics().
    """
    def __init__(self, env_ref, steps_per_ep: int,
                 out_returns: List[float], out_metrics: List[dict]):
        super().__init__()
        self.env_ref = env_ref
        self.steps_per_ep = int(steps_per_ep)
        self.out_returns = out_returns
        self.out_metrics = out_metrics
        self._ep_return = 0.0
        self._ep_infos = []
        self._step_in_ep = 0

    def _on_step(self) -> bool:
        # rewards, infos, dones come from SB3 runner locals
        rew = float(self.locals["rewards"])
        infos = self.locals["infos"]
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        done = bool(self.locals["dones"])

        self._ep_return += rew
        self._ep_infos.append(info)
        self._step_in_ep += 1

        if done or self._step_in_ep >= self.steps_per_ep:
            metrics = self.env_ref.compute_metrics(self._ep_infos, target_soc=self.env_ref.target_soc)
            self.out_returns.append(self._ep_return)
            self.out_metrics.append(metrics)
            # reset episode accumulators
            self._ep_return = 0.0
            self._ep_infos = []
            self._step_in_ep = 0
        return True

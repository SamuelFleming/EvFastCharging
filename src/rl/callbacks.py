# src/rl/callbacks.py
from __future__ import annotations
from pathlib import Path
import time
from typing import List
import pandas as pd, time
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLogger(BaseCallback):
    """
    Minimal per-episode logger for off-policy algos (TD3/SAC/DDPG).
    Accumulates episode returns + metrics using the env's compute_metrics().
    """
    def __init__(self, env_ref, steps_per_ep: int,
                 out_returns: List[float], out_metrics: List[dict], outdir=None):
        super().__init__()
        self.env_ref = env_ref
        self.steps_per_ep = int(steps_per_ep)
        self.out_returns = out_returns
        self.out_metrics = out_metrics
        self._ep_return = 0.0
        self._ep_infos = []
        self._step_in_ep = 0
        self.outdir = Path(outdir) if outdir is not None else None
        self._traj = []  # holds dicts per step
        self._saved_traj = False  # <-- ensure it exists

    def _on_step(self) -> bool:
        # rewards, infos, dones come from SB3 runner locals
        rew = float(self.locals["rewards"])
        infos = self.locals["infos"]
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        done = bool(self.locals["dones"])

        self._ep_return += rew
        self._ep_infos.append(info)
        self._step_in_ep += 1

        self._traj.append({
            "t_s": info.get("t_s", 0.0),
            "SoC": info.get("SoC", None),   # we’ll inject this from the env step (next snippet)
            "V":   info.get("V", None),
            "I_A": info.get("I_A", None),
            "Vmax_eff": info.get("Vmax_eff", None),
        })

        if done or self._step_in_ep >= self.steps_per_ep:
            #save the first trajectory we see
            if self.outdir is not None and self._traj and not self._saved_traj:
                self.outdir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                pd.DataFrame(self._traj).to_csv(self.outdir / f"episode_traj_{ts}.csv", index=False)
                self._saved_traj = True
            self._traj = []
            #save per episode metrics
            metrics = self.env_ref.compute_metrics(self._ep_infos, target_soc=self.env_ref.target_soc)
            self.out_returns.append(self._ep_return)
            self.out_metrics.append(metrics)
            # reset episode accumulators
            self._ep_return = 0.0
            self._ep_infos = []
            self._step_in_ep = 0
            self._traj = []
        return True

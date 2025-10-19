# src/rl/callbacks.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLogger(BaseCallback):
    """
    Minimal per-episode logger for off-policy algos (TD3/SAC/DDPG).
    Accumulates per-episode returns + metrics using the env's compute_metrics().
    Also saves the FIRST complete trajectory as episode_traj.csv in outdir.
    """
    def __init__(
        self,
        env_ref,
        steps_per_ep: int,
        out_returns: List[float],
        out_metrics: List[dict],
        outdir: Optional[Path] = None,
        seed: Optional[int] = None,
        reward_variant: str = "R1",
        target_soc: float = 0.8,
    ):
        super().__init__()
        self.env_ref = env_ref
        self.steps_per_ep = int(steps_per_ep)
        self.out_returns = out_returns
        self.out_metrics = out_metrics
        self.outdir = Path(outdir) if outdir is not None else None
        self.seed = None if seed is None else int(seed)
        self.reward_variant = str(reward_variant)
        self.target_soc = float(target_soc)

        self._ep_return = 0.0
        self._ep_infos = []
        self._step_in_ep = 0
        self._traj = []
        self._saved_traj = False

    def _on_step(self) -> bool:
        # SB3 runner locals
        rew = float(self.locals["rewards"])
        infos = self.locals["infos"]
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        done = bool(self.locals["dones"])

        self._ep_return += rew
        self._ep_infos.append(info)
        self._step_in_ep += 1

        # Collect trajectory for the first complete episode
        self._traj.append({
            "t_s": info.get("t_s", 0.0),
            "SoC": info.get("SoC", None),
            "V":   info.get("V", None),
            "I_A": info.get("I_A", None),
            "T":   info.get("T", None),
            "Vmax_eff": info.get("Vmax_eff", None),
        })

        if done or self._step_in_ep >= self.steps_per_ep:
            # Save the first completed episode trajectory (stable filename)
            if self.outdir is not None and self._traj and not self._saved_traj:
                self.outdir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(self._traj).to_csv(self.outdir / "episode_traj.csv", index=False)
                self._saved_traj = True

            # Compute per-episode metrics
            metrics = self.env_ref.compute_metrics(self._ep_infos, target_soc=self.target_soc)
            # Add run context
            metrics.update({
                "seed": self.seed,
                "reward_variant": self.reward_variant,
                "target_soc": self.target_soc,
            })
            self.out_returns.append(self._ep_return)
            self.out_metrics.append(metrics)

            # Reset episode accumulators
            self._ep_return = 0.0
            self._ep_infos = []
            self._step_in_ep = 0
            self._traj = []
        return True


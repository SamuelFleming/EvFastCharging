# src/rl/callbacks.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLogger(BaseCallback):
    """
    Logs per-episode return + metrics using env.compute_metrics().
    Saves the FIRST complete trajectory as episode_traj.csv in outdir.
    """
    def __init__(
        self,
        env_ref,
        out_returns: List[float],
        out_metrics: List[dict],
        outdir: Optional[Path] = None,
        seed: Optional[int] = None,
        reward_variant: str = "R1",
        target_soc: float = 0.8,
    ):
        super().__init__()
        self.env_ref = env_ref
        self.out_returns = out_returns
        self.out_metrics = out_metrics
        self.outdir = Path(outdir) if outdir is not None else None
        self.seed = None if seed is None else int(seed)
        self.reward_variant = str(reward_variant)
        self.target_soc = float(target_soc)

        self._ep_return = 0.0
        self._ep_infos = []
        self._traj = []
        self._saved_traj = False

    def _on_step(self) -> bool:
        rew = float(self.locals["rewards"])
        infos = self.locals["infos"]
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        done = bool(self.locals["dones"])

        self._ep_return += rew
        self._ep_infos.append(info)

        self._traj.append({
            "t_s": info.get("t_s", 0.0),
            "SoC": info.get("SoC", None),
            "V":   info.get("V", None),
            "I_A": info.get("I_A", None),
            "T":   info.get("T", None),
            "Vmax_eff": info.get("Vmax_eff", None),
        })

        if done:
            if self.outdir is not None and self._traj and not self._saved_traj:
                df = pd.DataFrame(self._traj)
                num_cols = ["SoC","V","I_A","T","Vmax_eff"]
                for c in num_cols:
                    if c in df:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["SoC","V"])
                self.outdir.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.outdir / "episode_traj.csv", index=False)
                self._saved_traj = True

            metrics = self.env_ref.compute_metrics(self._ep_infos, target_soc=self.target_soc)
            metrics.update({"seed": self.seed, "reward_variant": self.reward_variant, "target_soc": self.target_soc})
            self.out_returns.append(self._ep_return)
            self.out_metrics.append(metrics)

            self._ep_return = 0.0
            self._ep_infos = []
            self._traj = []

        return True


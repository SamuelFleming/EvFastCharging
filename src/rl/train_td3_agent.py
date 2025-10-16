# src/rl/train_td3_agent.py
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")


import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from src.rl.ev_charging_env import EVChargingEnv
from src.rl.callbacks import EpisodeLogger
from src.rl.rewards.r1_penalty import SoHAwareVmax

def main():
    base_out = Path("data/processed/rl/mvp_td3")
    base_out.mkdir(parents=True, exist_ok=True)

    # Create a per-run subfolder (timestamp)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = base_out / run_ts
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- Env params (MVP) -----
    env = EVChargingEnv(
        processed_dir="data/processed",
        dt_s=1.0,
        target_soc=0.6,          # you can set 0.6 while you’re on CC-only baseline, if you prefer
        action_bounds_C=(-0.05, 4.0),
        lambda_V=10.0, lambda_T=10.0
    )

    # Make Vmax SoH-aware (10% drop at SoH=0 → gentle effect)
    env.vmax_fn = SoHAwareVmax(Vmax_nominal=env.V_max, k_drop_perc=0.10)
    # Optional: force a demo SoH value (otherwise it uses metadata SoH or 1.0)
    env.soh = 0.75

    # ----- TD3 config -----
    n_episodes = 10
    steps_per_ep = 1800          # 30 minutes horizon @ 1 s
    total_steps = n_episodes * steps_per_ep

    # Exploration noise (small; action in C-rate)
    action_noise = NormalActionNoise(mean=np.array([0.0]), sigma=np.array([0.2]))

    model = TD3(
        "MlpPolicy", env,
        learning_rate=1e-3,
        buffer_size=50_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        verbose=0,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    returns, ep_metrics = [], []
    callback = EpisodeLogger(env, steps_per_ep, returns, ep_metrics, outdir=outdir)

    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)

    

    # ----- Save artefacts -----
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_path = outdir / f"td3_model_{ts}.zip"
    model.save(model_path.as_posix())

    # Metrics CSV
    mdf = pd.DataFrame(ep_metrics)
    mdf.to_csv(outdir / f"td3_episode_metrics_{ts}.csv", index=False)

    # Reward curve
    plt.figure()
    plt.plot(returns, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("TD3: Episode Return (MVP)")
    plt.tight_layout()
    plt.savefig(outdir / f"td3_returns_{ts}.png", dpi=140)
    plt.close()

    # ✅ Make the V–SoC plot if a trajectory exists
    traj_files = sorted(outdir.glob("episode_traj_*.csv"))
    if traj_files:
        traj = pd.read_csv(traj_files[-1]).dropna(subset=["SoC", "V"])
        if not traj.empty:
            plt.figure()
            plt.plot(traj["SoC"], traj["V"])
            plt.xlabel("SoC"); plt.ylabel("Voltage [V]")
            plt.title("RL Policy: V vs SoC (one episode)")
            plt.tight_layout()
            plt.savefig(outdir / "rl_v_vs_soc.png", dpi=140)
            plt.close()

    # Enriched summary (with knobs)
    summary = {
        "episodes": len(returns),
        "mean_return": float(np.mean(returns)) if returns else None,
        "mean_time_s": float(np.mean(mdf["time_s"])) if not mdf.empty else None,
        "reached_frac": float(np.mean(mdf["reached"])) if not mdf.empty else None,
        "knobs": {
            "env": {
                "dt_s": float(env.dt_s),
                "target_soc": float(env.target_soc),
                "lambda_V": float(env.lambda_V),
                "lambda_T": float(env.lambda_T),
                "action_bounds_C": [float(env.action_low), float(env.action_high)],
                "soh": float(getattr(env, "soh", 1.0)),
                "V_max_nominal": float(env.V_max),
                "T_max": float(env.T_max),
                "k_drop_perc": float(getattr(env.vmax_fn, "k_drop_perc", np.nan)) if env.vmax_fn else None,
            },
            "algo": {
                "algo": "TD3",
                "total_steps": int(total_steps),
                "episodes": int(n_episodes),
                "steps_per_ep": int(steps_per_ep),
                "learning_rate": 1e-3,
                "batch_size": int(model.batch_size),
                "action_noise_sigma": 0.2,
                "net_arch": [128, 128],
            },
        },
        "artifacts": {
            "model_zip": model_path.name,
            "metrics_csv": "td3_episode_metrics.csv",
            "returns_png": "td3_returns.png",
            "traj_csv": traj_files[-1].name if traj_files else None,
            "v_soc_png": "rl_v_vs_soc.png" if traj_files else None,
        }
    }
    (outdir / "td3_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved TD3 artefacts to", outdir)

if __name__ == "__main__":
    main()

# src/rl/train_td3_agent.py
from __future__ import annotations
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from src.rl.ev_charging_env import EVChargingEnv
from src.rl.callbacks import EpisodeLogger

def main():
    outdir = Path("data/processed/rl/mvp_td3")
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- Env params (MVP) -----
    env = EVChargingEnv(
        processed_dir="data/processed",
        dt_s=1.0,
        target_soc=0.8,          # you can set 0.6 while you’re on CC-only baseline, if you prefer
        action_bounds_C=(-0.05, 4.0),
        lambda_V=10.0, lambda_T=10.0
    )

    # ----- TD3 config -----
    n_episodes = 15
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
    callback = EpisodeLogger(env, steps_per_ep, returns, ep_metrics)

    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)

    # ----- Training loop (episode-aware logging) -----
    ep_metrics = []
    returns = []
    obs, _ = env.reset(seed=42)
    ep_return = 0.0
    ep_infos = []

    for step in range(total_steps):
        # 1) act
        action, _ = model.predict(obs, deterministic=False)

        # 2) env step
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 3) store transition in replay buffer
        done_flag = bool(terminated or truncated)
        # SB3 expects obs, next_obs, action, reward, done, infos=[info]
        model.replay_buffer.add(
            obs,
            next_obs,
            action,
            reward,
            done_flag,
            infos=[info],  # <-- list, not None
        )

        # 4) one gradient step
        model.train(gradient_steps=1, batch_size=model.batch_size)

        # 5) bookkeeping
        ep_return += float(reward)
        ep_infos.append(info)
        obs = next_obs  # advance

        # 6) episode end (either env terminated or we hit the fixed horizon)
        if done_flag or ((step + 1) % steps_per_ep == 0):
            metrics = env.compute_metrics(ep_infos, target_soc=env.target_soc)
            returns.append(ep_return)
            ep_metrics.append(metrics)

            # reset episode
            obs, _ = env.reset()
            ep_return = 0.0
            ep_infos = []

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

    # Simple text summary
    summary = {
        "episodes": len(returns),
        "mean_return": float(np.mean(returns)) if returns else None,
        "mean_time_s": float(np.mean(mdf["time_s"])) if not mdf.empty else None,
        "reached_frac": float(np.mean(mdf["reached"])) if not mdf.empty else None,
    }
    (outdir / f"td3_summary_{ts}.json").write_text(json.dumps(summary, indent=2))
    print("Saved TD3 artefacts to", outdir)

if __name__ == "__main__":
    main()

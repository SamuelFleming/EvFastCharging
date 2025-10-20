# src/rl/train_td3_agent.py
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")

import json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from src.rl.ev_charging_env import EVChargingEnv
from src.rl.callbacks import EpisodeLogger
from src.rl.rewards.r1_penalty import R1Penalty
from src.rl.rewards.r2_soh_aware import R2SoHAware
from src.utils.rl_registry import RunRegistry, load_dataset_meta

def append_rl_registry(registry_path: Path, row: dict, field_order: list[str]):
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = registry_path.exists()
    with registry_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in field_order})

def parse_net_arch(s: str) -> list[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return [128, 128]

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train TD3 for EV fast-charging (MVP)")
    # Run control
    ap.add_argument("--out-root", type=Path, default=Path("data/processed/rl/mvp_td3"))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps-per-ep", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", type=str, default=None, help="Optional friendly name for this run (no spaces recommended).")
    ap.add_argument("--notes", type=str, default=None, help="Optional free-text notes to include in summary and registry.")
    ap.add_argument("--dataset-meta", type=Path, default="data\processed\metadata\live_dataset.json",
                help="Path to live_dataset.json; defaults to <processed-dir>/metadata/live_dataset.json if present.")


    # Env
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--dt-s", type=float, default=1.0)
    ap.add_argument("--target-soc", type=float, default=0.8)
    ap.add_argument("--action-low", type=float, default=-0.05)
    ap.add_argument("--action-high", type=float, default=4.0)
    ap.add_argument("--soh", type=float, default=None)
    ap.add_argument("--v-max-nominal", type=float, default=None)  # if None, use meta
    ap.add_argument("--t-max", type=float, default=None)          # if None, use meta

    # Reward selection and penalties
    ap.add_argument("--reward", type=str, choices=["r1", "r2"], default="r1")
    ap.add_argument("--lambda-v", type=float, default=10.0)
    ap.add_argument("--lambda-t", type=float, default=10.0)
    ap.add_argument("--lambda-n", type=float, default=10.0)
    ap.add_argument("--vmax-drop-perc", type=float, default=0.10)  # for R2

    # Algo (TD3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer-size", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--noise-sigma", type=float, default=0.2)
    ap.add_argument("--net-arch", type=str, default="128,128")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    # ----- Per-run directory -----
    args.out_root.mkdir(parents=True, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    slug = (args.run_name.strip().replace(" ", "_") if args.run_name else None)
    run_id = f"{run_ts}_{slug}" if slug else run_ts
    run_dir = args.out_root / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----- One-liner config banner -----
    print(
        f"[TD3 {run_id}] episodes={args.episodes} steps/ep={args.steps_per_ep} "
        f"reward={args.reward.upper()} dt={args.dt_s}s target_soc={args.target_soc} "
        f"bounds=[{args.action_low},{args.action_high}] seed={args.seed}"
    )

    dataset_meta = load_dataset_meta(args.dataset_meta, args.processed_dir)
    dataset_id = dataset_meta.get("dataset_id") or dataset_meta.get("id")
    dataset_run_id = dataset_meta.get("dataset_run_id") or dataset_meta.get("run_id")

    # ----- Reward wiring -----
    # Create a temporary env to read meta (V_max/T_max) then rebuild with chosen reward
    env_probe = EVChargingEnv(processed_dir=str(args.processed_dir))
    V_nom = float(args.v_max_nominal) if args.v_max_nominal is not None else float(env_probe.V_max)
    T_max = float(args.t_max) if args.t_max is not None else float(env_probe.T_max)

    soh_used = 0.75 if args.soh is None else float(args.soh)
    k_drop_used = float(args.vmax_drop_perc) if args.reward == "r2" else None

    if args.reward == "r1":
        reward_impl = R1Penalty(lambda_V=args.lambda_v, lambda_T=args.lambda_t, lambda_N=args.lambda_n)
        reward_variant_name = "R1"
    else:
        reward_impl = R2SoHAware(
            Vmax_nominal=V_nom,
            soh=soh_used,
            k_drop_perc=float(args.vmax_drop_perc),
            lambda_V=args.lambda_v, lambda_T=args.lambda_t, lambda_N=args.lambda_n
        )
        reward_variant_name = "R2"

    # ----- Final env (with reward) -----
    env = EVChargingEnv(
        processed_dir=str(args.processed_dir),
        dt_s=float(args.dt_s),
        target_soc=float(args.target_soc),
        action_bounds_C=(float(args.action_low), float(args.action_high)),
        reward_impl=reward_impl,
        soh=soh_used,max_steps=args.steps_per_ep,          # NEW
        reach_bonus=1000.0 
    )

    # TD3 config
    total_steps = int(args.episodes) * int(args.steps_per_ep)
    action_noise = NormalActionNoise(mean=np.array([0.0]), sigma=np.array([float(args.noise_sigma)]))

    model = TD3(
        "MlpPolicy", env,
        learning_rate=float(args.lr),
        buffer_size=int(args.buffer_size),
        batch_size=int(args.batch_size),
        tau=float(args.tau),
        gamma=float(args.gamma),
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        verbose=0,
        policy_kwargs=dict(net_arch=parse_net_arch(args.net_arch)),
        seed=int(args.seed),
    )

    returns, ep_metrics = [], []
    callback = EpisodeLogger(
        env_ref=env,
        out_returns=returns,
        out_metrics=ep_metrics,
        outdir=run_dir,
        seed=int(args.seed),
        reward_variant=reward_variant_name,
        target_soc=float(args.target_soc),
    )

    # ----- Learn -----
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)

    # ----- Artefacts -----
    # 1) Model
    model_path = run_dir / "td3_model.zip"
    model.save(model_path.as_posix())

    # 2) Episode metrics
    mdf = pd.DataFrame(ep_metrics)
    mdf.to_csv(run_dir / "td3_episode_metrics.csv", index=False)

    # 3) Returns curve
    plt.figure()
    plt.plot(returns, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"TD3: Episode Return ({reward_variant_name})")
    plt.tight_layout()
    plt.savefig(run_dir / "td3_returns.png", dpi=140)
    plt.close()

    # 4) V–SoC plot (if trajectory exists)
    traj_path = run_dir / "episode_traj.csv"
    if traj_path.exists():
        traj = pd.read_csv(traj_path).dropna(subset=["SoC", "V"])
        if not traj.empty:
            plt.figure()
            plt.plot(traj["SoC"], traj["V"])
            # overlay horizontal line if Vmax_eff present
            if "Vmax_eff" in traj.columns and np.isfinite(traj["Vmax_eff"].iloc[0]):
                vline = float(traj["Vmax_eff"].iloc[0])
                plt.axhline(vline, linestyle="--")
            plt.xlabel("SoC")
            plt.ylabel("Voltage [V]")
            plt.title(f"RL Policy: V vs SoC ({reward_variant_name})")
            plt.tight_layout()
            plt.savefig(run_dir / "rl_v_vs_soc.png", dpi=140)
            plt.close()

    # 5) Summary JSON
    summary = {
        "run_id": run_id,
        "run_name": (args.run_name if args.run_name else None),
        "algo": "TD3",
        "reward_variant": reward_variant_name,
        "seed": int(args.seed),
        "paths": {
            "run_dir": str(run_dir),
            "episode_metrics_csv": "td3_episode_metrics.csv",
            "episode_traj_csv": "episode_traj.csv" if (run_dir / "episode_traj.csv").exists() else None,
            "returns_plot": "td3_returns.png",
            "v_vs_soc_plot": "rl_v_vs_soc.png" if (run_dir / "rl_v_vs_soc.png").exists() else None,
            "model_zip": "td3_model.zip",
        },
        "knobs": {
            "env": {
                "dt_s": float(args.dt_s),
                "target_soc": float(args.target_soc),
                "action_bounds": [float(args.action_low), float(args.action_high)],
                "soh": float(soh_used),
                "V_max_nominal": float(V_nom),
                "T_max": float(T_max),
                "k_drop_perc": (float(k_drop_used) if k_drop_used is not None else None),
            },
            "algo": {
                "algo": "TD3",
                "total_steps": int(int(args.episodes) * int(args.steps_per_ep)),
                "episodes": int(args.episodes),
                "steps_per_ep": int(args.steps_per_ep),
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "buffer_size": int(args.buffer_size),
                "gamma": float(args.gamma),
                "tau": float(args.tau),
                "noise_sigma": float(args.noise_sigma),
                "net_arch": parse_net_arch(args.net_arch),
            },
        },
        "notes": (args.notes if args.notes else None),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "td3_summary.json").write_text(json.dumps(summary, indent=2))

    #6) Update Registry
    registry_path = args.out_root.parent / "registry.csv"  # data/processed/rl/registry.csv
    registry = RunRegistry(registry_path)

    row = {
        "timestamp": run_id,
        "run_name": (args.run_name or ""),
        "algo": "TD3",
        "reward_variant": reward_variant_name,
        "seed": int(args.seed),

        "dataset_id": dataset_id or "",
        "dataset_run_id": dataset_run_id or "",

        "dt_s": float(args.dt_s),
        "target_soc": float(args.target_soc),
        "action_low": float(args.action_low),
        "action_high": float(args.action_high),

        "v_max_used": float(V_nom),
        "t_max_used": float(T_max),
        "soh": float(soh_used),
        "k_drop_perc": (float(k_drop_used) if k_drop_used is not None else ""),

        "episodes": int(args.episodes),
        "steps_per_ep": int(args.steps_per_ep),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "buffer_size": int(args.buffer_size),
        "gamma": float(args.gamma),
        "tau": float(args.tau),
        "noise_sigma": float(args.noise_sigma),
        "net_arch": ";".join(map(str, parse_net_arch(args.net_arch))),

        "summary_path": str((run_dir / "td3_summary.json").resolve()),
        "out_dir": str(run_dir.resolve()),
        "notes": (args.notes or ""),
    }
    registry.append(row)
    print(f"Registry updated → {registry_path}")

    # Final log line
    reached_frac = float(np.mean(mdf["reached"])) if not mdf.empty else np.nan
    print(f"Saved artefacts → {run_dir}")
    print(f"Reached target episodes: {reached_frac*100:.1f}%")

if __name__ == "__main__":
    main()


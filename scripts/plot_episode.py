# scripts/plot_episode.py
#!/usr/bin/env python3
"""
Plot an episode trajectory from a CSV.

Expected columns (case-sensitive if present):
  t_s, SoC, V, I_A, T, Vmax_eff

- If Vmax_eff is missing, the dashed ceiling is omitted.
- You can pass --target-soc to draw a reference line on the SoC panel.
- Output PNG is saved alongside the CSV unless --out is provided.

Usage (PowerShell):
  & C:/Users/User/anaconda3/envs/EvFastCharge/python.exe scripts/plot_episode.py `
     --csv data/processed/rl/mvp_td3/r1_t80_short/episode_traj.csv `
     --target-soc 0.80

  & C:/Users/User/anaconda3/envs/EvFastCharge/python.exe scripts/plot_episode.py `
     --csv data/processed/rl/mvp_td3/r2_t80_short/episode_traj.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser("Plot episode trajectory (RL run)")
    ap.add_argument("--csv", required=True, help="Path to episode trajectory CSV")
    ap.add_argument("--out", default=None, help="Output PNG path (default: alongside CSV)")
    ap.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    ap.add_argument("--title", default=None, help="Custom title (default: inferred from parent dir)")
    ap.add_argument("--target-soc", type=float, default=None, help="Draw target SoC line on SoC panel (e.g., 0.8)")
    ap.add_argument("--trim", type=float, default=None, help="Optional max time (s) to trim the view")
    return ap.parse_args()


def nice_title(csv_path: Path, user_title: str | None) -> str:
    if user_title:
        return user_title
    # try to infer run name from parent folder (e.g., r1_t80_short)
    parent = csv_path.parent.name
    return f"Episode trajectory — {parent}"


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalise column presence
    # Rename tolerant variants if users used different headers
    rename_map = {
        "time_s": "t_s", "t": "t_s",
        "current_A": "I_A", "I": "I_A",
        "temp_C": "T", "T_cell": "T",
        "Vmax": "Vmax_eff", "vmax_eff": "Vmax_eff"
    }
    for a, b in rename_map.items():
        if a in df.columns and b not in df.columns:
            df[b] = df[a]
    # Ensure required minimums exist
    required_any = [["t_s", "SoC"], ["V"], ["I_A"], ["T"]]
    missing_groups = [grp for grp in required_any if not any(c in df.columns for c in grp)]
    if missing_groups:
        raise ValueError(f"CSV missing necessary columns; need at least t_s+SoC, V, I_A, T. Found: {list(df.columns)}")
    return df


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)

    # Basic vectors (with graceful fallbacks)
    t = df["t_s"].values if "t_s" in df.columns else np.arange(len(df))
    if args.trim is not None:
        mask = t <= float(args.trim)
        df = df.loc[mask].reset_index(drop=True)
        t = df["t_s"].values if "t_s" in df.columns else np.arange(len(df))

    soc = df["SoC"].values if "SoC" in df.columns else None
    v = df["V"].values if "V" in df.columns else None
    i = df["I_A"].values if "I_A" in df.columns else None
    temp = df["T"].values if "T" in df.columns else None
    vmax = df["Vmax_eff"].values if "Vmax_eff" in df.columns else None

    # Compute simple derived flags (over-V events per step)
    overV = None
    if vmax is not None and v is not None:
        overV = (v > vmax).astype(int)

    # --- Plot
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.3, 1.0, 1.0, 1.0], hspace=0.25)

    # A: Voltage vs SoC (with Vmax_eff)
    axA = fig.add_subplot(gs[0, 0])
    if soc is not None and v is not None:
        axA.plot(soc, v, linewidth=1.0)
        axA.set_xlabel("SoC")
        axA.set_ylabel("Voltage [V]")
        if vmax is not None:
            # draw Vmax vs SoC as dashed (same length)
            axA.plot(soc, vmax, linestyle="--", linewidth=1.0)
            # optional “overV” markers
            if overV is not None and overV.any():
                # mark over-V points faintly
                axA.scatter(soc[overV == 1], v[overV == 1], s=6)
        axA.set_title(nice_title(csv_path, args.title))
        axA.grid(True, alpha=0.25)

    # B: Current vs time
    axB = fig.add_subplot(gs[1, 0], sharex=None)
    if i is not None:
        axB.plot(t, i, linewidth=1.0)
        axB.set_ylabel("Current [A]")
        axB.set_xlabel("Time [s]")
        axB.grid(True, alpha=0.25)

    # C: Temperature vs time
    axC = fig.add_subplot(gs[2, 0], sharex=axB)
    if temp is not None:
        axC.plot(t, temp, linewidth=1.0)
        axC.set_ylabel("Cell Temp [°C]")
        axC.set_xlabel("Time [s]")
        axC.grid(True, alpha=0.25)

    # D: SoC vs time (with optional target)
    axD = fig.add_subplot(gs[3, 0], sharex=axB)
    if soc is not None:
        axD.plot(t, soc, linewidth=1.0)
        if args.target_soc is not None:
            axD.axhline(float(args.target_soc), linestyle="--", linewidth=1.0)
        axD.set_ylabel("SoC")
        axD.set_xlabel("Time [s]")
        axD.grid(True, alpha=0.25)

    # Tidy x-lims if trim not set
    if args.trim is None and "t_s" in df.columns:
        axB.set_xlim(df["t_s"].min(), df["t_s"].max())

    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    print(f"[OK] Saved plot → {out_path}")


if __name__ == "__main__":
    main()

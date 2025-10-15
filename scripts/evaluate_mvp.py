# scripts/evaluate_mvp.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_latest_rl_metrics(rl_dir: Path) -> Path:
    files = sorted(rl_dir.glob("td3_episode_metrics_*.csv"))
    if not files:
        raise FileNotFoundError(f"No RL metrics CSV found in {rl_dir}")
    return files[-1]

def main():
    ap = argparse.ArgumentParser(description="Evaluate MVP: Baseline vs RL (R1)")
    ap.add_argument("--baseline-dir", type=Path, default=Path("data/processed/baselines/cccv_smoke"),
                    help="Folder containing baseline_cccv_summary.csv")
    ap.add_argument("--rl-dir", type=Path, default=Path("data/processed/rl/mvp_td3"),
                    help="Folder containing td3_episode_metrics_*.csv")
    ap.add_argument("--outdir", type=Path, default=Path("data/processed/eval/mvp"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Load baseline summary ---
    base_path = args.baseline_dir / "baseline_cccv_summary.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing {base_path}")
    bdf = pd.read_csv(base_path)

    # Try common column names
    # Expect: time_to_target_s, reached_target (bool)
    if "time_to_target_s" not in bdf.columns:
        # fallback if name differs (rare)
        tt_candidates = [c for c in bdf.columns if "time" in c and "target" in c]
        if tt_candidates:
            bdf = bdf.rename(columns={tt_candidates[0]: "time_to_target_s"})
        else:
            raise ValueError(f"Baseline summary lacks time_to_target_s; columns={list(bdf.columns)}")

    b_t = bdf["time_to_target_s"].dropna().to_numpy()
    baseline_t80_mean = float(np.mean(b_t)) if b_t.size else np.nan
    baseline_t80_sd   = float(np.std(b_t, ddof=1)) if b_t.size > 1 else np.nan
    baseline_n        = int(b_t.size)

    # --- Load latest RL metrics ---
    rl_metrics_path = load_latest_rl_metrics(args.rl_dir)
    rdf = pd.read_csv(rl_metrics_path)

    # Expect columns: time_s, reached, overV_events, overT_events
    for col in ["time_s","reached","overV_events","overT_events"]:
        if col not in rdf.columns:
            raise ValueError(f"RL metrics missing '{col}' in {rl_metrics_path}")

    rl_time_mean = float(rdf["time_s"].mean()) if not rdf.empty else np.nan
    rl_time_sd   = float(rdf["time_s"].std(ddof=1)) if len(rdf) > 1 else np.nan
    rl_reached_pct = float(100.0 * rdf["reached"].mean()) if not rdf.empty else np.nan
    rl_overV_mean  = float(rdf["overV_events"].mean()) if not rdf.empty else np.nan
    rl_overT_mean  = float(rdf["overT_events"].mean()) if not rdf.empty else np.nan

    # --- Print a concise report ---
    print("\n=== MVP Evaluation ===")
    print(f"Baseline t80:  mean={baseline_t80_mean:.1f}s  sd={baseline_t80_sd if np.isfinite(baseline_t80_sd) else float('nan'):.1f}s  (n={baseline_n})")
    print(f"RL time_s:     mean={rl_time_mean:.1f}s  sd={rl_time_sd if np.isfinite(rl_time_sd) else float('nan'):.1f}s  (n={len(rdf)})")
    print(f"RL reached %:  {rl_reached_pct:.1f}%")
    print(f"RL violations: overV={rl_overV_mean:.2f}  overT={rl_overT_mean:.2f}")
    print(f"Sources -> baseline: {base_path.name}   RL: {rl_metrics_path.name}")

    # --- Save comparison table ---
    comp = pd.DataFrame([
        {"metric":"time_to_target_s_mean", "baseline":baseline_t80_mean, "rl":rl_time_mean},
        {"metric":"time_to_target_s_sd",   "baseline":baseline_t80_sd,   "rl":rl_time_sd},
        {"metric":"reached_percent",       "baseline":np.nan,            "rl":rl_reached_pct},
        {"metric":"overV_events_mean",     "baseline":np.nan,            "rl":rl_overV_mean},
        {"metric":"overT_events_mean",     "baseline":np.nan,            "rl":rl_overT_mean},
    ])
    comp_path = args.outdir / "mvp_comparison.csv"
    comp.to_csv(comp_path, index=False)

    # --- Save tiny bar plot (time-to-target) ---
    plt.figure()
    x = ["Baseline t80", "RL time"]
    y = [baseline_t80_mean, rl_time_mean]
    plt.bar(x, y)
    plt.ylabel("Seconds")
    plt.title("Time-to-target: Baseline vs RL (MVP)")
    plt.tight_layout()
    png_path = args.outdir / "mvp_time_compare.png"
    plt.savefig(png_path, dpi=140)
    plt.close()

    # --- Save a JSON summary for quick slide paste ---
    summary = {
        "baseline": {"t80_mean_s": baseline_t80_mean, "t80_sd_s": baseline_t80_sd, "n": baseline_n},
        "rl": {"time_mean_s": rl_time_mean, "time_sd_s": rl_time_sd, "reached_percent": rl_reached_pct,
               "overV_mean": rl_overV_mean, "overT_mean": rl_overT_mean, "episodes": int(len(rdf))},
        "sources": {"baseline_summary": base_path.name, "rl_metrics": rl_metrics_path.name},
    }
    (args.outdir / "mvp_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {comp_path}\nSaved: {png_path}\nSaved: {args.outdir/'mvp_summary.json'}")

if __name__ == "__main__":
    main()

# scripts/evaluate_mvp.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_latest_run_dir(rl_root: Path) -> Path | None:
    """Return newest subfolder that contains td3_episode_metrics.csv, else None."""
    if not rl_root.exists():
        return None
    cand = []
    for p in rl_root.iterdir():
        if p.is_dir() and (p / "td3_episode_metrics.csv").exists():
            cand.append(p)
    return max(cand, key=lambda p: p.stat().st_mtime) if cand else None

def load_rl_metrics_path(rl_dir: Path, rl_run_dir: Path | None) -> Path:
    """
    Resolve the RL metrics CSV path under either:
      - explicit run dir (preferred), or
      - latest subfolder under rl_dir, or
      - legacy flat file td3_episode_metrics_*.csv under rl_dir.
    """
    # 1) explicit run dir
    if rl_run_dir is not None:
        metrics = rl_run_dir / "td3_episode_metrics.csv"
        if not metrics.exists():
            raise FileNotFoundError(f"Expected {metrics} in --rl-run-dir")
        return metrics

    # 2) latest subfolder
    latest = find_latest_run_dir(rl_dir)
    if latest is not None:
        metrics = latest / "td3_episode_metrics.csv"
        if metrics.exists():
            return metrics

    # 3) legacy flat files
    legacy = sorted(rl_dir.glob("td3_episode_metrics_*.csv"))
    if legacy:
        return legacy[-1]

    raise FileNotFoundError(
        f"No RL metrics found. Looked for a run subfolder with td3_episode_metrics.csv "
        f"or legacy td3_episode_metrics_*.csv under {rl_dir}"
    )

def main():
    ap = argparse.ArgumentParser(description="Evaluate MVP: Baseline vs RL (R1)")
    ap.add_argument("--baseline-dir", type=Path, default=Path("data/processed/baselines/cccv_smoke"),
                    help="Folder containing baseline_cccv_summary.csv")
    ap.add_argument("--rl-dir", type=Path, default=Path("data/processed/rl/mvp_td3"),
                    help="Parent folder that contains run subfolders")
    ap.add_argument("--rl-run-dir", type=Path, default=None,
                    help="(Optional) Specific run subfolder (e.g., data/processed/rl/mvp_td3/20251016_093020)")
    ap.add_argument("--outdir", type=Path, default=Path("data/processed/eval/mvp"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Load baseline summary ---
    base_path = args.baseline_dir / "baseline_cccv_summary.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing {base_path}")
    bdf = pd.read_csv(base_path)

    if "time_to_target_s" not in bdf.columns:
        tt_candidates = [c for c in bdf.columns if "time" in c and "target" in c]
        if tt_candidates:
            bdf = bdf.rename(columns={tt_candidates[0]: "time_to_target_s"})
        else:
            raise ValueError(f"Baseline summary lacks time_to_target_s; columns={list(bdf.columns)}")

    b_t = bdf["time_to_target_s"].dropna().to_numpy()
    baseline_t80_mean = float(np.mean(b_t)) if b_t.size else np.nan
    baseline_t80_sd   = float(np.std(b_t, ddof=1)) if b_t.size > 1 else np.nan
    baseline_n        = int(b_t.size)

    # --- Load RL metrics (auto or explicit run) ---
    rl_metrics_path = load_rl_metrics_path(args.rl_dir, args.rl_run_dir)
    rdf = pd.read_csv(rl_metrics_path)

    # Expect columns: time_s, reached, overV_events, overT_events
    for col in ["time_s","reached","overV_events","overT_events"]:
        if col not in rdf.columns:
            raise ValueError(f"RL metrics missing '{col}' in {rl_metrics_path}")

    # Episode-time stats
    ep_time_mean = float(rdf["time_s"].mean()) if not rdf.empty else np.nan
    ep_time_sd   = float(rdf["time_s"].std(ddof=1)) if len(rdf) > 1 else np.nan

    # t80 from reached episodes only
    reached_mask = rdf["reached"].astype(bool)
    rl_t80 = rdf.loc[reached_mask, "time_s"].to_numpy()
    rl_t80_mean = float(np.mean(rl_t80)) if rl_t80.size else np.nan
    rl_t80_sd   = float(np.std(rl_t80, ddof=1)) if rl_t80.size > 1 else np.nan
    rl_reached_pct = float(100.0 * reached_mask.mean()) if not rdf.empty else np.nan

    # violations
    rl_overV_mean  = float(rdf["overV_events"].mean()) if not rdf.empty else np.nan
    rl_overT_mean  = float(rdf["overT_events"].mean()) if not rdf.empty else np.nan

    # --- Print ---
    run_dir_for_msg = (rl_metrics_path.parent if rl_metrics_path.name == "td3_episode_metrics.csv"
                       else args.rl_dir)
    print("\n=== MVP Evaluation ===")
    print(f"Baseline t80:      mean={baseline_t80_mean:.1f}s  sd={baseline_t80_sd if np.isfinite(baseline_t80_sd) else float('nan'):.1f}s  (n={baseline_n})")
    print(f"RL episode time:   mean={ep_time_mean:.1f}s  sd={ep_time_sd if np.isfinite(ep_time_sd) else float('nan'):.1f}s  (n={len(rdf)})")
    print(f"RL t80 (reached):  mean={rl_t80_mean if np.isfinite(rl_t80_mean) else float('nan'):.1f}s  sd={rl_t80_sd if np.isfinite(rl_t80_sd) else float('nan'):.1f}s  reached%={rl_reached_pct:.1f}%")
    print(f"RL violations:     overV={rl_overV_mean:.2f}  overT={rl_overT_mean:.2f}")
    print(f"Sources -> baseline: {base_path.name}   RL: {rl_metrics_path.name} (run: {run_dir_for_msg})")

    # --- Save comparison table ---
    comp = pd.DataFrame([
        {"metric":"baseline_t80_mean_s",   "baseline":baseline_t80_mean, "rl":rl_t80_mean},
        {"metric":"t80_sd_s",              "baseline":baseline_t80_sd,   "rl":rl_t80_sd},
        {"metric":"rl_episode_time_mean_s","baseline":np.nan,            "rl":ep_time_mean},
        {"metric":"rl_episode_time_sd_s",  "baseline":np.nan,            "rl":ep_time_sd},
        {"metric":"reached_percent",       "baseline":np.nan,            "rl":rl_reached_pct},
        {"metric":"overV_events_mean",     "baseline":np.nan,            "rl":rl_overV_mean},
        {"metric":"overT_events_mean",     "baseline":np.nan,            "rl":rl_overT_mean},
    ])
    comp_path = args.outdir / "mvp_comparison.csv"
    comp.to_csv(comp_path, index=False)

    # --- Save tiny bar plot (time-to-target) ---
    plt.figure()
    have_rl_t80 = np.isfinite(rl_t80_mean)
    x = ["Baseline t80", "RL t80" if have_rl_t80 else "RL episode time"]
    y = [baseline_t80_mean, rl_t80_mean if have_rl_t80 else ep_time_mean]
    plt.bar(x, y)
    plt.ylabel("Seconds")
    plt.title("Time-to-target: Baseline vs RL (MVP)")
    plt.tight_layout()
    png_path = args.outdir / "mvp_time_compare.png"
    plt.savefig(png_path, dpi=140)
    plt.close()

    # --- Save JSON summary ---
    summary = {
        "baseline": {"t80_mean_s": baseline_t80_mean, "t80_sd_s": baseline_t80_sd, "n": baseline_n},
        "rl": {
            "episode_time_mean_s": ep_time_mean,
            "episode_time_sd_s": ep_time_sd,
            "t80_mean_s": rl_t80_mean,
            "t80_sd_s": rl_t80_sd,
            "reached_percent": rl_reached_pct,
            "overV_mean": rl_overV_mean,
            "overT_mean": rl_overT_mean,
            "episodes": int(len(rdf)),
        },
        "sources": {
            "baseline_summary": base_path.name,
            "rl_metrics": rl_metrics_path.name,
            "rl_run_dir": str(run_dir_for_msg),
        },
    }
    (args.outdir / "mvp_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {comp_path}\nSaved: {png_path}\nSaved: {args.outdir/'mvp_summary.json'}")

if __name__ == "__main__":
    main()

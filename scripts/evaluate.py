#!/usr/bin/env python
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- IO helpers ----------

def newest_subdir(root: Path) -> Optional[Path]:
    subs = [p for p in root.iterdir() if p.is_dir()]
    return max(subs, key=lambda p: p.stat().st_mtime) if subs else None

def discover_baselines(baseline_root: Path) -> List[Path]:
    out = []
    for p in baseline_root.iterdir():
        if p.is_dir() and (p / "baseline_cccv_summary.csv").exists():
            out.append(p)
    return sorted(out)

def load_baseline_csvs(dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for d in dirs:
        p = d / "baseline_cccv_summary.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["__src_dir"] = str(d)
            frames.append(df)
        else:
            print(f"[WARN] missing baseline file: {p}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_rl_episode_metrics(dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for d in dirs:
        p = d / "td3_episode_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["__rl_dir"] = str(d)
            frames.append(df)
        else:
            print(f"[WARN] missing RL metrics: {p}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def try_load_registry(csv_path: Path) -> pd.DataFrame:
    try:
        if csv_path.exists():
            return pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] failed to read registry {csv_path}: {e}")
    return pd.DataFrame()


# ---------- Filters & summaries ----------

def ci95(mean: float, sd: float, n: int) -> Tuple[float, float]:
    if n <= 1 or not np.isfinite(sd): return mean, 0.0
    half = 1.96 * (sd / math.sqrt(n))
    return mean, half

def filter_baselines(df: pd.DataFrame,
                     c_rate: Optional[float],
                     soc_range: Optional[Tuple[float, float]]) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    if c_rate is not None and "c_rate" in out.columns:
        out = out[np.isclose(out["c_rate"].astype(float), float(c_rate))]
    if soc_range is not None and "soc_init" in out.columns:
        lo, hi = soc_range
        out = out[(out["soc_init"].astype(float) >= lo) & (out["soc_init"].astype(float) <= hi)]
    return out

def summarise_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {"n": 0}
    # speed on reached rows only
    dft = df[df.get("reached_target", 0) == 1]
    n_t = len(dft)
    mean_t = float(dft["time_to_target_s"].mean()) if n_t > 0 else float("nan")
    sd_t   = float(dft["time_to_target_s"].std(ddof=1)) if n_t > 1 else float("nan")
    mean_t, half_t = ci95(mean_t, sd_t, max(n_t, 1))

    def agg(col):
        if col not in df.columns: return (float("nan"), 0.0)
        n = len(df)
        m = float(df[col].mean()) if n>0 else float("nan")
        sd = float(df[col].std(ddof=1)) if n>1 else float("nan")
        m, h = ci95(m, sd, max(n,1))
        return m, h

    mOV, hOV = agg("overV_events")
    mOT, hOT = agg("overT_events")
    mNP, hNP = agg("nep_zero_events")

    return {
        "n_episodes": int(len(df)),
        "n_reached": int(n_t),
        "t80_mean": mean_t, "t80_ci": half_t,
        "overV_mean": mOV, "overV_ci": hOV,
        "overT_mean": mOT, "overT_ci": hOT,
        "nep0_mean": mNP, "nep0_ci": hNP,
    }

def summarise_rl(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {"n": 0}
    dft = df[df.get("reached", 0) == 1]
    n_t = len(dft)
    mean_t = float(dft["time_s"].mean()) if n_t > 0 else float("nan")
    sd_t   = float(dft["time_s"].std(ddof=1)) if n_t > 1 else float("nan")
    mean_t, half_t = ci95(mean_t, sd_t, max(n_t, 1))

    def agg(col):
        if col not in df.columns: return (float("nan"), 0.0)
        n = len(df)
        m = float(df[col].mean()) if n>0 else float("nan")
        sd = float(df[col].std(ddof=1)) if n>1 else float("nan")
        m, h = ci95(m, sd, max(n,1))
        return m, h

    mOV, hOV = agg("overV_events")
    mOT, hOT = agg("overT_events")
    mNP, hNP = agg("nep_zero_events")

    return {
        "n_episodes": int(len(df)),
        "n_reached": int(n_t),
        "t80_mean": mean_t, "t80_ci": half_t,
        "overV_mean": mOV, "overV_ci": hOV,
        "overT_mean": mOT, "overT_ci": hOT,
        "nep0_mean": mNP, "nep0_ci": hNP,
    }


# ---------- Plotting ----------

def write_bar(figpath: Path, rows: List[Tuple[str, Dict[str, Any]]], metric_key: str, ylabel: str):
    labels = [k for k,_ in rows]
    means  = [float(v.get(metric_key + "_mean", np.nan)) for _, v in rows]
    cis    = [float(v.get(metric_key + "_ci", 0.0)) for _, v in rows]
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, means, yerr=cis, capsize=6)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(metric_key.replace("_", " ").upper())
    plt.tight_layout()
    plt.savefig(figpath, dpi=140)
    plt.close()


# ---------- Registry-aware run picking (optional) ----------

def pick_runs_by_dataset(baseline_root: Path, rl_root: Path, dataset_id: str) -> Tuple[List[Path], List[Path]]:
    breg = try_load_registry(baseline_root / "baselines_registry.csv")
    rreg = try_load_registry(rl_root.parent / "registry.csv")  # rl_root like data/processed/rl/mvp_td3
    bdirs, rdirs = [], []
    if not breg.empty and "dataset_id" in breg.columns:
        hits = breg[breg["dataset_id"].astype(str) == dataset_id]
        for _, row in hits.tail(1).iterrows():
            rel = row.get("out_dir", "")
            p = baseline_root / Path(rel)
            if (p / "baseline_cccv_summary.csv").exists():
                bdirs.append(p if p.is_absolute() else baseline_root / rel)
    if not rreg.empty and "dataset_id" in rreg.columns:
        hits = rreg[rreg["dataset_id"].astype(str) == dataset_id]
        for _, row in hits.tail(3).iterrows():  # grab a few latest RL runs
            out_dir = Path(row.get("out_dir", ""))
            if (out_dir / "td3_episode_metrics.csv").exists():
                rdirs.append(out_dir)
    return bdirs, rdirs


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser("evaluate — RQ1 speed & safety with 95% CIs")
    ap.add_argument("--baseline-dirs", nargs="*", type=Path, default=[],
                    help="Baseline run dirs containing baseline_cccv_summary.csv")
    ap.add_argument("--baseline-root", type=Path, default=None,
                    help="If given and no --baseline-dirs, auto-pick newest under this root")
    ap.add_argument("--baseline-c", type=float, default=None, help="Filter baselines by C-rate (exact match)")
    ap.add_argument("--baseline-soc-range", nargs=2, type=float, default=None, metavar=("LOW","HIGH"),
                    help="Filter baselines by initial SoC window, e.g., 0.1 0.3")

    ap.add_argument("--rl-run-dirs", nargs="*", type=Path, default=[],
                    help="RL run dirs each containing td3_episode_metrics.csv")
    ap.add_argument("--rl-root", type=Path, default=None,
                    help="If given and no --rl-run-dirs, auto-pick newest under this root")

    ap.add_argument("--match-dataset", type=str, default=None,
                    help="If set, choose runs via registries that match this dataset_id")

    ap.add_argument("--labels", nargs="*", type=str, default=None,
                    help="Optional labels for rows (same length as selected series)")

    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Discover runs
    baseline_dirs = list(args.baseline_dirs)
    rl_dirs = list(args.rl_run_dirs)

    if args.match_dataset and args.baseline_root and args.rl_root:
        bdirs, rdirs = pick_runs_by_dataset(args.baseline_root, args.rl_root, args.match_dataset)
        if bdirs: baseline_dirs = bdirs
        if rdirs: rl_dirs = rdirs

    if not baseline_dirs and args.baseline_root:
        newest_b = discover_baselines(args.baseline_root)
        if newest_b: baseline_dirs = [newest_b[-1]]

    if not rl_dirs and args.rl_root:
        newest_r = newest_subdir(args.rl_root)
        if newest_r: rl_dirs = [newest_r]

    # --- Load
    bdf = load_baseline_csvs(baseline_dirs)
    # Coerce missing safety cols to zeros (back-compat with older baseline runs)
    for col in ("overV_events", "overT_events", "nep_zero_events"):
        if col not in bdf.columns:
            bdf[col] = 0
        else:
            bdf[col] = pd.to_numeric(bdf[col], errors="coerce").fillna(0)
    if args.baseline_soc_range is not None:
        lo, hi = map(float, args.baseline_soc_range)
        bdf = filter_baselines(bdf, args.baseline_c, (lo, hi))
    else:
        bdf = filter_baselines(bdf, args.baseline_c, None)
    rdf = load_rl_episode_metrics(rl_dirs)

    # --- Summaries
    rows: List[Tuple[str, Dict[str, Any]]] = []
    if not bdf.empty: rows.append(("Baseline", summarise_baseline(bdf)))

    # If multiple RL runs, show R1 vs R2 (label by reward_variant when available)
    if not rdf.empty:
        if "reward_variant" in rdf.columns:
            # group by variant and summarise
            for label, g in rdf.groupby("reward_variant", sort=False):
                rows.append((str(label), summarise_rl(g)))
        else:
            rows.append(("RL", summarise_rl(rdf)))

    # Optional relabel
    if args.labels and len(args.labels) == len(rows):
        rows = list(zip(args.labels, [v for _, v in rows]))

    # --- Save outputs
    table = [{"label": name, **stats} for name, stats in rows]
    (args.outdir / "summary_rq1.csv").write_text(pd.DataFrame(table).to_csv(index=False))
    (args.outdir / "summary_rq1.json").write_text(json.dumps({
        "rows": table,
        "sources": {
            "baselines": [str(d) for d in baseline_dirs],
            "rl_runs": [str(d) for d in rl_dirs]
        }
    }, indent=2))

    # Plots
    if rows:
        write_bar(args.outdir / "bar_t80.png", rows, "t80", "seconds")
        write_bar(args.outdir / "bar_overV.png", rows, "overV", "count/episode")
        write_bar(args.outdir / "bar_overT.png", rows, "overT", "count/episode")
        write_bar(args.outdir / "bar_nep0.png", rows, "nep0", "count/episode")

    # --- Console summary
    print("\n=== RQ1 Summary (mean ±95% CI) ===")
    for name, stats in rows:
        t80 = stats.get("t80_mean", float("nan"))
        t80ci = stats.get("t80_ci", 0.0)
        t80_str = f"{t80:.1f} ± {t80ci:.1f}" if np.isfinite(t80) else "n/a"
        print(f"{name:>10} | t80: {t80_str} | overV: {stats.get('overV_mean', np.nan):.2f}±{stats.get('overV_ci',0.0):.2f} "
              f"| overT: {stats.get('overT_mean', np.nan):.2f}±{stats.get('overT_ci',0.0):.2f} "
              f"| nep≈0: {stats.get('nep0_mean', np.nan):.2f}±{stats.get('nep0_ci',0.0):.2f}")
    print(f"\nWrote {args.outdir/'summary_rq1.csv'} and {args.outdir/'summary_rq1.json'}")

if __name__ == "__main__":
    main()

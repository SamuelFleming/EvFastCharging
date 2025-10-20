# src/eda/plots.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# ---- Column expectations (dataset-agnostic) ----
# We assume your unified “signals” CSV has these canonical names:
# required base: t, V, I, T_cell, battery_id, cycle_id
# optional derived: SoC, dSoC, C_rate, subset
REQ_BASE = ["t", "V", "I", "T_cell", "battery_id", "cycle_id"]
OPT = ["SoC", "dSoC", "C_rate", "subset", "SoH", "overV", "overT", "phase"]

def _has(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(c in df.columns for c in cols)

def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _downsample(df: pd.DataFrame, target: int = 20_000) -> pd.DataFrame:
    if len(df) <= target:
        return df
    step = max(1, len(df) // target)
    return df.iloc[::step, :]

# ---------------------------------------------------------------------------
#                            CORE STATS & EXPORTS
# ---------------------------------------------------------------------------

def _safe_minmax(a: pd.Series) -> Tuple[float | None, float | None]:
    if a is None or a.empty:
        return None, None
    return float(np.nanmin(a.values)), float(np.nanmax(a.values))

# ---- Stats (robust) ----
def quick_stats(df: pd.DataFrame) -> Dict[str, float | int | None | dict]:
    """Lightweight stats, now enriched with RL-relevant summaries."""
    stats: Dict[str, float | int | None | dict] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "batteries": int(df["battery_id"].nunique()) if "battery_id" in df else None,
        "cycles": int(df["cycle_id"].nunique()) if "cycle_id" in df else None,
        "subsets": int(df["subset"].nunique()) if "subset" in df else None,
    }
    for col in ["V", "I", "T_cell", "SoC", "C_rate"]:
        if col in df.columns and len(df[col]):
            mn, mx = _safe_minmax(df[col])
            stats[f"{col}_min"] = mn
            stats[f"{col}_max"] = mx
            stats[f"{col}_mean"] = float(np.nanmean(df[col].values))
            stats[f"{col}_p95"] = float(np.nanpercentile(df[col].values, 95))
            stats[f"{col}_p99"] = float(np.nanpercentile(df[col].values, 99))

    # Safety event counts if available
    if "overV" in df.columns:
        stats["overV_events"] = int(np.nansum(df["overV"].values))
    if "overT" in df.columns:
        stats["overT_events"] = int(np.nansum(df["overT"].values))

    # SoH coverage
    if "SoH" in df.columns:
        soh_nonnull = int(df["SoH"].notna().sum())
        stats["SoH_samples"] = soh_nonnull
        stats["SoH_presence_ratio"] = float(soh_nonnull / max(1, len(df)))
    return stats

def write_json(obj: dict, outdir: Path, name: str) -> Path:
    outdir = _ensure_outdir(outdir)
    p = outdir / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return p


def write_stats_json(df: pd.DataFrame, outdir: Path) -> Path:
    return write_json(quick_stats(df), outdir, "eda_summary.json")

# ---------------------------------------------------------------------------
#                                  PLOTS
# ---------------------------------------------------------------------------
def plot_voltage_vs_soc(df: pd.DataFrame, outdir: Path) -> Path | None:
    if not _has(df, ["V","SoC"]):
        return None
    outdir = _ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(6,4))
    d = _downsample(df[["SoC","V"]].dropna(), 40_000)
    ax.scatter(d["SoC"], d["V"], s=2, alpha=0.4)
    ax.set_xlabel("SoC")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Voltage vs SoC")
    p = outdir / "voltage_vs_soc.png"
    fig.savefig(p, bbox_inches="tight", dpi=140); plt.close(fig)
    return p

def plot_temp_vs_time(df: pd.DataFrame, outdir: Path) -> Path | None:
    if not _has(df, ["t","T_cell"]):
        return None
    outdir = _ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(6,4))
    d = _downsample(df[["t","T_cell"]].sort_values("t"))
    ax.plot(d["t"], d["T_cell"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Cell Temp [°C]")
    ax.set_title("Temperature vs Time")
    p = outdir / "temp_vs_time.png"
    fig.savefig(p, bbox_inches="tight", dpi=140); plt.close(fig)
    return p

def plot_crate_vs_time(df: pd.DataFrame, outdir: Path) -> Path | None:
    if not _has(df, ["t","C_rate"]):
        return None
    outdir = _ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(6,4))
    d = _downsample(df[["t","C_rate"]].sort_values("t"))
    ax.plot(d["t"], d["C_rate"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("C-rate")
    ax.set_title("C-rate profile")
    p = outdir / "crate_vs_time.png"
    fig.savefig(p, bbox_inches="tight", dpi=140); plt.close(fig)
    return p

def plot_cv_tail_by_cycle(df: pd.DataFrame, outdir: Path, v_thresh: float = 4.15) -> Path | None:
    need = ["battery_id","cycle_id","t","V"]
    if not _has(df, need):
        return None
    outdir = _ensure_outdir(outdir)

    def per_cycle(g: pd.DataFrame) -> float:
        g = g.sort_values("t")
        mask = g["V"] >= v_thresh
        if not mask.any(): return 0.0
        t = g.loc[mask, "t"].values
        return float(t.max() - t.min()) if len(t) else 0.0

    cv = (df.groupby(["battery_id","cycle_id"], sort=False)
            .apply(per_cycle).reset_index(name="cv_tail_sec"))
    if cv.empty:
        return None

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(range(len(cv)), cv["cv_tail_sec"], marker="o", ms=3, lw=1)
    ax.set_xlabel("Cycle index")
    ax.set_ylabel("CV tail [s]  (charge-side fade proxy)")
    ax.set_title(f"CV-tail duration per cycle (V ≥ {v_thresh:.2f} V)")
    p = outdir / "cv_tail_by_cycle.png"
    fig.savefig(p, bbox_inches="tight", dpi=140); plt.close(fig)
    return p

def plot_correlation(df: pd.DataFrame, outdir: Path) -> Path | None:
    cols = [c for c in ["V","I","T_cell","SoC","dSoC","C_rate"] if c in df.columns]
    if len(cols) < 2:
        return None
    outdir = _ensure_outdir(outdir)
    d = df[cols]
    if len(d) > 50_000:
        d = d.sample(50_000, random_state=0)
    corr = d.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation (sample)")
    p = outdir / "correlation.png"
    fig.savefig(p, bbox_inches="tight", dpi=140); plt.close(fig)
    return p

def plot_crate_hist(df: pd.DataFrame, outdir: Path) -> Path | None:
    if "C_rate" not in df.columns:
        return None
    outdir = _ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df["C_rate"].clip(-5, 10), bins=60)
    ax.set_xlabel("C-rate (clipped to [-5, 10])")
    ax.set_ylabel("Count")
    ax.set_title("C-rate distribution")
    p = outdir / "crate_hist.png"
    fig.savefig(p, bbox_inches="tight", dpi=140); plt.close(fig)
    return p

# ---- SoH helpers & plots -----------------------------------------------------

def compute_cv_tail_per_cycle(df: pd.DataFrame, v_thresh: float = 4.15) -> pd.DataFrame:
    """Return per-(battery_id,cycle_id) CV-tail duration in seconds."""
    need = ["battery_id","cycle_id","t","V"]
    if not _has(df, need):
        return pd.DataFrame(columns=["battery_id","cycle_id","cv_tail_sec"])

    def per_cycle(g: pd.DataFrame) -> float:
        g = g.sort_values("t")
        mask = g["V"] >= v_thresh
        if not mask.any():
            return 0.0
        t = g.loc[mask, "t"].values
        return float(t.max() - t.min()) if len(t) else 0.0

    cv = (df.groupby(["battery_id","cycle_id"], sort=False)
            .apply(per_cycle).reset_index(name="cv_tail_sec"))
    return cv

def plot_soh_vs_cycle(df_or_soh: pd.DataFrame, outdir: Path) -> Path | None:
    """
    Accepts either a DataFrame that already has ['battery_id','cycle_id','SoH'],
    or a larger signals table that includes SoH. Plots SoH vs cycle for each battery.
    """
    cols = {"battery_id","cycle_id","SoH"}
    if not cols.issubset(df_or_soh.columns):
        return None
    outdir = _ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(6,4))
    for bid, g in df_or_soh[["battery_id","cycle_id","SoH"]].dropna().groupby("battery_id"):
        gg = g.sort_values("cycle_id")
        ax.plot(gg["cycle_id"], gg["SoH"], marker="o", ms=3, lw=1, label=str(bid))
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH (capacity ratio)")
    ax.set_title("SoH trend by cycle")
    if df_or_soh["battery_id"].nunique() <= 8:
        ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    p = outdir / "soh_vs_cycle.png"
    fig.savefig(p, bbox_inches="tight", dpi=140); plt.close(fig)
    return p

def plot_soh_vs_cv_tail(df_signals: pd.DataFrame, outdir: Path, v_thresh: float = 4.15) -> Path | None:
    """
    From a signals table that includes SoH (per-row or joinable), compute per-cycle CV tail
    and scatter SoH vs CV tail to visualise relation between degradation and charge tail length.
    """
    need = {"battery_id","cycle_id","V","t"}
    if not need.issubset(df_signals.columns):
        return None
    outdir = _ensure_outdir(outdir)

    # Per-cycle cv tail
    cv = compute_cv_tail_per_cycle(df_signals, v_thresh=v_thresh)
    # Try to get SoH per cycle from the same DF
    if "SoH" in df_signals.columns:
        soh_per_cycle = (df_signals[["battery_id","cycle_id","SoH"]]
                         .dropna()
                         .drop_duplicates(["battery_id","cycle_id"]))
    else:
        # Not available → can't plot
        return None

    dfm = cv.merge(soh_per_cycle, on=["battery_id","cycle_id"], how="inner")
    if dfm.empty:
        return None

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(dfm["cv_tail_sec"], dfm["SoH"], s=8, alpha=0.5)
    ax.set_xlabel(f"CV tail [s] (V ≥ {v_thresh:.2f} V)")
    ax.set_ylabel("SoH")
    ax.set_title("SoH vs CV-tail duration (per cycle)")
    ax.grid(True, alpha=0.3)
    p = outdir / "soh_vs_cv_tail.png"
    fig.savefig(p, bbox_inches="tight", dpi=140); plt.close(fig)
    return p


# ------------------------ NEW: Additional EDA plots --------------------------

def plot_over_events_by_cycle(df: pd.DataFrame, outdir: Path) -> Dict[str, str] | None:
    """Two figure files: overV_by_cycle.png and overT_by_cycle.png."""
    need = ["battery_id", "cycle_id"]
    if not _has(df, need):
        return None
    outdir = _ensure_outdir(outdir)

    arts: Dict[str, str] = {}
    for col in ["overV", "overT"]:
        if col in df.columns:
            agg = (
                df.groupby(["battery_id", "cycle_id"], sort=False)[col]
                .sum(min_count=1)
                .reset_index(name=f"{col}_count")
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(len(agg)), agg[f"{col}_count"], lw=1)
            ax.set_xlabel("Cycle index")
            ax.set_ylabel(f"{col} count")
            ax.set_title(f"{col} events per cycle")
            p = outdir / f"{col}_by_cycle.png"
            fig.savefig(p, bbox_inches="tight", dpi=140)
            plt.close(fig)
            arts[f"{col}_by_cycle"] = str(p)
    return arts if arts else None


def plot_soc_hist(df: pd.DataFrame, outdir: Path) -> Path | None:
    if "SoC" not in df.columns:
        return None
    outdir = _ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["SoC"].clip(0, 1), bins=50)
    ax.set_xlabel("SoC")
    ax.set_ylabel("Count")
    ax.set_title("SoC distribution")
    p = outdir / "soc_hist.png"
    fig.savefig(p, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return p


def plot_current_vs_time(df: pd.DataFrame, outdir: Path) -> Path | None:
    if not _has(df, ["t", "I"]):
        return None
    outdir = _ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(6, 4))
    d = _downsample(df[["t", "I"]].sort_values("t"))
    ax.plot(d["t"], d["I"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Current [A] (sign per convention)")
    ax.set_title("Current vs Time")
    p = outdir / "current_vs_time.png"
    fig.savefig(p, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return p

# ----------------------------- Diagnostics ----------------------------------

def run_diagnostics(df: pd.DataFrame) -> Dict[str, object]:
    """Return missingness, completeness, and high-level warnings."""
    diag: Dict[str, object] = {}

    # Missingness & completeness
    miss = df.isna().sum().to_dict()
    nonnull_ratio = {k: float(1.0 - (v / max(1, len(df)))) for k, v in miss.items()}
    diag["missing_count"] = miss
    diag["nonnull_ratio"] = nonnull_ratio

    # Warnings
    warnings: list[str] = []
    if "SoH" not in df.columns or df["SoH"].notna().sum() < 5:
        warnings.append("SoH missing or very sparse — ageing analysis may be unreliable.")
    if "overV" not in df.columns and "overT" not in df.columns:
        warnings.append("Safety flags (overV/overT) not present — cannot compute safety metrics.")
    if "cycle_id" not in df.columns:
        warnings.append("cycle_id missing — cycle-based plots/metrics disabled.")
    if "SoC" in df.columns:
        smin, smax = df["SoC"].min(), df["SoC"].max()
        if smax < 0.8:
            warnings.append("Max SoC < 0.8 — t80 comparisons may be uninformative.")
    diag["warnings"] = warnings
    return diag


# --------------------------- RL Insights (read-only) -------------------------

def rl_insights(df: pd.DataFrame, meta_limits: dict | None = None) -> Dict[str, object]:
    """
    Suggest sane ranges/targets for RL based on empirical percentiles.
    DOES NOT write anything back to the dataset. Purely advisory.
    """
    ins: Dict[str, object] = {}

    # Percentile-based safety ceilings from the data
    if "V" in df.columns and len(df["V"]):
        ins["v_limit_p99"] = float(np.nanpercentile(df["V"], 99))
    if "T_cell" in df.columns and len(df["T_cell"]):
        ins["t_limit_p99"] = float(np.nanpercentile(df["T_cell"], 99))
    if "C_rate" in df.columns and len(df["C_rate"]):
        ins["crate_limit_p99"] = float(np.nanpercentile(df["C_rate"], 99))

    # Target SoC suggestion
    if "SoC" in df.columns and len(df["SoC"]):
        # Use 0.8 if feasible; otherwise take 90th percentile
        max_soc = float(np.nanmax(df["SoC"]))
        ins["target_soc_suggestion"] = float(min(0.80, max_soc)) if max_soc >= 0.8 else float(
            np.nanpercentile(df["SoC"], 90)
        )

    # Safety flags prevalence
    if "overV" in df.columns:
        ins["overV_prevalence"] = float(np.nanmean(df["overV"]))
    if "overT" in df.columns:
        ins["overT_prevalence"] = float(np.nanmean(df["overT"]))

    # SoH coverage indicator
    if "SoH" in df.columns:
        ins["soh_nonnull_ratio"] = float(df["SoH"].notna().mean())

    # Merge known dataset limits if provided (e.g., from Tbl3 or metadata json)
    if meta_limits:
        ins["dataset_limits"] = {
            k: float(v) for k, v in meta_limits.items() if k in ("V_max", "T_max", "capacity_Ah", "i_cut_c")
        }

    return ins


def write_diagnostics_json(df: pd.DataFrame, outdir: Path) -> Path:
    return write_json(run_diagnostics(df), outdir, "eda_diagnostics.json")
json.dumps

def write_rl_insights_json(df: pd.DataFrame, outdir: Path, meta_limits: dict | None = None) -> Path:
    return write_json(rl_insights(df, meta_limits=meta_limits), outdir, "rl_insights.json")



# ---- Convenience: run all available plots safely ----
def run_all(df: pd.DataFrame, outdir: Path) -> dict:
    outdir = Path(outdir)
    arts: dict = {}

    # JSONs
    arts["stats_json"] = str(write_stats_json(df, outdir))
    arts["diagnostics_json"] = str(write_diagnostics_json(df, outdir))

    # Plots
    for fn, name in [
        (plot_voltage_vs_soc, "voltage_vs_soc"),
        (plot_temp_vs_time, "temp_vs_time"),
        (plot_crate_vs_time, "crate_vs_time"),
        (plot_cv_tail_by_cycle, "cv_tail_by_cycle"),
        (plot_correlation, "correlation"),
        (plot_crate_hist, "crate_hist"),
        (plot_soh_vs_cycle, "soh_vs_cycle"),
        (plot_soh_vs_cv_tail, "soh_vs_cv_tail"),
        # New:
        (plot_over_events_by_cycle, "over_events_by_cycle"),
        (plot_soc_hist, "soc_hist"),
        (plot_current_vs_time, "current_vs_time"),
    ]:
        p = fn(df, outdir)
        if isinstance(p, dict):
            arts.update(p)
        elif p is not None:
            arts[name] = str(p)

    return arts

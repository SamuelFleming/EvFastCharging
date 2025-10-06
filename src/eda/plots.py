# src/eda/plots.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# ---- Column expectations (dataset-agnostic) ----
# We assume your unified “signals” CSV has these canonical names:
# required base: t, V, I, T_cell, battery_id, cycle_id
# optional derived: SoC, dSoC, C_rate, subset
REQ_BASE = ["t", "V", "I", "T_cell", "battery_id", "cycle_id"]
OPT = ["SoC", "dSoC", "C_rate", "subset"]

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

# ---- Stats (robust) ----
def quick_stats(df: pd.DataFrame) -> Dict[str, float | int | None]:
    stats = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "batteries": int(df["battery_id"].nunique()) if "battery_id" in df else None,
        "cycles": int(df["cycle_id"].nunique()) if "cycle_id" in df else None,
        "subsets": int(df["subset"].nunique()) if "subset" in df else None,
    }
    for col in ["V","I","T_cell","SoC","C_rate"]:
        if col in df.columns and len(df[col]):
            stats[f"{col}_min"] = float(np.nanmin(df[col].values))
            stats[f"{col}_max"] = float(np.nanmax(df[col].values))
    return stats

def write_stats_json(df: pd.DataFrame, outdir: Path) -> Path:
    outdir = _ensure_outdir(outdir)
    p = outdir / "eda_summary.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(quick_stats(df), f, indent=2)
    return p

# ---- Individual plots (each returns the saved file path) ----
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

# ---- Convenience: run all available plots safely ----
def run_all(df: pd.DataFrame, outdir: Path) -> dict:
    outdir = Path(outdir)
    arts = {}
    arts["stats_json"] = str(write_stats_json(df, outdir))
    for fn, name in [
        (plot_voltage_vs_soc, "voltage_vs_soc"),
        (plot_temp_vs_time,   "temp_vs_time"),
        (plot_crate_vs_time,  "crate_vs_time"),
        (plot_cv_tail_by_cycle,"cv_tail_by_cycle"),
        (plot_correlation,    "correlation"),
        (plot_crate_hist,     "crate_hist"),
        # New:
        (plot_soh_vs_cycle,   "soh_vs_cycle"),
        (plot_soh_vs_cv_tail, "soh_vs_cv_tail"),
    ]:
        p = fn(df, outdir)
        if p is not None:
            arts[name] = str(p)
    return arts

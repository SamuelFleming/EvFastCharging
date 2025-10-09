#!/usr/bin/env python
"""
CC–CV baseline simulation using PyBaMM.

MVP goals:
- Sweep CC currents (1C–4C) with CV hold at V_max until I_cut (in C-rate).
- SoC_init in [0.1, 0.3] (configurable).
- Export per-run timeseries + summary (time-to-80% SoC).
- Generate quick plots (V–SoC, I–t, T–t) with optional NASA overlay (from processed CSVs).
- Depend only on data_interface/schema contract (no raw loading).

Usage (from repo root):
    python -m src.baselines.simulate_cccv_pybamm \
        --processed-dir data/processed \
        --output-dir data/processed/baselines/cccv_mvp \
        --soc-inits 0.1 0.3 \
        --cc-c 1 2 3 4 \
        --v-max 4.2 \
        --i-cut-c 0.05 \
        --target-soc 0.8 \
        --thermal lumped \
        --param-set Chen2020 \
        --overlay-nasa

Notes:
- Requires PyBaMM >= 23.x, pandas, numpy, matplotlib.
- If Tbl3_metadata.csv is missing, V_max/C_max defaults are used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project utilities (your contract)
from src.utils.data_interface import load_mvp_tables
from src.utils.schema import ensure_columns

# --- PyBaMM imports (heavy) ---
import pybamm


# ----------------------------
# Data containers / utilities
# ----------------------------
@dataclass
class RunConfig:
    soc_init: float
    c_rate: float
    v_max: float
    i_cut_c: float
    target_soc: float
    thermal: str
    param_set: str
    cell_id: str
    subset: str


@dataclass
class RunSummary:
    soc_init: float
    c_rate: float
    time_to_target_s: Optional[float]
    reached_target: bool
    termination: str


def _safe_get_solution_var(sol: pybamm.Solution, name_candidates: List[str]) -> Optional[np.ndarray]:
    """
    Try to extract a variable from the solution by trying common names.
    Returns None if not found.
    """
    for name in name_candidates:
        try:
            return sol[name](sol.t)
        except KeyError:
            continue
    return None


def _compute_soc_from_current(t_s: np.ndarray, I_A: np.ndarray, soc0: float, cap_Ah: float) -> np.ndarray:
    """Fallback SoC estimate using a tiny cumulative trapezoid (no SciPy)."""
    if cap_Ah is None or cap_Ah <= 0:
        raise ValueError("Nominal capacity [A.h] must be positive to compute SoC.")
    # cumulative trapezoid integral of I(t) over time
    dt = np.diff(t_s)
    mid = 0.5 * (I_A[1:] + I_A[:-1])
    cum = np.empty_like(I_A, dtype=float)
    cum[0] = 0.0
    cum[1:] = np.cumsum(mid * dt)
    dq_Ah = cum / 3600.0  # A*s -> A*h
    soc = soc0 - dq_Ah / cap_Ah
    return np.clip(soc, 0.0, 1.0)



def _time_to_soc(t_s: np.ndarray, soc: np.ndarray, target: float) -> Tuple[Optional[float], bool]:
    """
    Linear interpolation to find the time when SoC first reaches 'target'.
    Returns (time_s, reached_flag).
    """
    if len(t_s) == 0:
        return None, False
    if soc[0] >= target:
        return 0.0, True
    above = np.where(soc >= target)[0]
    if len(above) == 0:
        return None, False
    i = int(above[0])
    if i == 0:
        return t_s[0], True
    # interpolate between i-1 and i
    t0, t1 = t_s[i - 1], t_s[i]
    s0, s1 = soc[i - 1], soc[i]
    if s1 == s0:
        return t1, True
    frac = (target - s0) / (s1 - s0)
    return float(t0 + frac * (t1 - t0)), True


def _infer_nominal_capacity_Ah(param_values: pybamm.ParameterValues) -> Optional[float]:
    for k in [
        "Nominal cell capacity [A.h]",
        "Cell capacity [A.h]",
        "Nominal capacity [A.h]",
    ]:
        try:
            return float(param_values[k])
        except KeyError:
            continue
        except Exception:
            pass
    return None


# ----------------------------
# Core simulation routine
# ----------------------------
def simulate_cccv_single(
    cfg: RunConfig,
) -> Tuple[pd.DataFrame, RunSummary]:
    """
    Run a single CC–CV charge in PyBaMM and return (timeseries_df, summary).
    """

    # --- Model & options ---
    # SPMe is a good speed/physics compromise; enable thermal model if requested.
    options = {"thermal": cfg.thermal}
    model = pybamm.lithium_ion.SPMe(options=options)

    # --- Parameter set ---
    param_values = pybamm.ParameterValues(cfg.param_set)
    # Set ambient temperature for thermal models (lumped/xyz) if desired:
    # param_values.update({"Ambient temperature [K]": 298.15})

    # # Initialise SoC via param (supported by standard Li-ion sets)
    # param_values.set_initial_stoichiometries(cfg.soc_init)

    try:
        param_values.set_initial_stoichiometries(cfg.soc_init)
    except Exception:
        x, y = pybamm.lithium_ion.get_initial_stoichiometries(cfg.soc_init, param_values)
        cn_max = float(param_values["Maximum concentration in negative electrode [mol.m-3]"])
        cp_max = float(param_values["Maximum concentration in positive electrode [mol.m-3]"])
        param_values.update({
            "Initial concentration in negative electrode [mol.m-3]": x * cn_max,
            "Initial concentration in positive electrode [mol.m-3]": y * cp_max,
        })

    # --- Build experiment: CC at C until V_max, then CV until I < i_cut ---
    # Convert decimal C-rate (e.g., 0.05) to "C/<denom>" (e.g., C/20)
    cut_c = float(cfg.i_cut_c)
    if not (0 < cut_c < 1):
        raise ValueError("i_cut_c must be in (0, 1) for 'C/<denominator>' syntax")
    denom = int(round(1.0 / cut_c))  # 0.05 -> 20
    cut_str = f"C/{denom}"

    print("[EXPERIMENT]")
    print(f"  Charge at {cfg.c_rate} C until {cfg.v_max} V")
    print(f"  Hold at {cfg.v_max} V until {cut_str}")

    exp = pybamm.Experiment([
        f"Charge at {cfg.c_rate} C until {cfg.v_max} V",
        f"Hold at {cfg.v_max} V until {cut_str}",   # <-- FIXED: exact token "C/<denom>"
    ])

    # --- Simulation (robust chain for Windows) ---
    solve_ok = False
    last_err = None

    # Try CasADi (DAE-capable)
    try:
        solver = pybamm.CasadiSolver(mode="safe")
        sim = pybamm.Simulation(model=model, parameter_values=param_values, experiment=exp, solver=solver)
        sol = sim.solve()
        solve_ok = True
    except Exception as e:
        print("[WARN] CasADi (default) failed:", repr(e))
        last_err = e

    # Try IDAKLU (DAE-capable) if available
    if not solve_ok:
        try:
            from pybamm import IDAKLUSolver
            solver = IDAKLUSolver()
            sim = pybamm.Simulation(model=model, parameter_values=param_values, experiment=exp, solver=solver)
            sol = sim.solve()
            solve_ok = True
        except Exception as e:
            print("[WARN] IDAKLUSolver failed or unavailable:", repr(e))
            last_err = e

    # FINAL FALLBACK: run CC-only (ODE) so SciPy can solve and we can still get t_to_80%
    if not solve_ok:
        print("[WARN] No DAE-capable solver available. Falling back to SPM + SciPy(BDF) with CC-only step.")
        # Rebuild ODE model and ODE-friendly experiment
        model = pybamm.lithium_ion.SPM(options={"thermal": "isothermal"})
        # Long enough CC to reach 80% from 10% at ~1C (2 hours = 7200 s). Adjust if needed.
        exp_cc_only = pybamm.Experiment([f"Charge at {cfg.c_rate} C for 7200 seconds"])
        try:
            solver = pybamm.ScipySolver(method="BDF", rtol=1e-6, atol=1e-8)
            sim = pybamm.Simulation(model=model, parameter_values=param_values, experiment=exp_cc_only, solver=solver)
            sol = sim.solve()
            solve_ok = True
        except Exception as e:
            last_err = e

    if not solve_ok:
        raise RuntimeError("All solver options failed. Install a DAE solver (CasADi/IDAS or scikits.odes).") from last_err

    # --- Extract signals ---
    t_s = sol["Time [s]"](sol.t)
    V = sol["Terminal voltage [V]"](sol.t)
    I = sol["Current [A]"](sol.t)

    # Temperature (K or C)
    T_candidates_K = [
        "Volume-averaged cell temperature [K]",
        "X-averaged cell temperature [K]",
        "Cell temperature [K]",
    ]
    T_candidates_C = [
        "Volume-averaged cell temperature [C]",
        "X-averaged cell temperature [C]",
        "Cell temperature [C]",
    ]
    T = _safe_get_solution_var(sol, T_candidates_C)
    if T is None:
        T_K = _safe_get_solution_var(sol, T_candidates_K)
        if T_K is not None:
            T = T_K - 273.15
    if T is None:
        # Isothermal model or variable missing: fallback to constant 25°C
        T = np.full_like(V, 25.0)

    # SoC from model if available; else coulomb count
    soc = _safe_get_solution_var(sol, [
        "State of Charge",
        "Cell state of charge",
        "X-averaged cell state of charge",
        "x-averaged cell state of charge",
    ])
    if soc is None:
        cap_Ah = _infer_nominal_capacity_Ah(param_values)
        if cap_Ah is None:
            cap_Ah = 2.9  # pragmatic fallback for MVP
        soc = _compute_soc_from_current(t_s, I, cfg.soc_init, cap_Ah)

    # Time-to-target metric
    t80_s, reached = _time_to_soc(t_s, soc, cfg.target_soc)

    # Compose dataframe
    df = pd.DataFrame(
        {
            "t_s": t_s,
            "V": V,
            "I_A": I,
            "T_C": T,
            "SoC": soc,
            "c_rate": cfg.c_rate,
            "soc_init": cfg.soc_init,
            "v_max": cfg.v_max,
            "i_cut_c": cfg.i_cut_c,
            "target_soc": cfg.target_soc,
            "cell_id": cfg.cell_id,
            "subset": cfg.subset,
        }
    )

    # Termination text from experiment summary, if available
    term = "completed"
    try:
        term = sim.summary["termination"].splitlines()[-1]
    except Exception:
        pass

    summary = RunSummary(
        soc_init=cfg.soc_init,
        c_rate=cfg.c_rate,
        time_to_target_s=t80_s,
        reached_target=reached,
        termination=term,
    )
    return df, summary


# ----------------------------
# Plotting
# ----------------------------
def _plot_v_soc(
    dfs: List[pd.DataFrame],
    outdir: Path,
    overlay_nasa_df: Optional[pd.DataFrame] = None,
) -> None:
    plt.figure()
    for c_rate, grp in df_groupby_c(dfs):
        plt.plot(grp["SoC"], grp["V"], label=f"{c_rate:.2g}C")
    if overlay_nasa_df is not None:
        # Bin by SoC and overlay median NASA V(SoC)
        bins = np.linspace(0, 1.0, 51)
        cats = pd.cut(overlay_nasa_df["SoC"], bins, include_lowest=True)
        med = overlay_nasa_df.groupby(cats)["V"].median().reset_index()
        # x at bin centers
        x = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(x[: len(med)], med["V"].to_numpy(), linewidth=1.5, linestyle="--", label="NASA median")
    plt.xlabel("SoC")
    plt.ylabel("Voltage [V]")
    plt.title("CC–CV Baseline: V vs SoC")
    plt.legend()
    out = outdir / "cccv_v_vs_soc.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_i_t(dfs: List[pd.DataFrame], outdir: Path) -> None:
    plt.figure()
    for c_rate, grp in df_groupby_c(dfs):
        plt.plot(grp["t_s"], grp["I_A"], label=f"{c_rate:.2g}C")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.title("CC–CV Baseline: Current vs Time")
    plt.legend()
    out = outdir / "cccv_current_vs_time.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_t_t(dfs: List[pd.DataFrame], outdir: Path) -> None:
    plt.figure()
    for c_rate, grp in df_groupby_c(dfs):
        plt.plot(grp["t_s"], grp["T_C"], label=f"{c_rate:.2g}C")
    plt.xlabel("Time [s]")
    plt.ylabel("Cell Temperature [°C]")
    plt.title("CC–CV Baseline: Temperature vs Time")
    plt.legend()
    out = outdir / "cccv_temperature_vs_time.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def df_groupby_c(dfs: List[pd.DataFrame]) -> List[Tuple[float, pd.DataFrame]]:
    """Merge and yield by c_rate."""
    if len(dfs) == 1:
        return [(dfs[0]["c_rate"].iloc[0], dfs[0])]
    big = pd.concat(dfs, ignore_index=True)
    out = []
    for c, g in big.groupby("c_rate", sort=True):
        # average over runs if multiple soc_init for same c_rate (plot smooth curve)
        g_sorted = g.sort_values("SoC").drop_duplicates(subset=["SoC"], keep="last")
        out.append((c, g_sorted))
    return out


# ----------------------------
# NASA overlay (optional)
# ----------------------------
def build_nasa_overlay(processed_dir: Path, soc_min: float, soc_max: float) -> Optional[pd.DataFrame]:
    """
    Construct a light overlay df from Tbl1_signals: charge phase points
    constrained to SoC in [soc_min, soc_max], columns 'SoC', 'V'.
    """
    try:
        signals_df, episodes_df, meta = load_mvp_tables(processed_dir, with_soh="none")
    except Exception:
        return None

    # Only charge phase
    df = signals_df.query("phase == 'charge'").copy()
    ensure_columns(df, ["SoC", "V"], "Tbl1")
    df = df[(df["SoC"] >= soc_min) & (df["SoC"] <= soc_max)]
    # Keep only reasonable voltages
    df = df[(df["V"] > 0.0) & (df["V"] < 5.5)]
    return df[["SoC", "V"]].copy()


# ----------------------------
# Main CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Simulate CC–CV baseline in PyBaMM")
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed/baselines/cccv_mvp"))
    ap.add_argument("--soc-inits", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--cc-c", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--v-max", type=float, default=None, help="Overrides Tbl3 V_max if provided")
    ap.add_argument("--i-cut-c", type=float, default=0.05)
    ap.add_argument("--target-soc", type=float, default=0.5)
    ap.add_argument("--thermal", choices=["isothermal", "lumped", "x-lumped"], default="lumped")
    ap.add_argument("--param-set", type=str, default="Chen2020", help="e.g., Chen2020, Ecker2015, Marquis2019")
    ap.add_argument("--overlay-nasa", action="store_true", help="Overlay NASA V–SoC median curve")
    args = ap.parse_args()

    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata via contract (for defaults)
    _, _, meta = load_mvp_tables(args.processed_dir, with_soh="none")
    v_max = args.v_max if args.v_max is not None else float(meta.get("V_max", 4.2))
    # We don’t enforce T_max in baseline; we just simulate and record T for eval.

    # Optional NASA overlay
    overlay = build_nasa_overlay(args.processed_dir, soc_min=min(args.soc_inits), soc_max=args.target_soc) if args.overlay_nasa else None

    # Run sweeps
    all_runs: List[pd.DataFrame] = []
    summaries: List[RunSummary] = []

    for s0 in args.soc_inits:
        for c in args.cc_c:
            cfg = RunConfig(
                soc_init=s0,
                c_rate=c,
                v_max=v_max,
                i_cut_c=args.i_cut_c,
                target_soc=args.target_soc,
                thermal=args.thermal,
                param_set=args.param_set,
                cell_id=str(meta.get("cell_id", "unknown")),
                subset=str(meta.get("subset", "unknown")),
            )
            print(f"[CCCV] sim: SoC0={cfg.soc_init:.2f}, CC={cfg.c_rate:.2g}C, Vmax={cfg.v_max:.2f}V, Icut={cfg.i_cut_c:.3g}C, thermal={cfg.thermal}, set={cfg.param_set}")
            df, summ = simulate_cccv_single(cfg)
            all_runs.append(df)
            summaries.append(summ)

    # Export timeseries
    ts_path = outdir / "baseline_cccv_results.csv"
    pd.concat(all_runs, ignore_index=True).to_csv(ts_path, index=False)
    print(f"Saved timeseries -> {ts_path}")

    # Export summary
    summary_df = pd.DataFrame([asdict(s) for s in summaries])
    sum_path = outdir / "baseline_cccv_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    print(f"Saved summary -> {sum_path}")

    # Plots
    _plot_v_soc(all_runs, outdir, overlay_nasa_df=overlay)
    _plot_i_t(all_runs, outdir)
    _plot_t_t(all_runs, outdir)
    print(f"Saved figures -> {outdir}")


if __name__ == "__main__":
    main()

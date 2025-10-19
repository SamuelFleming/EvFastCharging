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

Enhancements (as of 18th October)
- Read defaults (soc_target, V_max, dt) from live_dataset.json unless CLI overrides.
- Organise outputs per-run under data/processed/baselines/<run_name>/ (RL-style).
- Write run-level summary.json and live_dataset_snapshot.json.
- Append registries at data/processed/baselines/: baselines_registry.csv and baselines_registry_detailed.csv.

Back-compat:
- If --run-name is not provided, we use --output-dir as-is (legacy flat folder).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import json

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
    """Try to extract a variable from the solution by trying common names."""
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
def simulate_cccv_single(cfg: RunConfig) -> Tuple[pd.DataFrame, RunSummary, str, bool]:
    """Run a single CC–CV charge in PyBaMM and return (timeseries_df, summary)."""
    # --- Model & options ---
    options = {"thermal": cfg.thermal}
    model = pybamm.lithium_ion.SPMe(options=options)

    # --- Parameter set ---
    param_values = pybamm.ParameterValues(cfg.param_set)

    # Initialise SoC robustly
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

    # --- Experiment: CC at C until V_max, then CV until I < i_cut ---
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
        f"Hold at {cfg.v_max} V until {cut_str}",
    ])

    solver_backend = "unkown"
    cc_only = False
    # --- Solver chain ---
    solve_ok = False
    last_err = None

    # Try CasADi (DAE-capable)
    try:
        solver = pybamm.CasadiSolver(mode="safe")
        sim = pybamm.Simulation(model=model, parameter_values=param_values, experiment=exp, solver=solver)
        sol = sim.solve()
        solver_backend = "casadi"
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
            solver_backend = "idaklu"
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
            solver_backend = "spm_scipy_fallback"
            cc_only = True
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
        cap_Ah = _infer_nominal_capacity_Ah(param_values) or 2.9
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
    return df, summary, solver_backend, cc_only


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
    """Construct a light overlay df from Tbl1_signals: charge-phase points in SoC range."""
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
# Registry helpers
# ----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_csv_row(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


# ----------------------------
# Main CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Simulate CC–CV baseline in PyBaMM")
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    # New: output-root + run-name (preferred), but keep output-dir for back-compat
    ap.add_argument("--output-root", type=Path, default=Path("data/processed/baselines"))
    ap.add_argument("--run-name", type=str, default=None, help="Run folder name under --output-root (e.g., cccv_smoke). If omitted, legacy --output-dir is used as-is.")
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed/baselines/cccv_mvp"),
                    help="(Deprecated) Legacy folder to write into directly when --run-name is not provided.")
    # Live dataset meta
    ap.add_argument("--dataset-meta", type=Path, default=Path("data/processed/metadata/live_dataset.json"),
                    help="Path to live_dataset.json for defaults (soc_target, limits).")
    ap.add_argument("--soc-inits", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--cc-c", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--v-max", type=float, default=None, help="Overrides V_max from live_dataset/Tbl3 if provided")
    ap.add_argument("--i-cut-c", type=float, default=0.05)
    ap.add_argument("--target-soc", type=float, default=None, help="If omitted, defaults to live_dataset.json soc_target (else 0.8)")
    ap.add_argument("--thermal", choices=["isothermal", "lumped", "x-lumped"], default="lumped")
    ap.add_argument("--param-set", type=str, default="Chen2020", help="e.g., Chen2020, Ecker2015, Marquis2019")
    ap.add_argument("--overlay-nasa", action="store_true", help="Overlay NASA V–SoC median curve")
    args = ap.parse_args()

    # Determine outdir (RL-style run folder if run-name provided)
    if args.run_name:
        outdir = args.output_root / args.run_name
    else:
        outdir = args.output_dir  # legacy behavior
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata via contract (tables + minimal meta)
    _, _, meta_tbl = load_mvp_tables(args.processed_dir, with_soh="none")

    # Read live dataset JSON (defaults): dataset_id, run_id, limits, soc_target
    live_meta = {}
    if args.dataset_meta.exists():
        try:
            live_meta = json.loads(args.dataset_meta.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Could not parse live dataset meta at {args.dataset_meta}: {e}")

    dataset_id = live_meta.get("dataset_id", "unknown")
    dataset_run_id = live_meta.get("run_id", "unknown")
    extraction_cfg = live_meta.get("extraction_config", {})
    limits = extraction_cfg.get("limits", {})

    # Defaults from live meta if CLI omitted
    target_soc = args.target_soc if args.target_soc is not None else float(extraction_cfg.get("soc_target", 0.8))
    v_max_default = float(limits.get("V_max", 4.2))
    v_max = args.v_max if args.v_max is not None else float(meta_tbl.get("V_max", v_max_default))

    # Optional NASA overlay
    overlay = build_nasa_overlay(args.processed_dir, soc_min=min(args.soc_inits), soc_max=target_soc) if args.overlay_nasa else None

    # Run sweeps
    all_runs: List[pd.DataFrame] = []
    summaries: List[RunSummary] = []
    backends: set[str] = set()
    any_cc_only = False

    for s0 in args.soc_inits:
        for c in args.cc_c:
            cfg = RunConfig(
                soc_init=s0,
                c_rate=c,
                v_max=v_max,
                i_cut_c=args.i_cut_c,
                target_soc=target_soc,
                thermal=args.thermal,
                param_set=args.param_set,
                cell_id=str(meta_tbl.get("cell_id", "unknown")),
                subset=str(meta_tbl.get("subset", "unknown")),
            )
            print(f"[CCCV] sim: SoC0={cfg.soc_init:.2f}, CC={cfg.c_rate:.2g}C, Vmax={cfg.v_max:.2f}V, Icut={cfg.i_cut_c:.3g}C, thermal={cfg.thermal}, set={cfg.param_set}")
            df, summ, backend, cc_only = simulate_cccv_single(cfg)
            all_runs.append(df)
            summaries.append(summ)
            backends.add(backend)
            any_cc_only = any_cc_only or cc_only

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

    # Write run summary.json & snapshot the live dataset meta
    run_name = args.run_name or outdir.name
    created_at = _utc_now_iso()
    snapshot_name = "live_dataset_snapshot.json"
    try:
        if live_meta:
            (outdir / snapshot_name).write_text(json.dumps(live_meta, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Could not write {snapshot_name}: {e}")

    run_summary_json = {
        "run_name": run_name,
        "algo": "CCCV",
        "created_at_utc": created_at,
        "dataset_id": dataset_id,
        "dataset_run_id": dataset_run_id,
        "param_set": args.param_set,
        "thermal": args.thermal,
        "target_soc": float(target_soc),
        "v_max_used": float(v_max),
        "i_cut_c": float(args.i_cut_c),
        "soc_inits": list(map(float, args.soc_inits)),
        "cc_c": list(map(float, args.cc_c)),
        "solver_backends": sorted(backends),    # e.g., ["spm_scipy_fallback"]
        "cc_only_fallback": bool(any_cc_only),  # True if any sweep member used fallback
        "live_dataset_snapshot": snapshot_name if live_meta else None,
        "results_csv": "baseline_cccv_results.csv",
        "summary_csv": "baseline_cccv_summary.csv",
        "figures": [
            "cccv_v_vs_soc.png",
            "cccv_current_vs_time.png",
            "cccv_temperature_vs_time.png",
        ],
    }
    try:
        (outdir / "summary.json").write_text(json.dumps(run_summary_json, indent=2), encoding="utf-8")
        print(f"Wrote run summary -> {outdir / 'summary.json'}")
    except Exception as e:
        print(f"[WARN] Could not write summary.json: {e}")

    # Append registries
    registry_root = args.output_root  # registry files live under output_root
    run_registry_csv = registry_root / "baselines_registry.csv"
    detailed_registry_csv = registry_root / "baselines_registry_detailed.csv"

    # Run-level registry row
    run_row = {
        "timestamp": created_at,
        "run_name": run_name,
        "algo": "CCCV",
        "dataset_id": dataset_id,
        "dataset_run_id": dataset_run_id,
        "param_set": args.param_set,
        "thermal": args.thermal,
        "target_soc": float(target_soc),
        "v_max_used": float(v_max),
        "i_cut_c": float(args.i_cut_c),
        "soc_inits": json.dumps(list(map(float, args.soc_inits))),
        "cc_c": json.dumps(list(map(float, args.cc_c))),
        "summary_path": str(Path(run_name) / "baseline_cccv_summary.csv") if args.run_name else str(outdir / "baseline_cccv_summary.csv"),
        "out_dir": str(Path(run_name)) if args.run_name else str(outdir),
        "notes": "",
    }
    try:
        _append_csv_row(run_registry_csv, run_row)
        print(f"Updated run registry -> {run_registry_csv}")
    except Exception as e:
        print(f"[WARN] Could not append {run_registry_csv}: {e}")

    # Detailed registry rows (one per sweep)
    try:
        det_rows = []
        for r in summaries:
            det_rows.append({
                "timestamp": created_at,
                "run_name": run_name,
                "algo": "CCCV",
                "dataset_id": dataset_id,
                "soc_init": float(r.soc_init),
                "c_rate": float(r.c_rate),
                "time_to_target_s": (float(r.time_to_target_s) if r.time_to_target_s is not None else None),
                "reached_target": bool(r.reached_target),
                "termination": str(r.termination),
                "out_dir": str(Path(run_name)) if args.run_name else str(outdir),
            })
        header = not detailed_registry_csv.exists()
        pd.DataFrame(det_rows).to_csv(detailed_registry_csv, mode="a", header=header, index=False)
        print(f"Updated detailed registry -> {detailed_registry_csv}")
    except Exception as e:
        print(f"[WARN] Could not append {detailed_registry_csv}: {e}")


if __name__ == "__main__":
    main()


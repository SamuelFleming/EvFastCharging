# src/eda/run_eda.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from .plots import (
    run_all,
    plot_voltage_vs_soc, plot_temp_vs_time, plot_crate_vs_time,
    plot_cv_tail_by_cycle, plot_correlation, plot_crate_hist,
    write_stats_json,
    plot_soh_vs_cycle, plot_soh_vs_cv_tail
)

PLOT_FUNCS = {
    "all": run_all,
    "voltage_vs_soc": plot_voltage_vs_soc,
    "temp_vs_time": plot_temp_vs_time,
    "crate_vs_time": plot_crate_vs_time,
    "cv_tail_by_cycle": plot_cv_tail_by_cycle,
    "correlation": plot_correlation,
    "crate_hist": plot_crate_hist,
    # new targeted plots:
    "soh_vs_cycle": plot_soh_vs_cycle,
    "soh_vs_cv_tail": plot_soh_vs_cv_tail,
}

def main():
    ap = argparse.ArgumentParser(description="General EDA runner for unified signals CSV (optionally merge SoH).")
    ap.add_argument("--csv", required=True, help="Path to processed signals CSV (Tbl1 or merged signals).")
    ap.add_argument("--outdir", default="data/processed/figures", help="Where to write figures.")
    ap.add_argument("--which", default="all", choices=list(PLOT_FUNCS.keys()),
                    help="Which plot to run (default: all).")
    ap.add_argument("--vthresh", type=float, default=4.15,
                    help="Voltage threshold for CV-tail proxy (default: 4.15 V)")
    ap.add_argument("--soh", default=None,
                    help="Optional SoH CSV (battery_id,cycle_id,SoH). If provided, will merge into the dataframe in-memory.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print(f"[EDA] reading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Optional SoH merge
    if args.soh:
        soh_path = Path(args.soh)
        if not soh_path.exists():
            raise FileNotFoundError(soh_path)
        print(f"[EDA] merging SoH from: {soh_path}")
        df_soh = pd.read_csv(soh_path)
        if not {"battery_id","cycle_id","SoH"}.issubset(df_soh.columns):
            raise ValueError("SoH CSV must contain battery_id, cycle_id, SoH")
        if "battery_id" not in df.columns and "cell_id" in df.columns:
            df = df.copy()
            df["battery_id"] = df["cell_id"]
            print("[EDA] Added 'battery_id' from 'cell_id' in signals df")
        if not {"battery_id","cycle_id"}.issubset(df.columns):
            raise ValueError("Signals CSV must contain battery_id/cell_id and cycle_id for SoH merge")
        df = df.merge(df_soh[["battery_id","cycle_id","SoH"]],
                      on=["battery_id","cycle_id"], how="left")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[EDA] outdir:  {outdir}")

    # Dispatch
    if args.which == "all":
        print("[EDA] running: all plots")
        arts = run_all(df, outdir)
        for k, v in arts.items():
            print(f"[EDA] wrote: {k} -> {v}")
    elif args.which == "cv_tail_by_cycle":
        print(f"[EDA] running: cv_tail_by_cycle (V >= {args.vthresh} V)")
        p = plot_cv_tail_by_cycle(df, outdir, v_thresh=args.vthresh)
        print(f"[EDA] wrote: {p}")
    elif args.which == "soh_vs_cv_tail":
        print(f"[EDA] running: soh_vs_cv_tail (V ≥ {args.vthresh} V)")
        p = plot_soh_vs_cv_tail(df, outdir, v_thresh=args.vthresh)
        print(f"[EDA] wrote: {p}")
    else:
        print(f"[EDA] running: {args.which}")
        p = PLOT_FUNCS[args.which](df, outdir)
        print(f"[EDA] wrote: {p}")

    stats_path = write_stats_json(df, outdir)
    print("[EDA] summary stats:", stats_path)

if __name__ == "__main__":
    main()


# src/eda/run_eda.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from .plots import (
    run_all,
    plot_voltage_vs_soc, plot_temp_vs_time, plot_crate_vs_time,
    plot_cv_tail_by_cycle, plot_correlation, plot_crate_hist,
    write_stats_json
)

PLOT_FUNCS = {
    "all": run_all,
    "voltage_vs_soc": plot_voltage_vs_soc,
    "temp_vs_time": plot_temp_vs_time,
    "crate_vs_time": plot_crate_vs_time,
    "cv_tail_by_cycle": plot_cv_tail_by_cycle,
    "correlation": plot_correlation,
    "crate_hist": plot_crate_hist,
}

def main():
    ap = argparse.ArgumentParser(description="General EDA runner for unified signals CSV.")
    ap.add_argument("--csv", required=True, help="Path to processed signals CSV (e.g., data/processed/clean_nasa_charge_*.csv)")
    ap.add_argument("--outdir", default="data/processed/figures", help="Where to write figures (default: %(default)s)")
    ap.add_argument("--which", default="all", choices=list(PLOT_FUNCS.keys()),
                    help="Which plot to run (default: all).")
    ap.add_argument("--vthresh", type=float, default=4.15,
                    help="Voltage threshold for CV-tail proxy (default: 4.15 V)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print(f"[EDA] reading: {csv_path}")
    df = pd.read_csv(csv_path)
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
    else:
        print(f"[EDA] running: {args.which}")
        p = PLOT_FUNCS[args.which](df, outdir)
        print(f"[EDA] wrote: {p}")

    stats_path = write_stats_json(df, outdir)
    print("[EDA] summary stats:", stats_path)

if __name__ == "__main__":
    main()


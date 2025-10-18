# src/nasa_data_extract/join_soh.py
from __future__ import annotations
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np

from .extract_discharge_cycles import build_discharge_capacity_table
from .read import get_root, list_mat_files
from .merge_soh import merge_soh_into_tbls

def main():
    ap = argparse.ArgumentParser(description="Build SoH from discharge cycles and (optionally) merge into Tbl1/Tbl2.")
    ap.add_argument("--live-json", default="data/processed/metadata/live_dataset.json",
                    help="Live dataset metadata JSON describing the exact raw files/subsets to use "
                         "(default: %(default)s). If raw_files present, it takes precedence over subsets/--limit.")
    ap.add_argument("--out-soh", default="data/processed/TblSoH_discharge_capacity.csv",
                    help="Where to write the SoH table (default: %(default)s)")
    ap.add_argument("--tbl1", default="data/processed/Tbl1_signals.csv",
                    help="Path to Tbl1 (signals) to merge SoH into (default: %(default)s)")
    ap.add_argument("--tbl2", default="data/processed/Tbl2_episodes.csv",
                    help="Path to Tbl2 (episodes) to merge SoH into (default: %(default)s)")
    ap.add_argument("--out-tbl1", default=None,
                    help="Output path for Tbl1+SoH (default: alongside tbl1 as Tbl1_signals_with_SoH.csv)")
    ap.add_argument("--out-tbl2", default=None,
                    help="Output path for Tbl2+SoH (default: alongside tbl2 as Tbl2_episodes_with_SoH.csv)")
    ap.add_argument("--limit", type=int, default=5,
                    help="Limit number of .mat files parsed for SoH when subsets fallback is used (default: %(default)s)")
    args = ap.parse_args()

    print("[join_soh] Starting SoH construction pipeline")
    root = get_root()
    print(f"[join_soh] Using RAW ROOT (read.py): {root}")

    # -------- Resolve discharge inputs from live_dataset.json --------
    resolved_files: list[Path] | None = None
    live_json_path = Path(args.live_json)
    if live_json_path.exists():
        try:
            meta = json.loads(live_json_path.read_text(encoding="utf-8"))
            prov = meta.get("provenance", {})
            raw_root = Path(prov.get("raw_root", str(root)))
            raw_files = prov.get("raw_files", []) or []
            subsets = prov.get("subsets", []) or []

            if raw_files:
                resolved_files = [(raw_root / rf).resolve() for rf in raw_files]
                print(f"[join_soh] live_json raw_files found: {len(resolved_files)} file(s). Ignoring --limit and subsets.")
            elif subsets:
                resolved_files = list_mat_files(subsets=subsets, limit=args.limit)
                print(f"[join_soh] live_json subsets found: {subsets} → selected {len(resolved_files)} file(s) with limit={args.limit}.")
            else:
                print("[join_soh] live_json has neither raw_files nor subsets; falling back to legacy discovery.")
        except Exception as e:
            print(f"[join_soh] WARNING: Failed to read/parse live JSON at {live_json_path}: {e}. Falling back to legacy discovery.")
    else:
        print(f"[join_soh] live_json not found at {live_json_path}. Falling back to legacy discovery.")

    # Step 1: discharge table
    df_cap = build_discharge_capacity_table(files=resolved_files)  # if None, builder will legacy-discover (limit=5)
    if df_cap.empty:
        print("[join_soh] No discharge capacity data extracted — exiting early.")
        return

    print(f"[join_soh] Discharge capacity table → {len(df_cap):,} rows")
    print(df_cap.head())
    # Minimal visibility into the chosen batteries/cycles
    try:
        bset = sorted(df_cap['battery_id'].unique().tolist())
        print(f"[join_soh] Discharge batteries discovered: {bset[:10]}{' …' if len(bset)>10 else ''} (total={len(bset)})")
    except Exception:
        pass

    # Step 2: SoH
    df_cap = (
        df_cap.sort_values(["battery_id", "cycle_id"])
              .groupby("battery_id")
              .apply(lambda g: g.assign(
                  capacity_ref_Ah=g["capacity_discharge_Ah"].iloc[0],
                  SoH=g["capacity_discharge_Ah"] / g["capacity_discharge_Ah"].iloc[0]
              ))
              .reset_index(drop=True)
    )
    print("[join_soh] SoH computed successfully.")
    print(df_cap.head())

    # Step 3: write SoH
    out_path = Path(args.out_soh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_cap.to_csv(out_path, index=False)
    print(f"[join_soh] Wrote SoH table to: {out_path.resolve()}")

    # Step 4: (optional) merge into Tbl1/Tbl2 if they exist or paths provided
    tbl1 = Path(args.tbl1)
    tbl2 = Path(args.tbl2)
    if tbl1.exists() and tbl2.exists():
        print(f"[join_soh] Merging SoH into tables:\n  Tbl1: {tbl1}\n  Tbl2: {tbl2}")
        merge_soh_into_tbls(
            soh_csv=out_path,
            tbl1_csv=tbl1,
            tbl2_csv=tbl2,
            out_tbl1_csv=Path(args.out_tbl1) if args.out_tbl1 else None,
            out_tbl2_csv=Path(args.out_tbl2) if args.out_tbl2 else None,
        )
    else:
        print("[join_soh] Skipping table merge: Tbl1/Tbl2 not found at the provided defaults/paths.")

    print("[join_soh] Done ✅")

if __name__ == "__main__":
    main()




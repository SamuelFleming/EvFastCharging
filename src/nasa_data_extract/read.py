# src/nasa_data_extract/read.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Union
import numpy as np

RAW_ROOT = Path(r"C:\Users\User\OneDrive\Desktop\Tech Projects\EvFastChargingRL\data\raw\NASABatteryAging")



##RAW_ROOT = Path(r"data\raw\NASABatteryAging")

mats = sorted(RAW_ROOT.glob("**/*.mat"))
print(f'RAW ROOT: {RAW_ROOT}')
#print(mats)
print(f"found {len(mats)} .mat files")
for p in mats[:12]:
    print("•", p.relative_to(RAW_ROOT))

# Folder name -> metadata drawn from your README sets
SUBSET_META = {
    "1. BatteryAgingARC-FY08Q4": {
        "ambient_C": "room",
        "discharge_style": "CC 2A (mostly)",
        "notes": "EOL at ~1.4Ah (30% fade)."
    },  # :contentReference[oaicite:0]{index=0}

    "2. BatteryAgingARC_25_26_27_28_P1": {
        "ambient_C": 24,
        "discharge_style": "0.05Hz square wave, 4A, 50% duty",
        "cutoff_V": {25: 2.0, 26: 2.2, 27: 2.5, 28: 2.7}
    },  # :contentReference[oaicite:1]{index=1}

    "3. BatteryAgingARC_25-44": {
        "ambient_C": 43,
        "discharge_style": "CC 4A",
        "cutoff_V": {29: 2.0, 30: 2.2, 31: 2.5, 32: 2.7}
    },  # :contentReference[oaicite:2]{index=2}

    "4. BatteryAgingARC_45_46_47_48": {
        "ambient_C": 4,
        "discharge_style": "CC 4A and/or 1A (mixed)",
        "cutoff_V": {41: 2.0, 42: 2.2, 43: 2.5, 44: 2.7},
        "notes": "Several very low capacity runs noted."
    },  # :contentReference[oaicite:3]{index=3}

    "5. BatteryAgingARC_49_50_51_52": {
        "ambient_C": 4,
        "discharge_style": "CC 2A",
        "cutoff_V": {49: 2.0, 50: 2.2, 51: 2.5, 52: 2.7},
        "notes": "Experiment control software crash; some very low capacity/voltage runs."
    },  # :contentReference[oaicite:4]{index=4}

    "6. BatteryAgingARC_53_54_55_56": {
        "ambient_C": 4,
        "discharge_style": "CC 2A",
        "cutoff_V": {53: 2.0, 54: 2.2, 55: 2.5, 56: 2.7}
    },  # :contentReference[oaicite:5]{index=5}
}
# All groups share the same charge protocol: CC 1.5A → CV 4.2 V, terminate at ~20 mA. 

try:
    import scipy.io as sio
except Exception:
    sio = None

try:
    import mat73
except Exception:
    mat73 = None


def get_subsetMeta() -> dict:
    return SUBSET_META


def get_root() -> Path:
    return RAW_ROOT

def list_mat_files(limit: int | None = None) -> list[Path]:
    mats = sorted(RAW_ROOT.glob("**/*.mat"))
    return mats if limit is None else mats[:limit]

def load_mat_safely(path):
    if sio is not None:
        try:
            return sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            pass
        except Exception:
            pass
    if mat73 is not None:
        return mat73.loadmat(str(path))
    raise RuntimeError("Install scipy and/or mat73 to read MATLAB files.")

def iter_cycles(battery_struct):
    for key in ("cycle", "cycles", "Cycle", "Cycles"):
        if hasattr(battery_struct, key):
            cyc = getattr(battery_struct, key)
            return (cyc if isinstance(cyc, (list, np.ndarray)) else [cyc])
        if isinstance(battery_struct, dict) and key in battery_struct:
            cyc = battery_struct[key]
            return (cyc if isinstance(cyc, (list, np.ndarray)) else [cyc])
    return []

def get_field(x, *names):
    for nm in names:
        if hasattr(x, nm): return getattr(x, nm)
        if isinstance(x, dict) and nm in x: return x[nm]
    return None

def to_1d(x):
    """Return a 1-D numpy array (at least), or None."""
    if x is None:
        return None
    arr = np.asarray(x)
    # if scalar / 0-D, make it length-1
    if arr.ndim == 0:
        arr = arr.reshape(1)
    # squeeze higher dims down to 1-D
    arr = np.squeeze(arr)
    # ensure still at least 1-D
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr

def list_mat_files(
    subset: Optional[Union[str, Iterable[str]]] = None,
    limit: Optional[int] = None,
    exact: bool = True,
) -> list[Path]:
    """List .mat files, optionally restricted to one/many subset folder names."""
    mats = sorted(RAW_ROOT.glob("**/*.mat"))
    if subset is None:
        return mats if limit is None else mats[:limit]

    if isinstance(subset, str):
        wanted = {subset}
    else:
        wanted = set(subset)

    def keep(p: Path) -> bool:
        top = p.relative_to(RAW_ROOT).parts[0]
        if exact:
            return top in wanted
        # substring match if exact=False
        return any(k.lower() in top.lower() for k in wanted)

    filtered = [p for p in mats if keep(p)]
    return filtered if limit is None else filtered[:limit]

if __name__ == "__main__":
    # Quick self-check (optional)
    mats = list_mat_files(limit=5)
    print(f"[read.py] Found {len(mats)} .mat files under:", RAW_ROOT)
    for p in mats:
        print(" •", p.relative_to(RAW_ROOT))



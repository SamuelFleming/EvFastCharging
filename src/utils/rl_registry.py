# src/utils/registry.py
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Optional, Dict, List, Any

REGISTRY_FIELDS: List[str] = [
    "timestamp","run_name","algo","reward_variant","seed",
    "dataset_id","dataset_run_id",
    "dt_s","target_soc","action_low","action_high",
    "v_max_used","t_max_used","soh","k_drop_perc",
    "episodes","steps_per_ep","lr","batch_size","buffer_size","gamma","tau","noise_sigma","net_arch",
    "summary_path","out_dir","notes"
]

class RunRegistry:
    """
    CSV-backed registry for RL runs. Mirrors the baseline style.
    """
    def __init__(self, registry_csv: Path):
        self.registry_csv = Path(registry_csv)

    def append(self, row: Dict[str, Any]):
        self.registry_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.registry_csv.exists()
        with self.registry_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in REGISTRY_FIELDS})

def load_dataset_meta(dataset_meta_path: Optional[Path], processed_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load dataset metadata. Resolution order:
      1) dataset_meta_path (if provided)
      2) <processed_dir>/metadata/live_dataset.json (if exists)
      3) return {}
    """
    # explicit path wins
    if dataset_meta_path:
        p = Path(dataset_meta_path)
        if p.exists():
            return _read_json(p)

    # fallback under processed_dir
    if processed_dir:
        cand = Path(processed_dir) / "metadata" / "live_dataset.json"
        if cand.exists():
            return _read_json(cand)

    return {}

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}

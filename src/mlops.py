from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
MANIFEST_PATH = EXPERIMENT_DIR / "metadata.json"
RESULTS_PATH = EXPERIMENT_DIR / "results.csv"


def _read_manifest() -> Dict[str, object]:
    if not MANIFEST_PATH.exists():
        return {"project": "Port-Optimus", "runs": []}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def next_run_id(prefix: str = "ALPHA") -> str:
    year = datetime.now().year
    manifest = _read_manifest()
    matching = [
        run
        for run in manifest.get("runs", [])
        if str(run.get("run_id", "")).startswith(f"{prefix}-{year}-")
    ]
    return f"{prefix}-{year}-{len(matching) + 1:03d}"


def create_manifest_run(
    params: Dict[str, object],
    metrics: Optional[Dict[str, float]] = None,
    model_path: Optional[str] = None,
    stress_test: str = "Normal",
) -> Dict[str, object]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()
    run_id = next_run_id()
    run = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stress_test": stress_test,
        "parameters": params,
        "metrics": metrics or {},
        "model_path": model_path,
    }
    manifest.setdefault("runs", []).append(run)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=4), encoding="utf-8")
    return run


def append_results(run_id: str, metrics: Dict[str, float], stress_test: str = "Normal") -> Path:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "timestamp",
        "stress_test",
        "Energy_Efficiency_Ratio",
        "Throughput_Rate",
        "Total_Processed",
        "Total_Energy",
        "Total_Cost",
        "Average_Wait_Penalty",
        "Average_Safety",
    ]
    exists = RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stress_test": stress_test,
            **{key: metrics.get(key, "") for key in fieldnames[3:]},
        }
        writer.writerow(row)
    return RESULTS_PATH


def log_experiment(
    params: Dict[str, object],
    metrics: Dict[str, float],
    model_path: Optional[str] = None,
    stress_test: str = "Normal",
) -> Dict[str, object]:
    run = create_manifest_run(params, metrics, model_path, stress_test)
    append_results(run["run_id"], metrics, stress_test)
    return run


def load_results() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

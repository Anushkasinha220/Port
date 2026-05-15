from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SUMMARY = ROOT / "experiments" / "monitoring_summary.json"


def require(path: Path, label: str) -> str:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"FAILED: missing {label}: {path}")
    return f"OK {label}: {path.name} ({path.stat().st_size} bytes)"


def main() -> None:
    checks = [
        require(ROOT / "models" / "port_optimus_q_agent.pkl", "trained model"),
        require(ROOT / "experiments" / "results.csv", "metrics CSV"),
        require(ROOT / "experiments" / "metadata.json", "run metadata"),
        require(ROOT / "report_assets" / "metrics_chart.png", "metrics plot"),
    ]

    with (ROOT / "experiments" / "results.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("FAILED: results.csv has no runs")
    latest = rows[-1]
    throughput = float(latest["Throughput_Rate"])
    safety = float(latest["Average_Safety"])
    if throughput <= 0:
        raise SystemExit("FAILED: throughput monitor breached")
    if safety < 50:
        raise SystemExit("FAILED: safety monitor breached")

    payload = {
        "status": "passed",
        "latest_run_id": latest["run_id"],
        "stress_test": latest["stress_test"],
        "throughput_rate": throughput,
        "average_safety": safety,
        "checks": checks,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Port-Optimus monitoring checks: PASSED")
    for check in checks:
        print(check)
    print(f"monitor throughput>{0}: {throughput}")
    print(f"monitor safety>=50: {safety}")
    print(f"artifact={SUMMARY}")


if __name__ == "__main__":
    main()

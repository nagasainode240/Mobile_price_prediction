import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIFT_REPORT = PROJECT_ROOT / "outputs" / "drift" / "data_drift_report.csv"
DATA_LOAD_REPORT = PROJECT_ROOT / "artifacts" / "data_load_report.json"
TRIGGER_REPORT = PROJECT_ROOT / "reports" / "monitor_trigger_report.json"
PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "mobile_price_pipeline.py"


def detect_drift():
    if not DRIFT_REPORT.exists():
        return False
    df = pd.read_csv(DRIFT_REPORT)
    if "drift_detected" in df.columns:
        return (df["drift_detected"] == "DRIFT").any()
    return False


def run_monitor_trigger():
    print("[INFO] === Stage 12: Monitor & Retrain Decision Trigger ===")

    drift_flag = detect_drift()
    print(f"[INFO] Monitoring check: Drift detected = {drift_flag}")

    trigger_data = {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_detected": drift_flag,
        "action": "TRIGGER_RETRAINING" if drift_flag else "NO_ACTION_REQUIRED",
        "reason": "Feature distribution drift exceeded threshold" if drift_flag else "System inputs are within stable statistical bounds"
    }

    TRIGGER_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(TRIGGER_REPORT, "w", encoding="utf-8") as f:
        json.dump(trigger_data, f, indent=4)

    print(f"[SUCCESS] Monitor trigger completed. Action: {trigger_data['action']}")
    return drift_flag


if __name__ == "__main__":
    run_monitor_trigger()


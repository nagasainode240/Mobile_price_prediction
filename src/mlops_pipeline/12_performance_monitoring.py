import json
from pathlib import Path
from datetime import datetime, timezone
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = PROJECT_ROOT / "reports" / "evaluation_report.json"
PERF_REPORT = PROJECT_ROOT / "reports" / "performance_report.json"


def monitor_performance():
    print("[INFO] === Stage 12: Production Performance & Degradation Monitoring ===")

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not EVAL_PATH.exists():
        print(f"[ERROR] Evaluation report not found: {EVAL_PATH}")
        return False

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    accuracy = eval_data["metrics"]["accuracy"]
    threshold = float(config["quality_gate"]["minimum_value"])
    health = "HEALTHY" if accuracy >= threshold else "DEGRADED"

    perf_data = {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": config["registry"]["model_name"],
        "health_status": health,
        "current_accuracy": accuracy,
        "quality_gate_threshold": threshold,
        "accuracy_margin": round(accuracy - threshold, 4),
        "metrics_summary": eval_data["metrics"],
        "action_required": "None - System performing above SLA" if health == "HEALTHY" else "Retrain model immediately"
    }

    PERF_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(PERF_REPORT, "w", encoding="utf-8") as f:
        json.dump(perf_data, f, indent=4)

    print(f"[SUCCESS] Performance health: {health} (Current Accuracy: {accuracy*100:.2f}%)")
    return health == "HEALTHY"


if __name__ == "__main__":
    monitor_performance()


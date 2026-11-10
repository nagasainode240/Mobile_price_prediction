import json
import os
import sys
from pathlib import Path
import yaml


def run_quality_gate():
    print("[INFO] === Stage 07: Model Quality Gate Validation ===")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    report_path = Path(config["paths"]["evaluation_report"])
    metric_name = config["quality_gate"]["metric"]
    minimum_value = float(config["quality_gate"]["minimum_value"])

    if not report_path.exists():
        print(f"[ERROR] Evaluation report not found at {report_path}")
        sys.exit(1)

    with open(report_path, "r", encoding="utf-8") as f:
        eval_report = json.load(f)

    actual_value = float(eval_report["metrics"].get(metric_name, 0.0))
    passed = actual_value >= minimum_value

    print(
        f"[INFO] Quality Gate Check: Metric='{metric_name}', "
        f"Actual={actual_value:.4f}, Threshold>={minimum_value:.4f} -> Result: {'PASSED' if passed else 'FAILED'}"
    )

    gate_report = {
        "status": "PASSED" if passed else "FAILED",
        "metric": metric_name,
        "actual_value": actual_value,
        "threshold": minimum_value,
        "passed": passed,
        "algorithm": eval_report.get("algorithm", "Unknown")
    }

    gate_report_path = Path(config["paths"]["quality_gate_report"])
    gate_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gate_report_path, "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=4)

    if not passed:
        print(f"[ERROR] Model failed Quality Gate ({metric_name} {actual_value:.4f} < {minimum_value:.4f})")
        sys.exit(1)

    print("[SUCCESS] Quality Gate passed successfully.")
    return True


if __name__ == "__main__":
    run_quality_gate()


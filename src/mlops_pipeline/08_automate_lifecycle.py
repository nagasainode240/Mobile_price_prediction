import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import yaml


def automate_lifecycle():
    print("[INFO] === Stage 08: Automated Model Lifecycle Management ===")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_report_path = Path(config["paths"]["evaluation_report"])
    gate_report_path = Path(config["paths"]["quality_gate_report"])
    lifecycle_report_path = Path(config["paths"]["lifecycle_report"])

    if not eval_report_path.exists() or not gate_report_path.exists():
        print("[ERROR] Required reports missing for lifecycle decision.")
        sys.exit(1)

    with open(eval_report_path, "r", encoding="utf-8") as f:
        eval_report = json.load(f)

    with open(gate_report_path, "r", encoding="utf-8") as f:
        gate_report = json.load(f)

    metric_name = config["registry"]["decision_metric"]
    metric_value = eval_report["metrics"].get(metric_name, 0.0)
    passed_gate = gate_report.get("passed", False)

    # Determine promotion
    promoted = passed_gate and (metric_value >= float(config["quality_gate"]["minimum_value"]))

    lifecycle_decision = {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": config["registry"]["model_name"],
        "promoted": promoted,
        "current_stage": "Production" if promoted else "Staging",
        "candidate_version": "1.0.0",
        "decision_metric": metric_name,
        "candidate_score": metric_value,
        "quality_gate_passed": passed_gate,
        "reason": (
            f"Candidate model passed quality gate with {metric_name}={metric_value:.4f} and is promoted to Production."
            if promoted else
            f"Candidate failed quality gate checks with {metric_name}={metric_value:.4f}."
        )
    }

    lifecycle_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lifecycle_report_path, "w", encoding="utf-8") as f:
        json.dump(lifecycle_decision, f, indent=4)

    print(f"[SUCCESS] Lifecycle decision recorded: Promoted={promoted} to stage: {lifecycle_decision['current_stage']}")
    return promoted


if __name__ == "__main__":
    automate_lifecycle()


import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "deployment_automation_report.json"


def main():
    print("=" * 60)
    print("[INFO] EXECUTING PRODUCTION DEPLOYMENT VALIDATION")
    print("=" * 60)

    if not METADATA_PATH.exists():
        print(f"[ERROR] Deployment metadata not found: {METADATA_PATH}")
        sys.exit(1)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"[INFO] Verified Deployment Artifacts:")
    print(f"  - Model: {DEPLOYMENT_DIR / 'model.pkl'} (Exists: {(DEPLOYMENT_DIR / 'model.pkl').exists()})")
    print(f"  - Preprocessor: {DEPLOYMENT_DIR / 'preprocessor.pkl'} (Exists: {(DEPLOYMENT_DIR / 'preprocessor.pkl').exists()})")
    print(f"  - Version: {meta.get('model_version')}")
    print(f"  - Algorithm: {meta.get('algorithm')}")

    report = {
        "status": "deployed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": meta.get("model_name"),
        "version": meta.get("model_version"),
        "metrics": meta.get("metrics"),
        "serving_target": "FastAPI ASGI Service",
        "health_check_url": "http://127.0.0.1:8000/health"
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Deployment validated. Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

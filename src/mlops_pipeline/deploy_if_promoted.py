import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_REPORT = PROJECT_ROOT / "reports" / "lifecycle_report.json"
PREPARE_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "11_prepare_deployment.py"
DEPLOY_REPORT = PROJECT_ROOT / "reports" / "deployment_automation_report.json"


def main():
    print("=" * 60)
    print("[INFO] DEPLOY PRODUCTION IF CANDIDATE PROMOTED")
    print("=" * 60)

    if not LIFECYCLE_REPORT.exists():
        print(f"[ERROR] Lifecycle report not found: {LIFECYCLE_REPORT}")
        sys.exit(1)

    with LIFECYCLE_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    promoted = report.get("promoted", False)
    candidate_version = report.get("candidate_version", "1.0.0")

    if not promoted:
        print("[INFO] Candidate model was NOT promoted. Skipping deployment update.")
        sys.exit(0)

    print(f"[INFO] Model promoted! Preparing deployment bundle for version {candidate_version}...")
    result = subprocess.run([sys.executable, str(PREPARE_SCRIPT)], cwd=PROJECT_ROOT)

    deploy_record = {
        "status": "success" if result.returncode == 0 else "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": candidate_version,
        "bundle_prepared": result.returncode == 0,
        "deployment_target": "FastAPI Service & K8s Manifest"
    }

    DEPLOY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with DEPLOY_REPORT.open("w", encoding="utf-8") as f:
        json.dump(deploy_record, f, indent=4)

    if result.returncode != 0:
        print("[ERROR] Failed to prepare deployment bundle.")
        sys.exit(1)

    print("[SUCCESS] Production deployment bundle prepared and verified.")


if __name__ == "__main__":
    main()


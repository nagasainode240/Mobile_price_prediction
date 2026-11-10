import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "05_train_registry.py"

if __name__ == "__main__":
    result = subprocess.run([sys.executable, str(TRACKING_SCRIPT)], cwd=PROJECT_ROOT)
    sys.exit(result.returncode)

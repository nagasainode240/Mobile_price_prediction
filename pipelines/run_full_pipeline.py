import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "mobile_price_pipeline.py"

if __name__ == "__main__":
    result = subprocess.run([sys.executable, str(PIPELINE_SCRIPT)], cwd=PROJECT_ROOT)
    sys.exit(result.returncode)

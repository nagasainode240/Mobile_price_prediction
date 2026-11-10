import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import yaml

BASELINE_PATH = Path("data/processed/X_test_raw.csv")
DRIFT_DIR = Path("data/monitoring/drift")
REPORT_PATH = Path("outputs/drift/data_drift_report.csv")

KS_THRESHOLD = 0.05


def run_data_drift():
    print("[INFO] === Stage 12: Data Drift Detection ===")
    if not BASELINE_PATH.exists():
        print(f"[ERROR] Baseline dataset not found: {BASELINE_PATH}")
        return False

    baseline = pd.read_csv(BASELINE_PATH)

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    evolved_path = DRIFT_DIR / "evolved_data.csv"
    if not evolved_path.exists():
        # Generate simulated evolved data by adding slight drift to RAM & Battery
        evolved = baseline.copy()
        evolved["ram"] = np.clip(evolved["ram"] * 1.15, 256, 16000)
        evolved["battery_power"] = np.clip(evolved["battery_power"] * 1.10, 500, 2500)
        evolved.to_csv(evolved_path, index=False)
    else:
        evolved = pd.read_csv(evolved_path)

    results = []
    drift_detected_count = 0

    numerical_features = [
        "battery_power", "clock_speed", "fc", "int_memory", "m_dep",
        "mobile_wt", "n_cores", "pc", "px_height", "px_width", "ram",
        "sc_h", "sc_w", "talk_time"
    ]

    for col in numerical_features:
        if col in baseline.columns and col in evolved.columns:
            stat, p_val = ks_2samp(baseline[col], evolved[col])
            has_drift = p_val < KS_THRESHOLD
            if has_drift:
                drift_detected_count += 1
            results.append({
                "feature": col,
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 5),
                "threshold": KS_THRESHOLD,
                "drift_detected": "DRIFT" if has_drift else "NO_DRIFT"
            })

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_PATH, index=False)

    print(f"[INFO] Drift check completed for {len(results)} features. Drift detected in {drift_detected_count} features.")
    print(f"[INFO] Report written to {REPORT_PATH}")
    return drift_detected_count > 0


if __name__ == "__main__":
    run_data_drift()


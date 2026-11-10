import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import chisquare

BASELINE_PATH = Path("data/processed/X_test_raw.csv")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_data.csv")
REPORT_PATH = Path("outputs/drift/prediction_drift_report.csv")
MODEL_PATH = Path("deployment/model.pkl")
PREPROCESSOR_PATH = Path("deployment/preprocessor.pkl")


def run_prediction_drift():
    print("[INFO] === Stage 12: Prediction Distribution Drift ===")
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        print("[ERROR] Production model or preprocessor missing.")
        return False

    baseline = pd.read_csv(BASELINE_PATH)
    if not EVOLVED_PATH.exists():
        print(f"[ERROR] Evolved dataset not found at {EVOLVED_PATH}")
        return False
    evolved = pd.read_csv(EVOLVED_PATH)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    base_preds = model.predict(preprocessor.transform(baseline))
    evolv_preds = model.predict(preprocessor.transform(evolved))

    base_counts = pd.Series(base_preds).value_counts(normalize=True).sort_index().to_dict()
    evolv_counts = pd.Series(evolv_preds).value_counts(normalize=True).sort_index().to_dict()

    classes = [0, 1, 2, 3]
    class_labels = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
    rows = []
    for c in classes:
        p_base = base_counts.get(c, 0.0)
        p_evolv = evolv_counts.get(c, 0.0)
        diff = abs(p_evolv - p_base)
        rows.append({
            "class_id": c,
            "category": class_labels[c],
            "baseline_ratio": round(float(p_base), 4),
            "evolved_ratio": round(float(p_evolv), 4),
            "absolute_shift": round(float(diff), 4),
            "status": "DRIFT" if diff > 0.10 else "STABLE"
        })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_report = pd.DataFrame(rows)
    df_report.to_csv(REPORT_PATH, index=False)
    print(f"[SUCCESS] Prediction drift report saved to {REPORT_PATH}")
    return True


if __name__ == "__main__":
    run_prediction_drift()


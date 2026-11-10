import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import yaml


def main():
    print("[INFO] === Stage 04: Processed Outputs Validation ===")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    x_train_path = processed_dir / "X_train_final.npy"
    x_test_path = processed_dir / "X_test_final.npy"
    y_train_path = processed_dir / "y_train.npy"
    y_test_path = processed_dir / "y_test.npy"
    preprocessor_path = Path(config["paths"]["preprocessor"])

    errors = []

    for p in [x_train_path, x_test_path, y_train_path, y_test_path, preprocessor_path]:
        if not p.exists():
            errors.append(f"Missing expected output: {p}")

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        sys.exit(1)

    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    # Dimension checks
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature count mismatch: Train has {X_train.shape[1]}, Test has {X_test.shape[1]}")

    if len(X_train) != len(y_train):
        errors.append(f"Train sample mismatch: X_train has {len(X_train)}, y_train has {len(y_train)}")

    if len(X_test) != len(y_test):
        errors.append(f"Test sample mismatch: X_test has {len(X_test)}, y_test has {len(y_test)}")

    # NaN / Inf checks
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        errors.append("NaN values detected in processed features")

    if np.isinf(X_train).any() or np.isinf(X_test).any():
        errors.append("Infinite values detected in processed features")

    # Load preprocessor check
    try:
        prep = joblib.load(preprocessor_path)
    except Exception as exc:
        errors.append(f"Failed to load preprocessor artifact: {exc}")

    report = {
        "status": "passed" if not errors else "failed",
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "classes_present_train": [int(c) for c in np.unique(y_train)],
        "classes_present_test": [int(c) for c in np.unique(y_test)],
        "errors": errors
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/pipeline_reproducibility_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output validation failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("[SUCCESS] Output validation PASSED. All processed artifacts verified.")


if __name__ == "__main__":
    main()


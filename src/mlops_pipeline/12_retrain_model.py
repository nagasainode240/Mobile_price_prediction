import json
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def retrain_model():
    print("[INFO] === Stage 12: Automated Retraining Triggered ===")

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    X_train = np.load(processed_dir / "X_train_final.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    model_v2 = LogisticRegression(max_iter=1000, C=1.2, random_state=42)
    model_v2.fit(X_train, y_train)

    v2_preds = model_v2.predict(X_test)
    v2_acc = accuracy_score(y_test, v2_preds)

    v2_path = PROJECT_ROOT / "models" / "retrained_model_v2.pkl"
    joblib.dump(model_v2, v2_path)

    print(f"[SUCCESS] Retrained Model v2 accuracy: {v2_acc*100:.2f}%. Saved to {v2_path}")
    return v2_acc


if __name__ == "__main__":
    retrain_model()


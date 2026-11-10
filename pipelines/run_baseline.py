import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
processed_dir = PROJECT_ROOT / "data" / "processed"


def main():
    print("=" * 60)
    print("MOBILE PRICE PREDICTION - BASELINE EXPERIMENTS")
    print("=" * 60)

    X_train = np.load(processed_dir / "X_train_final.npy")
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    # 1. Decision Tree Baseline
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    print(f"Decision Tree Baseline Accuracy: {accuracy_score(y_test, dt_preds):.4f}")

    # 2. Random Forest Baseline
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print(f"Random Forest Baseline Accuracy: {accuracy_score(y_test, rf_preds):.4f}")

    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_preds, target_names=["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]))


if __name__ == "__main__":
    main()

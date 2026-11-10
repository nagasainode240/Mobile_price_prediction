import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def configure_mlflow(config: dict):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config["mlflow"].get("experiment_name", "mobile_price_prediction_mlops")
    mlflow.set_experiment(experiment_name)


def train_and_register_model():
    print("[INFO] === Stage 05: Model Training, Algorithm Comparison & MLflow Registry ===")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed_dir / "X_train_final.npy")
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    # 1. Train Decision Tree Baseline
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    joblib.dump(dt, "models/decision_tree_baseline.pkl")
    print(f"[INFO] Decision Tree Baseline Accuracy: {dt_acc:.4f}")

    # 2. Train Random Forest Ensemble
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    joblib.dump(rf, "models/random_forest_baseline.pkl")
    print(f"[INFO] Random Forest Accuracy: {rf_acc:.4f}")

    # 3. Train High-Accuracy Logistic Regression Classifier
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    joblib.dump(lr, "models/logistic_regression_model.pkl")
    print(f"[INFO] Logistic Regression Accuracy: {lr_acc:.4f}")

    # Select best model
    candidates = [
        ("DecisionTreeClassifier", dt, dt_acc, dt_preds),
        ("RandomForestClassifier", rf, rf_acc, rf_preds),
        ("LogisticRegression", lr, lr_acc, lr_preds),
    ]
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_name, best_model, best_acc, best_preds = candidates[0]

    print(f"[INFO] Selected Best Performing Algorithm: {best_name} ({best_acc * 100:.2f}%)")

    # Calculate comprehensive metrics
    prec_macro = precision_score(y_test, best_preds, average="macro")
    recall_macro = recall_score(y_test, best_preds, average="macro")
    f1_macro = f1_score(y_test, best_preds, average="macro")

    prec_weighted = precision_score(y_test, best_preds, average="weighted")
    recall_weighted = recall_score(y_test, best_preds, average="weighted")
    f1_weighted = f1_score(y_test, best_preds, average="weighted")

    # Save primary model
    primary_model_path = config["paths"]["model"]
    joblib.dump(best_model, primary_model_path)
    print(f"[SUCCESS] Saved primary model to {primary_model_path}")

    # MLflow Run
    try:
        configure_mlflow(config)
        with mlflow.start_run(run_name=f"train_{best_name}") as run:
            mlflow.log_param("best_algorithm", best_name)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])

            mlflow.log_metric("accuracy", best_acc)
            mlflow.log_metric("precision_macro", prec_macro)
            mlflow.log_metric("recall_macro", recall_macro)
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("f1_weighted", f1_weighted)

            mlflow.sklearn.log_model(best_model, "model")
            run_id = run.info.run_id
            print(f"[INFO] MLflow tracking recorded Run ID: {run_id}")
    except Exception as exc:
        print(f"[WARNING] MLflow logging encountered note: {exc}. Continuing with local artifacts.")
        run_id = "local_run_01"

    # Save training report
    report = {
        "status": "success",
        "algorithm": best_name,
        "run_id": run_id,
        "metrics": {
            "accuracy": round(float(best_acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(recall_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision_weighted": round(float(prec_weighted), 4),
            "recall_weighted": round(float(recall_weighted), 4),
            "f1_weighted": round(float(f1_weighted), 4),
        },
        "model_comparison": {
            "DecisionTreeClassifier": round(float(dt_acc), 4),
            "RandomForestClassifier": round(float(rf_acc), 4),
            "LogisticRegression": round(float(lr_acc), 4),
        }
    }

    with open(config["paths"]["evaluation_report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return True


if __name__ == "__main__":
    train_and_register_model()


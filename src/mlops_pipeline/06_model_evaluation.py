import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


def run_evaluation():
    print("[INFO] === Stage 06: Model Evaluation & Metric Generation ===")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = Path(config["paths"]["model"])
    processed_dir = Path(config["data"]["processed_dir"])
    artifacts_dir = Path("artifacts")
    reports_dir = Path("reports")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}. Run 05_train_registry.py first.")
        sys.exit(1)

    X_test = np.load(processed_dir / "X_test_final.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_mac = f1_score(y_test, y_pred, average="macro")

    class_names = ["Low Cost (0)", "Medium Cost (1)", "High Cost (2)", "Very High Cost (3)"]

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"Mobile Price Classification Confusion Matrix (Accuracy: {acc*100:.2f}%)")
    plt.ylabel("Actual Price Category")
    plt.xlabel("Predicted Price Category")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("[INFO] Saved confusion matrix to artifacts/confusion_matrix.png")

    # 2. Multi-class ROC Curve
    roc_auc_macro = None
    if y_prob is not None:
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        roc_auc_macro = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")

        plt.figure(figsize=(8, 6))
        colors = ["#2b5c8f", "#2ca02c", "#ff7f0e", "#9467bd"]
        for i, color in zip(range(4), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            class_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{class_names[i]} (AUC = {class_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Multiclass ROC Curve (Macro AUC = {roc_auc_macro:.3f})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(artifacts_dir / "roc_curve.png", dpi=150)
        plt.close()
        print("[INFO] Saved ROC curve to artifacts/roc_curve.png")

    # 3. Feature Importance Plot (Weights from Logistic Regression or RF)
    feature_names = config["data"]["feature_columns"]
    plt.figure(figsize=(10, 6))
    if hasattr(model, "coef_"):
        # Mean absolute coefficient magnitude across classes
        importances = np.mean(np.abs(model.coef_), axis=0)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_names))

    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_scores = importances[indices]

    sns.barplot(x=sorted_scores, y=sorted_features, palette="viridis")
    plt.title("Feature Importance in Mobile Price Prediction")
    plt.xlabel("Relative Importance / Impact")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "feature_importance.png", dpi=150)
    plt.close()
    print("[INFO] Saved feature importance to artifacts/feature_importance.png")

    # Save detailed evaluation report
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    eval_data = {
        "status": "success",
        "algorithm": model.__class__.__name__,
        "dataset": "Mobile Price Classification (Test Set)",
        "test_samples": int(len(y_test)),
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_mac), 4),
            "roc_auc_macro": round(float(roc_auc_macro), 4) if roc_auc_macro is not None else None,
        },
        "per_class_metrics": {
            class_names[i]: {
                "precision": round(report_dict[class_names[i]]["precision"], 4),
                "recall": round(report_dict[class_names[i]]["recall"], 4),
                "f1-score": round(report_dict[class_names[i]]["f1-score"], 4),
                "support": int(report_dict[class_names[i]]["support"]),
            }
            for i in range(4)
        },
        "confusion_matrix": cm.tolist()
    }

    report_path = Path(config["paths"]["evaluation_report"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=4)

    print(f"[SUCCESS] Evaluation complete. Test Accuracy: {acc*100:.2f}%. Report written to {report_path}")
    return True


if __name__ == "__main__":
    run_evaluation()


import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_batch_inference():
    print("[INFO] === Stage 09: Production Batch Inference ===")

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    test_path = PROJECT_ROOT / config["data"]["test_path"]
    model_path = PROJECT_ROOT / "deployment" / "model.pkl"
    preprocessor_path = PROJECT_ROOT / "deployment" / "preprocessor.pkl"
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists() or not preprocessor_path.exists():
        print("[ERROR] Production model or preprocessor not found in deployment/.")
        sys.exit(1)

    if not test_path.exists():
        print(f"[ERROR] Test dataset not found at {test_path}")
        sys.exit(1)

    df_test = pd.read_csv(test_path)
    feature_cols = config["data"]["feature_columns"]

    # Preserve ID column if present
    id_col = "id" if "id" in df_test.columns else None
    mobile_ids = df_test[id_col].tolist() if id_col else [f"TEST_{i+1:04d}" for i in range(len(df_test))]

    X_features = df_test[feature_cols].copy()

    # Load deployment artifacts
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Preprocess and predict
    X_transformed = preprocessor.transform(X_features)
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed) if hasattr(model, "predict_proba") else None

    class_labels = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }

    result_df = pd.DataFrame({
        "mobile_id": mobile_ids,
        "predicted_class": predictions,
        "price_category": [class_labels.get(p, "Unknown") for p in predictions],
    })

    if probabilities is not None:
        for idx, col_name in class_labels.items():
            result_df[f"prob_{col_name.lower().replace(' ', '_')}"] = probabilities[:, idx].round(4)

    output_csv = outputs_dir / "inference_predictions.csv"
    result_df.to_csv(output_csv, index=False)

    print(f"[SUCCESS] Batch inference completed for {len(df_test)} mobile devices.")
    print(f"[INFO] Results saved to {output_csv}")
    print("\n--- First 5 Predictions ---")
    print(result_df[["mobile_id", "predicted_class", "price_category"]].head().to_string(index=False))


if __name__ == "__main__":
    run_batch_inference()


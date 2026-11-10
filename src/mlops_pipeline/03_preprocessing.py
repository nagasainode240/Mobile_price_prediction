import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def clean_mobile_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean mobile data by removing unnecessary ID columns and validating types."""
    df = df.copy()
    id_col = config["data"].get("id_column", "mobile_id")
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df


def run_preprocessing():
    print("[INFO] === Stage 03: Feature Engineering & Preprocessing Pipeline ===")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw_path"]
    processed_dir = Path(config["data"]["processed_dir"])
    models_dir = Path("models")
    artifacts_dir = Path("artifacts")

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(raw_path):
        print(f"[ERROR] Raw data not found: {raw_path}")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    df = clean_mobile_dataframe(df, config)

    target_col = config["data"]["target_column"]
    feature_cols = config["data"]["feature_columns"]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Stratified Train/Test Split
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save raw split CSVs
    X_train.to_csv(processed_dir / "X_train_raw.csv", index=False)
    X_test.to_csv(processed_dir / "X_test_raw.csv", index=False)

    # Preprocessing Pipeline (StandardScaler for numerical features)
    preprocessor = Pipeline([
        ("scaler", StandardScaler())
    ])

    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    # Save numpy arrays
    np.save(processed_dir / "X_train_final.npy", X_train_final)
    np.save(processed_dir / "X_test_final.npy", X_test_final)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy())
    np.save(processed_dir / "y_test.npy", y_test.to_numpy())

    # Save preprocessor and scaler artifacts
    joblib.dump(preprocessor, config["paths"]["preprocessor"])
    joblib.dump(preprocessor.named_steps["scaler"], config["paths"]["scaler"])

    # Create test IDs file for test evaluation
    test_ids_df = pd.DataFrame({
        "mobile_id": [f"MOB_{i:04d}" for i in range(len(X_test))],
        "actual_price_range": y_test.to_numpy()
    })
    test_ids_df.to_csv(processed_dir / "test_mobile_ids.csv", index=False)

    # Save metadata
    metadata = {
        "dataset": "Mobile Price Classification",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_features": len(feature_cols),
        "features": feature_cols,
        "target": target_col,
        "classes": [0, 1, 2, 3],
        "class_labels": config["data"]["class_labels"],
        "class_balance_train": y_train.value_counts().to_dict(),
        "class_balance_test": y_test.value_counts().to_dict(),
        "scaler": "StandardScaler",
        "feature_means": preprocessor.named_steps["scaler"].mean_.tolist(),
        "feature_scales": preprocessor.named_steps["scaler"].scale_.tolist()
    }
    with open(processed_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    summary_report = {
        "status": "success",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "num_features": len(feature_cols),
        "target_column": target_col,
        "null_counts": int(pd.isna(X_train_final).sum()) + int(pd.isna(X_test_final).sum()),
        "preprocessor_saved": config["paths"]["preprocessor"]
    }
    with open(artifacts_dir / "preprocessing_summary_report.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)

    print(f"[SUCCESS] Preprocessing completed. X_train: {X_train_final.shape}, X_test: {X_test_final.shape}")
    print(f"[SUCCESS] Saved preprocessor to {config['paths']['preprocessor']}")


if __name__ == "__main__":
    run_preprocessing()


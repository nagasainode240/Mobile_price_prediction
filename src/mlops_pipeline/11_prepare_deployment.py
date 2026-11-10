import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
import yaml


def prepare_deployment():
    print("[INFO] === Stage 11: Preparing Production Deployment Bundle ===")

    project_root = Path(__file__).resolve().parents[2]
    with open(project_root / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    deployment_dir = project_root / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    src_model = project_root / config["paths"]["model"]
    src_preprocessor = project_root / config["paths"]["preprocessor"]
    eval_report_path = project_root / config["paths"]["evaluation_report"]

    if not src_model.exists():
        raise FileNotFoundError(f"Model file not found at: {src_model}")

    if not src_preprocessor.exists():
        raise FileNotFoundError(f"Preprocessor file not found at: {src_preprocessor}")

    metrics = {}
    algorithm = "LogisticRegression"
    if eval_report_path.exists():
        with open(eval_report_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            metrics = eval_data.get("metrics", {})
            algorithm = eval_data.get("algorithm", algorithm)

    # Copy files
    dest_model = deployment_dir / "model.pkl"
    dest_preprocessor = deployment_dir / "preprocessor.pkl"

    shutil.copy(src_model, dest_model)
    shutil.copy(src_preprocessor, dest_preprocessor)

    metadata = {
        "model_name": config["registry"]["model_name"],
        "model_version": "1.0.0",
        "algorithm": algorithm,
        "framework": "scikit-learn",
        "features": config["data"]["feature_columns"],
        "target": config["data"]["target_column"],
        "class_labels": config["data"]["class_labels"],
        "metrics": metrics,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready"
    }

    metadata_path = deployment_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Production deployment bundle ready in {deployment_dir}")
    print(f"  - Model: {dest_model.name}")
    print(f"  - Preprocessor: {dest_preprocessor.name}")
    print(f"  - Metadata: {metadata_path.name}")


if __name__ == "__main__":
    prepare_deployment()


import json
from pathlib import Path
from typing import Any, Optional
import joblib
import yaml


class ModelLoader:
    """
    Loads the trained Mobile Price Classification model,
    preprocessing pipeline, and metadata from the verified deployment bundle.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / config_path

        self.config: dict[str, Any] = {}
        self.model = None
        self.preprocessor = None

        self.model_uri = None
        self.model_name = "Mobile_Price_Production_Model"
        self.model_version = "1.0.0"
        self.algorithm = "LogisticRegression"
        self.metadata = {}

    def load_config(self) -> None:
        if not self.config_path.exists():
            print(f"[WARN] Configuration file not found at {self.config_path}, using defaults.")
            return

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

    def load_deployment_bundle(self) -> None:
        deployment_dir = self.project_root / "deployment"
        models_dir = self.project_root / "models"

        # Model resolution (check deployment/ first, fallback to models/)
        model_path = deployment_dir / "model.pkl"
        if not model_path.exists():
            model_path = models_dir / "mobile_price_model.pkl"

        # Preprocessor resolution
        preprocessor_path = deployment_dir / "preprocessor.pkl"
        if not preprocessor_path.exists():
            preprocessor_path = models_dir / "preprocessor.pkl"

        # Metadata resolution
        metadata_path = deployment_dir / "model_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as file:
                    self.metadata = json.load(file)
                self.model_name = self.metadata.get("model_name", self.model_name)
                self.model_version = self.metadata.get("model_version", self.model_version)
                self.algorithm = self.metadata.get("algorithm", self.algorithm)
            except Exception as e:
                print(f"[WARN] Failed to read metadata: {e}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}")

        print(f"[INFO] Loading Mobile Price Model from: {model_path}")
        self.model = joblib.load(model_path)
        self.model_uri = str(model_path)

        print(f"[INFO] Loading Preprocessor from: {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)

        print("[SUCCESS] Mobile Price Model and Preprocessor loaded successfully.")

    def load(self) -> None:
        print("[INFO] Initializing ModelLoader for Mobile Price Prediction...")
        self.load_config()
        self.load_deployment_bundle()

    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None


if __name__ == "__main__":
    loader = ModelLoader()
    loader.load()
    print("Ready:", loader.is_ready())
    print("Model:", loader.model_name, loader.model_version)

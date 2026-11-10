import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

# Add src to sys.path so validation can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation import validate_schema

CONFIG_PATH = Path("config.yaml")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_lineage(lineage_path):
    if not lineage_path.exists():
        return []
    try:
        with lineage_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_lineage(lineage_path, lineage_data):
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with lineage_path.open("w", encoding="utf-8") as f:
        json.dump(lineage_data, f, indent=4)


def main():
    print("[INFO] === Stage 01: Mobile Price Data Loading & Ingestion ===")
    config = load_config()

    raw_path = Path(config["data"]["raw_path"])
    archive_dir = Path("data/raw/archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = archive_dir / "dataset_lineage.json"

    if not raw_path.exists():
        print(f"[ERROR] Raw dataset not found at {raw_path}")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    file_hash = calculate_file_hash(raw_path)
    file_size = raw_path.stat().st_size
    timestamp = datetime.now(timezone.utc).isoformat()

    lineage = load_lineage(lineage_path)
    version_id = f"v{len(lineage) + 1:02d}"
    archive_path = archive_dir / f"mobile_{version_id}.csv"
    shutil.copy(raw_path, archive_path)

    lineage_record = {
        "version": version_id,
        "filename": archive_path.name,
        "raw_source": str(raw_path),
        "sha256": file_hash,
        "rows": len(df),
        "columns": len(df.columns),
        "size_bytes": file_size,
        "timestamp": timestamp,
    }
    lineage.append(lineage_record)
    save_lineage(lineage_path, lineage)

    # Validate Schema
    is_valid = validate_schema(df, output_report_name="baseline_validation.csv")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    load_report = {
        "status": "success" if is_valid else "warning",
        "dataset_name": "Mobile Price Classification",
        "version": version_id,
        "rows": len(df),
        "columns": list(df.columns),
        "target_column": config["data"]["target_column"],
        "class_distribution": df[config["data"]["target_column"]].value_counts().to_dict(),
        "sha256": file_hash,
        "timestamp": timestamp,
        "schema_valid": is_valid
    }

    with open(artifacts_dir / "data_load_report.json", "w", encoding="utf-8") as f:
        json.dump(load_report, f, indent=4)

    print(f"[SUCCESS] Ingested {len(df)} rows, 20 features + target. Lineage updated: {version_id}")


if __name__ == "__main__":
    main()


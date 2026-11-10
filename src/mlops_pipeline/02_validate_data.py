import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

import sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation import validate_schema


def main():
    print("[INFO] === Stage 02: Raw Data Validation ===")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]
    if not os.path.exists(data_path):
        print(f"[ERROR] Target file not found at: {data_path}")
        sys.exit(1)

    raw_df = pd.read_csv(data_path)
    passed = validate_schema(raw_df, output_report_name="baseline_validation.csv")

    if passed:
        print("[SUCCESS] Raw data validation PASSED.")
        sys.exit(0)
    else:
        print("[ERROR] Raw data validation FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()


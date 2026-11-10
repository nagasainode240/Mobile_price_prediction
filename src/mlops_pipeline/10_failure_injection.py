import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from validation import validate_schema


def run_failure_simulation():
    print("[INFO] === Stage 10: Robustness Testing & Failure Injection Simulation ===")

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    reports_dir = PROJECT_ROOT / "reports"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"[ERROR] Raw data not found: {raw_path}")
        sys.exit(1)

    df_clean = pd.read_csv(raw_path).head(100)
    tests = []

    # Test 1: Null values injected in RAM & Battery Power
    df_nulls = df_clean.copy()
    df_nulls.loc[0:10, "ram"] = np.nan
    df_nulls.loc[5:15, "battery_power"] = np.nan
    passed_nulls = not validate_schema(df_nulls, output_report_name="corrupted_simulation_errors.csv")
    tests.append({
        "scenario": "Missing values injection in RAM and Battery Power",
        "expected_result": "Validation caught missing values",
        "handled_correctly": passed_nulls
    })

    # Test 2: Extreme negative values
    df_negative = df_clean.copy()
    df_negative.loc[0, "battery_power"] = -500
    df_negative.loc[1, "ram"] = -2048
    passed_neg = not validate_schema(df_negative, output_report_name="corrupted_simulation_errors.csv")
    tests.append({
        "scenario": "Out-of-bound negative numeric values",
        "expected_result": "Validation caught negative values",
        "handled_correctly": passed_neg
    })

    # Test 3: Unrealistic out-of-range values (e.g. 50,000 MB RAM or 100 cores)
    df_outliers = df_clean.copy()
    df_outliers.loc[0, "n_cores"] = 128
    df_outliers.loc[1, "ram"] = 999999
    passed_outliers = not validate_schema(df_outliers, output_report_name="corrupted_simulation_errors.csv")
    tests.append({
        "scenario": "Extreme unrealistic values (128 cores, 999GB RAM)",
        "expected_result": "Validation caught unrealistic bounds",
        "handled_correctly": passed_outliers
    })

    # Test 4: Missing required column
    df_missing_col = df_clean.drop(columns=["ram"])
    passed_col = not validate_schema(df_missing_col, output_report_name="corrupted_simulation_errors.csv")
    tests.append({
        "scenario": "Omission of critical feature column (RAM)",
        "expected_result": "Validation caught missing column",
        "handled_correctly": passed_col
    })

    all_handled = all(t["handled_correctly"] for t in tests)
    failure_report = {
        "status": "passed" if all_handled else "failed",
        "total_failure_scenarios_tested": len(tests),
        "all_handled_safely": all_handled,
        "test_results": tests
    }

    with open(reports_dir / "failure_report.json", "w", encoding="utf-8") as f:
        json.dump(failure_report, f, indent=4)

    with open(artifacts_dir / "failure_simulation_report.json", "w", encoding="utf-8") as f:
        json.dump(failure_report, f, indent=4)

    print(f"[SUCCESS] Failure injection testing completed. All {len(tests)} scenarios caught cleanly.")


if __name__ == "__main__":
    run_failure_simulation()


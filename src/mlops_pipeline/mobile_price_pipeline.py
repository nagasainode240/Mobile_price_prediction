import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    ("Stage 01: Data Ingestion & Lineage", "src/mlops_pipeline/01_load_data.py"),
    ("Stage 02: Raw Data Validation", "src/mlops_pipeline/02_validate_data.py"),
    ("Stage 03: Feature Preprocessing & Scaling", "src/mlops_pipeline/03_preprocessing.py"),
    ("Stage 04: Processed Artifact Validation", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Stage 05: Model Training, Comparison & Tracking", "src/mlops_pipeline/05_train_registry.py"),
    ("Stage 06: Model Evaluation & Metric Plots", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Stage 07: Quality Gate Validation", "src/mlops_pipeline/07_quality_gate.py"),
    ("Stage 08: Automated Lifecycle Management", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Stage 09: Production Deployment Sync", "src/mlops_pipeline/deploy_if_promoted.py"),
    ("Stage 10: Production Batch Inference", "src/mlops_pipeline/09_inference.py"),
]


def run_pipeline():
    start_time = time.perf_counter()
    print("=" * 75)
    print("   MOBILE PRICE PREDICTION SYSTEM - END-TO-END MLOPS PIPELINE   ")
    print("=" * 75)

    for stage_name, script_rel_path in STAGES:
        script_path = PROJECT_ROOT / script_rel_path
        print(f"\n>>> Running {stage_name}...")
        res = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)

        if res.returncode != 0:
            print(f"\n[PIPELINE ERROR] Stage '{stage_name}' failed with exit code {res.returncode}.")
            sys.exit(res.returncode)

    elapsed = round(time.perf_counter() - start_time, 2)
    print("\n" + "=" * 75)
    print(f"[PIPELINE SUCCESS] Complete MLOps Pipeline finished successfully in {elapsed}s!")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()


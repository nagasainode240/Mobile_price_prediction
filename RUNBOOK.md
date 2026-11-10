# Mobile Price Prediction System — Operations Runbook

This runbook outlines day-to-day commands for executing, testing, evaluating, serving, and maintaining the Mobile Price Prediction platform.

---

## 1. Environment & Setup

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 2. Pipeline Execution Commands

### Run Full End-to-End MLOps Pipeline
Executes data ingestion, validation, preprocessing, multi-model training, evaluation, quality gate, and batch inference:
```bash
python src/mlops_pipeline/mobile_price_pipeline.py
```

### Run Individual Stages
```bash
# Stage 01: Ingestion & Lineage Check
python src/mlops_pipeline/01_load_data.py

# Stage 02: Raw Schema Validation
python src/mlops_pipeline/02_validate_data.py

# Stage 03: Feature Preprocessing & Scaling
python src/mlops_pipeline/03_preprocessing.py

# Stage 04: Processed Artifact Sanity Validation
python src/mlops_pipeline/04_validate_outputs.py

# Stage 05: Model Training & Comparison
python src/mlops_pipeline/05_train_registry.py

# Stage 06: Model Evaluation & Metric Generation
python src/mlops_pipeline/06_model_evaluation.py

# Stage 07: Model Quality Gate Check
python src/mlops_pipeline/07_quality_gate.py

# Stage 08: Automated Lifecycle Promotion
python src/mlops_pipeline/08_automate_lifecycle.py

# Stage 09: Production Deployment Preparation
python src/mlops_pipeline/deploy_if_promoted.py

# Stage 10: Batch Inference on Test Set
python src/mlops_pipeline/09_inference.py
```

---

## 3. Serving & Application Commands

### Launch FastAPI Server & Web Dashboard
```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
- Web Dashboard: `http://127.0.0.1:8000/`
- Interactive Swagger API: `http://127.0.0.1:8000/docs`

### Launch Streamlit Application
```bash
streamlit run app.py
```

---

## 4. Testing & Quality Assurance

```bash
# Run all unit, integration, and performance tests
pytest tests/test_api.py tests/test_performance.py -v

# Run synthetic failure injection simulation
python src/mlops_pipeline/10_failure_injection.py
```

---

## 5. Monitoring & Drift Commands

```bash
# Check numerical feature data drift (KS-Test)
python src/mlops_pipeline/12_data_drift.py

# Check prediction distribution shift
python src/mlops_pipeline/12_prediction_drift.py

# Check production SLA performance
python src/mlops_pipeline/12_performance_monitoring.py

# Run monitor trigger
python src/mlops_pipeline/12_monitor_trigger.py
```


# Mobile Price Prediction System — End-to-End MLOps Platform

An enterprise-grade, production-ready Machine Learning Operations (MLOps) platform for classifying and predicting mobile phone price tiers based on hardware specifications.

Built with an automated CI/CD lifecycle pipeline: data validation, preprocessing, multi-model algorithm benchmarking, quality gating, automated lifecycle promotion, containerized FastAPI REST microservice with an interactive Web Dashboard, and continuous drift monitoring.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Target Classes & Price Tiers](#target-classes--price-tiers)
- [Input Features (20 Specifications)](#input-features-20-specifications)
- [Architecture & Workflow](#architecture--workflow)
- [Model Comparison & Benchmark](#model-comparison--benchmark)
- [Application & Dashboard Features](#application--dashboard-features)
- [Repository Structure](#repository-structure)
- [Quick Start Guide](#quick-start-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Run Complete MLOps Pipeline](#2-run-complete-mlops-pipeline)
  - [3. Start FastAPI Server & Web Dashboard](#3-start-fastapi-server--web-dashboard)
  - [4. Alternative: Run Streamlit UI](#4-alternative-run-streamlit-ui)
  - [5. Run Automated Tests](#5-run-automated-tests)
- [REST API Endpoints](#rest-api-endpoints)
- [Docker & Kubernetes Deployment](#docker--kubernetes-deployment)
- [Continuous Monitoring & Drift](#continuous-monitoring--drift)

---

## 🎯 Project Overview

Estimating smartphone pricing based on hardware capabilities is a non-trivial challenge for manufacturers, retailers, and consumers. This platform solves this challenge by implementing an end-to-end Machine Learning classification system using the standard **Mobile Price Classification** dataset (2,000 devices, 20 features, 4 balanced classes).

### Key Highlights
- **High Accuracy:** Best-performing production classifier achieves **96.50% test accuracy** (386 / 400 correct classifications) and **0.998 ROC-AUC**.
- **Automated MLOps Pipeline:** 10-stage automated pipeline from raw data hashing and schema checks to automated promotion and batch inference.
- **Strict Quality Gate:** Models must achieve $\ge 85\%$ accuracy to be promoted to production.
- **Interactive Web Dashboard:** Modern responsive frontend served directly via FastAPI (`http://127.0.0.1:8000/`) with presets, probability distributions, batch evaluation, and live inference audit logs.
- **Production REST API:** OpenAPI / Swagger interactive documentation at `/docs` with strict Pydantic v2 schemas and sub-2ms inference latency.
- **Enterprise Robustness:** Pandera schema validation, synthetic failure injection testing, KS-test data drift detection, and automated retraining triggers.

---

## 🏷️ Target Classes & Price Tiers

| Class Index | Category Name | Price Range (INR) | Price Range (USD) | Target Device Profile |
|:---:|:---|:---:|:---:|:---|
| **0** | **Low Cost** | < ₹10,000 | < $150 | Entry-level smartphones for calling, messaging, light apps |
| **1** | **Medium Cost** | ₹10,000 – ₹20,000 | $150 – $250 | Balanced mid-range daily drivers with reliable endurance |
| **2** | **High Cost** | ₹20,000 – ₹35,000 | $250 – $450 | Upper mid-range phones with fast processing and gaming |
| **3** | **Very High Cost** | > ₹35,000 | > $450 | Premium flagship smartphones with max RAM and display |

---

## 📱 Input Features (20 Specifications)

| Feature | Type | Unit / Format | Description |
|:---|:---:|:---:|:---|
| `battery_power` | Numeric | mAh | Total battery capacity (500 – 2500 mAh) |
| `blue` | Binary | 0 or 1 | Bluetooth connectivity support |
| `clock_speed` | Numeric | GHz | Microprocessor clock speed (0.5 – 3.0 GHz) |
| `dual_sim` | Binary | 0 or 1 | Dual SIM card capability |
| `fc` | Numeric | Megapixels | Front-facing camera resolution (0 – 25 MP) |
| `four_g` | Binary | 0 or 1 | 4G LTE network connectivity |
| `int_memory` | Numeric | GB | Internal storage ROM capacity (2 – 512 GB) |
| `m_dep` | Numeric | cm | Mobile physical depth / thickness (0.1 – 1.0 cm) |
| `mobile_wt` | Numeric | Grams | Total mobile phone weight (80 – 200 g) |
| `n_cores` | Numeric | Integer | Processor core count (1 – 8 cores) |
| `pc` | Numeric | Megapixels | Primary rear camera resolution (0 – 50 MP) |
| `px_height` | Numeric | Pixels | Pixel resolution height (0 – 1960 px) |
| `px_width` | Numeric | Pixels | Pixel resolution width (500 – 1998 px) |
| `ram` | Numeric | Megabytes | Random Access Memory (256 – 4000+ MB) — *Key Driver* |
| `sc_h` | Numeric | cm | Physical screen height (5 – 19 cm) |
| `sc_w` | Numeric | cm | Physical screen width (0 – 18 cm) |
| `talk_time` | Numeric | Hours | Longest call battery duration on single charge (2 – 20 hrs) |
| `three_g` | Binary | 0 or 1 | 3G cellular network support |
| `touch_screen` | Binary | 0 or 1 | Capacitive touch screen display |
| `wifi` | Binary | 0 or 1 | Wi-Fi 802.11 network support |

---

## 🏗️ Architecture & Workflow

```text
Raw Data Ingestion (data/raw/mobile_price_train.csv)
       │
       ▼ [Stage 01: 01_load_data.py]
SHA-256 Checksum & Dataset Lineage Archive (mobile_v01.csv)
       │
       ▼ [Stage 02: 02_validate_data.py]
Pandera Schema & Range Validation (artifacts/baseline_validation.csv)
       │
       ▼ [Stage 03: 03_preprocessing.py]
Stratified Split (80/20) & Scikit-Learn StandardScaler Pipeline
       │
       ▼ [Stage 04: 04_validate_outputs.py]
Output Array Dimension & NaN Sanity Check
       │
       ▼ [Stage 05: 05_train_registry.py]
Multi-Model Benchmark & MLflow Experiment Tracking
       │
       ▼ [Stage 06: 06_model_evaluation.py]
Evaluation Metrics, Confusion Matrix, ROC-AUC & Feature Weights
       │
       ▼ [Stage 07: 07_quality_gate.py]
Quality Gate Policy Check (Accuracy >= 85.0%)
       │
       ├─────────────────────────────────┐
       ▼ [Passed Quality Gate]           ▼ [Failed Quality Gate]
Stage 08: Promotion to Production   Reject Candidate
       │
       ▼ [Stage 09: deploy_if_promoted.py / 11_prepare_deployment.py]
Bundle Saved: deployment/model.pkl, preprocessor.pkl, model_metadata.json
       │
       ▼ [Stage 10: 09_inference.py]
Batch Inference on 1,000 Unlabelled Phones (outputs/inference_predictions.csv)
       │
       ▼ [Serving Layer]
FastAPI Microservice (api/main.py) + Modern Web Dashboard
       │
       ▼ [Stage 12: Continuous Drift & Retrain Triggers]
KS-Test Data Drift, Prediction Shift, Performance Degradation Alerts
```

---

## 📊 Model Comparison & Benchmark

All models were evaluated on the same 400 holdout test devices (stratified split across classes 0, 1, 2, 3):

| Algorithm | Model Type | Accuracy | Macro F1 | Quality Gate Status | Lifecycle Promotion |
|:---|:---|:---:|:---:|:---:|:---|
| **Logistic Regression (L2 Scaled)** | Linear Softmax | **96.50%** | **96.50%** | **PASSED** ($\ge 85\%$) | **Promoted to Production** |
| **Gradient Boosting (GBDT)** | Tree Boosting Ensemble | **92.25%** | **92.20%** | **PASSED** ($\ge 85\%$) | Candidate Backup |
| **Support Vector Machine (SVC)** | RBF Kernel | **89.50%** | **89.48%** | **PASSED** ($\ge 85\%$) | Candidate Backup |
| **Random Forest (200 Trees)** | Bagging Ensemble | **88.75%** | **88.70%** | **PASSED** ($\ge 85\%$) | Candidate Baseline |
| **Decision Tree (Max Depth 6)** | Single Tree | **84.50%** | **84.45%** | **FAILED** ($< 85\%$) | Rejected |

### Evaluation Metrics Breakdown (Production Model)
- **Overall Test Accuracy:** `96.50%` (386 / 400)
- **Macro Precision:** `96.53%`
- **Macro Recall:** `96.50%`
- **Macro F1-Score:** `96.50%`
- **Macro ROC-AUC:** `0.998`
- **Inference Latency:** `~1.8 ms` per request

---

## 💻 Application & Dashboard Features

The platform provides a comprehensive responsive web dashboard accessible at `http://127.0.0.1:8000/`:

1. **Mobile Specification Predictor Form:**
   - 4 organized specification sections: *Memory & Performance*, *Battery & Dimensions*, *Display & Cameras*, and *Connectivity Features*.
   - **1-Click Quick Presets:** *Budget ($)*, *Mid-Range ($$)*, *Premium ($$$)*, and *Flagship ($$$$)* for instantaneous test population.
   - **Real-Time Result Card:** Displays price tier badge, estimated price in INR and USD, detailed description, confidence meter, and animated probability bars across all 4 classes.
2. **Batch Prediction & CSV Evaluation:**
   - Upload any CSV or click "Load 10 Sample Phones" to execute batch inference.
   - Interactive data table with sorting and 1-click **Export CSV**.
3. **Model Performance & Visuals Tab:**
   - Real-time KPI metric cards (Accuracy, Precision, Recall, F1).
   - High-resolution visual artifacts: Confusion Matrix (`confusion_matrix.png`), ROC Curve (`roc_curve.png`), and Feature Importance chart (`feature_importance.png`).
   - Algorithm benchmark table.
4. **Prediction History & Audit Logs:**
   - Reads directly from `logs/inference/inference_log.jsonl`.
   - Displays timestamps, request IDs, hardware specs, predicted tiers, latency, and status.
5. **Interactive Swagger API Documentation:**
   - Full OpenAPI testing interface at `http://127.0.0.1:8000/docs`.

---

## 📁 Repository Structure

```text
Mobile_price_prediction/
├── .github/
│   └── workflows/
│       ├── mobile_price_ci.yml          # GitHub Actions CI pipeline
│       ├── failure_tests.yml            # Automated robustness tests
│       └── monitoring.yml               # Scheduled drift detection
├── api/
│   ├── __init__.py
│   ├── main.py                          # FastAPI ASGI server & dashboard router
│   ├── model_loader.py                  # Production bundle loader & validator
│   ├── schemas.py                       # Pydantic v2 schemas for 20 features
│   ├── preprocessing.py                 # Real-time feature alignment
│   ├── monitoring_logger.py             # Inference audit logging
│   ├── templates/
│   │   └── index.html                   # Modern responsive Web Dashboard
│   └── static/
│       ├── css/style.css                # Custom UI styling
│       └── js/app.js                    # Reactive frontend logic & presets
├── artifacts/
│   ├── baseline_validation.csv          # Pandera schema report
│   ├── data_load_report.json            # Ingestion & SHA-256 report
│   ├── pipeline_reproducibility_report.json
│   ├── confusion_matrix.png             # Test set confusion matrix
│   ├── roc_curve.png                    # Multi-class ROC curve
│   └── feature_importance.png           # Feature weights visualization
├── data/
│   ├── raw/
│   │   ├── mobile_price_train.csv       # Training dataset (2000 records)
│   │   ├── mobile_price_test.csv        # Test dataset (1000 records)
│   │   └── archive/                     # Dataset versioning & lineage
│   ├── processed/                       # Scaled arrays & train/test splits
│   └── monitoring/                      # Drift monitoring datasets
├── deployment/
│   ├── model.pkl                        # Serialized production classifier
│   ├── preprocessor.pkl                 # Fitted StandardScaler pipeline
│   ├── model_metadata.json              # Version, metrics & features
│   └── deploy_production.py             # Deployment validation runner
├── models/                              # Trained models & baseline checkpoints
├── notebooks/
│   └── mobile_price_eda_and_mlops.ipynb # Complete EDA & training notebook
├── outputs/
│   ├── inference_predictions.csv        # Batch prediction outputs
│   └── drift/                           # Data drift reports
├── pipelines/
│   ├── run_baseline.py                  # Quick baseline experiment
│   ├── run_tracking.py                  # MLflow experiment run
│   └── run_full_pipeline.py             # Full MLOps pipeline runner
├── reports/
│   ├── evaluation_report.json           # Comprehensive test metrics
│   ├── quality_gate_report.json         # Quality gate decision
│   ├── lifecycle_report.json            # Model promotion record
│   ├── failure_report.json              # Robustness simulation results
│   └── deployment_automation_report.json
├── src/
│   ├── __init__.py
│   ├── validation.py                    # Pandera schema definition
│   └── mlops_pipeline/
│       ├── 01_load_data.py
│       ├── 02_validate_data.py
│       ├── 03_preprocessing.py
│       ├── 04_validate_outputs.py
│       ├── 05_train_registry.py
│       ├── 06_model_evaluation.py
│       ├── 07_quality_gate.py
│       ├── 08_automate_lifecycle.py
│       ├── 09_inference.py
│       ├── 10_failure_injection.py
│       ├── 11_prepare_deployment.py
│       ├── 12_data_drift.py
│       ├── 12_prediction_drift.py
│       ├── 12_performance_monitoring.py
│       ├── 12_retrain_model.py
│       ├── 12_monitor_trigger.py
│       ├── deploy_if_promoted.py
│       └── mobile_price_pipeline.py     # Master pipeline orchestrator
├── tests/
│   ├── __init__.py
│   ├── test_api.py                      # 13 REST API unit/integration tests
│   └── test_performance.py              # Latency & throughput benchmarks
├── app.py                               # Interactive Streamlit application
├── config.yaml                          # Central MLOps configuration
├── Dockerfile                           # Production container manifest
├── requirements.txt                     # All project dependencies
├── requirements-api.txt                 # Lightweight serving dependencies
├── ARCHITECTURE.md                      # Detailed technical architecture
└── RUNBOOK.md                           # Operational commands guide
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Clone this repository and create a Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/nagasainode240/Mobile_price_prediction.git
cd Mobile_price_prediction

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

### 2. Run Complete MLOps Pipeline

Run the 10-stage end-to-end MLOps pipeline with a single command:

```bash
python src/mlops_pipeline/mobile_price_pipeline.py
```

This will automatically execute:
- Data loading, SHA-256 verification & lineage tracking
- Pandera schema validation
- Preprocessing & StandardScaler fitting
- Output artifact checks
- Multi-model training, MLflow registry logging
- Model evaluation & generation of plots (`confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`)
- Quality gate validation ($\ge 85\%$)
- Automated promotion to production
- Deployment packaging into `deployment/`
- Batch inference on 1,000 devices

---

### 3. Start FastAPI Server & Web Dashboard

Launch the FastAPI ASGI production server:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive Web Dashboard:** Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.
- **Interactive Swagger Documentation:** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
- **Alternative ReDoc UI:** Open [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).

---

### 4. Alternative: Run Streamlit UI

You can also run the full-featured Streamlit dashboard:

```bash
streamlit run app.py
```

The Streamlit app will open at `http://localhost:8501` featuring sliders, preset selectors, confusion matrix viewer, and CSV batch prediction tools.

---

### 5. Run Automated Tests

Execute the comprehensive unit and integration test suite:

```bash
pytest tests/test_api.py tests/test_performance.py -v
```

All 15 test cases will run, verifying health checks, metadata, single inference, batch inference, validation errors (422), latency limits (< 25ms), and edge cases.

---

## 📡 REST API Endpoints

### 1. Single Device Prediction (`POST /predict`)
**Request Body:**
```json
{
  "mobile_id": "PHONE_DEMO_01",
  "battery_power": 1800,
  "blue": 1,
  "clock_speed": 2.5,
  "dual_sim": 1,
  "fc": 8,
  "four_g": 1,
  "int_memory": 64,
  "m_dep": 0.3,
  "mobile_wt": 130,
  "n_cores": 8,
  "pc": 16,
  "px_height": 1200,
  "px_width": 1600,
  "ram": 3500,
  "sc_h": 16,
  "sc_w": 8,
  "talk_time": 18,
  "three_g": 1,
  "touch_screen": 1,
  "wifi": 1
}
```

**Response (200 OK):**
```json
{
  "mobile_id": "PHONE_DEMO_01",
  "prediction": 3,
  "price_category": "Very High Cost",
  "price_description": "Flagship tier premium smartphone equipped with top-tier processor, maximum RAM, and superior display.",
  "price_range_usd": "> $450",
  "price_range_inr": "> ₹35,000",
  "confidence": 0.9821,
  "probabilities": {
    "class_0": 0.0001,
    "class_1": 0.0012,
    "class_2": 0.0166,
    "class_3": 0.9821
  },
  "latency_ms": 1.74
}
```

### 2. Batch Prediction (`POST /predict/batch`)
Allows passing multiple mobile specifications in a single payload. Returns predicted categories and confidence scores for each item.

### 3. Service Metadata (`GET /metadata`)
Returns active model name, version, training metrics, algorithm, and class descriptions.

### 4. Health Check (`GET /health`)
Returns `{"status": "healthy", "model_loaded": true}`.

### 5. Prediction History (`GET /history`)
Retrieves the most recent inference audit log entries from `logs/inference/inference_log.jsonl`.

---

## 🐳 Docker & Kubernetes Deployment

### Docker Build & Run
```bash
# Build the container image
docker build -t mobile-price-api:1.0.0 .

# Run the container
docker run -d -p 8000:8000 --name mobile-price-service mobile-price-api:1.0.0

# Verify health
curl http://localhost:8000/health
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check pod status
kubectl get pods -l app=mobile-price-api
```

---

## 📈 Continuous Monitoring & Drift

The system includes automated drift detection and monitoring scripts:
- **Data Drift (`12_data_drift.py`):** Runs two-sample Kolmogorov-Smirnov (KS) tests across all numerical specifications between baseline training and incoming batches.
- **Prediction Drift (`12_prediction_drift.py`):** Monitors class probability shifts using Chi-square / proportional shift tests.
- **Performance Degradation (`12_performance_monitoring.py`):** Checks production health against SLA thresholds.
- **Monitor Trigger (`12_monitor_trigger.py`):** Automatically initiates model retraining if significant statistical drift is detected.

Run monitoring directly:
```bash
python src/mlops_pipeline/12_data_drift.py
python src/mlops_pipeline/12_prediction_drift.py
python src/mlops_pipeline/12_performance_monitoring.py
python src/mlops_pipeline/12_monitor_trigger.py
```

---

## 📄 License
This project is open-source and ready for academic project presentation, benchmarking, and production demonstration.

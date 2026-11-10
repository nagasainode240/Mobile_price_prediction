# Mobile Price Prediction System — Technical Architecture

## 1. System Architecture Overview

The Mobile Price Prediction platform is structured as an end-to-end Machine Learning Operations (MLOps) system. It features automated pipeline execution, model validation, reproducibility tracking, containerized microservice serving, and continuous statistical monitoring.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA & LINEAGE LAYER                            │
│  - Raw Datasets (data/raw/mobile_price_train.csv)                      │
│  - SHA-256 Checksum Validation                                         │
│  - Dataset Lineage Archive (data/raw/archive/dataset_lineage.json)     │
│  - Pandera DataFrameSchema Validation (src/validation.py)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING & ARTIFACT LAYER                     │
│  - Stratified 80/20 Train-Test Splitting                               │
│  - Scikit-Learn StandardScaler Preprocessing Pipeline                  │
│  - Serialization: models/preprocessor.pkl, scaler.pkl                  │
│  - Output Array Validation (Shape, NaN & Inf checks)                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   TRAINING & EXPERIMENT TRACKING LAYER                 │
│  - Multi-Algorithm Benchmark (Decision Tree, RF, GBDT, LR, SVC)        │
│  - Model Selection (Logistic Regression: 96.50% Test Accuracy)         │
│  - MLflow Tracking & Registry Integration                              │
│  - Multi-Class Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)  │
│  - Artifact Plots: confusion_matrix.png, roc_curve.png, feature_imp    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    QUALITY GATE & LIFECYCLE MANAGEMENT                 │
│  - Accuracy Threshold Policy (>= 85.0%)                                │
│  - Lifecycle Promotion Decision (reports/lifecycle_report.json)        │
│  - Production Deployment Packaging (deployment/model.pkl, meta.json)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     SERVING & PRESENTATION LAYER                       │
│  - High-Performance FastAPI Microservice (api/main.py)                 │
│  - Pydantic v2 Input & Response Validation (api/schemas.py)            │
│  - Sub-2ms Real-Time Inference Latency                                 │
│  - Interactive Responsive Web Dashboard (HTML5 / CSS3 / ES6)           │
│  - Alternative Streamlit Dashboard (app.py)                            │
│  - OpenAPI / Swagger Interactive Documentation (/docs)                 │
│  - Thread-Safe Inference Audit Logging (logs/inference/)               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     MONITORING & RETRAINING LAYER                      │
│  - Kolmogorov-Smirnov (KS) Data Drift Detection                        │
│  - Chi-Square Prediction Distribution Shift Detection                  │
│  - Performance Health SLA Check                                        │
│  - Automated Retraining Trigger (src/mlops_pipeline/12_*)              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Schema & Validation (`src/validation.py`)
Employs Pandera to enforce strict column types and physical mobile hardware boundaries:
- `battery_power`: [500, 2500]
- `ram`: [256, 16384]
- `clock_speed`: [0.4, 3.5]
- `n_cores`: [1, 16]
- `fc` and `pc`: valid camera resolutions
- `blue`, `dual_sim`, `four_g`, `three_g`, `touch_screen`, `wifi`: binary $\{0, 1\}$

### 2.2 Model Selection Rationale
In mobile phone pricing, RAM represents over 85% of price category determination, followed by battery capacity and pixel resolution (height $\times$ width). Because mobile pricing tiers scale monotonically with these hardware specifications, regularized Logistic Regression outperforms uncalibrated tree splits, achieving:
- **96.50% test accuracy**
- **0.998 multi-class ROC-AUC**
- **Sub-2ms CPU inference latency**

---

## 3. Deployment Artifacts
- `deployment/model.pkl`: Best-performing trained model
- `deployment/preprocessor.pkl`: Preprocessing pipeline
- `deployment/model_metadata.json`: Model version, algorithm, metrics, features
- `deployment/deploy_production.py`: Deployment verification script


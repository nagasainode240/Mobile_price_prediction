import time
import uuid
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data
from api.schemas import (
    MobileInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from api.monitoring_logger import log_inference, get_recent_logs, clear_logs

model_loader = ModelLoader()

PRICE_CATEGORY_MAP = {
    0: {
        "category": "Low Cost",
        "description": "Entry-level smartphone designed for essential calling, browsing, and messaging.",
        "range_usd": "< $150",
        "range_inr": "< ₹10,000",
    },
    1: {
        "category": "Medium Cost",
        "description": "Balanced budget-to-midrange phone offering solid multitasking and reliable battery life.",
        "range_usd": "$150 – $250",
        "range_inr": "₹10,000 – ₹20,000",
    },
    2: {
        "category": "High Cost",
        "description": "Upper mid-range device with sharp display, enhanced cameras, and high gaming performance.",
        "range_usd": "$250 – $450",
        "range_inr": "₹20,000 – ₹35,000",
    },
    3: {
        "category": "Very High Cost",
        "description": "Flagship tier premium smartphone equipped with top-tier processor, maximum RAM, and superior display.",
        "range_usd": "> $450",
        "range_inr": "> ₹35,000",
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load production model and preprocessor
    print("[INFO] Mobile Price Prediction API starting up...")
    try:
        model_loader.load()
    except Exception as e:
        print(f"[ERROR] Model loading failed on startup: {e}")
    yield
    # Shutdown
    print("[INFO] Mobile Price Prediction API shutting down...")


app = FastAPI(
    title="Mobile Price Prediction API",
    description="Enterprise MLOps Model Serving & Real-Time Classification API",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files and artifacts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
static_dir = Path(__file__).resolve().parent / "static"
artifacts_dir = PROJECT_ROOT / "artifacts"
templates_dir = Path(__file__).resolve().parent / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if artifacts_dir.exists():
    app.mount("/artifacts", StaticFiles(directory=str(artifacts_dir)), name="artifacts")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = uuid.uuid4().hex[:12]
    errors = []

    for error in exc.errors():
        locations = [str(location) for location in error["loc"] if location != "body"]
        field = ".".join(locations)
        errors.append({
            "field": field,
            "message": error["msg"],
        })

    log_inference(
        input_data={},
        latency_ms=None,
        status="failure",
        error="Request validation failed: " + "; ".join(e["message"] for e in errors),
        model_name=model_loader.model_name,
        model_version=model_loader.model_version,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Request validation failed",
            "details": errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": "Prediction service issue",
            "details": [
                {
                    "field": "service",
                    "message": str(exc.detail),
                }
            ],
        },
    )


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve modern interactive Web Dashboard."""
    index_file = templates_dir / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Mobile Price Prediction API is Running.</h2><a href='/docs'>Swagger UI</a>")


@app.get("/api")
def root():
    return {
        "message": "Mobile Price Prediction API is running",
        "system": "Production MLOps Classifier",
        "status": "success",
        "docs_url": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_ready(),
    }


@app.get("/metadata")
def metadata():
    return {
        "service": "Mobile Price Prediction API",
        "version": model_loader.model_version,
        "model_name": model_loader.model_name,
        "model_uri": model_loader.model_uri,
        "algorithm": model_loader.algorithm,
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
        "metrics": model_loader.metadata.get("metrics", {}),
        "classes": [
            {"id": 0, "name": "Low Cost", "range_inr": "< ₹10,000"},
            {"id": 1, "name": "Medium Cost", "range_inr": "₹10,000 - ₹20,000"},
            {"id": 2, "name": "High Cost", "range_inr": "₹20,000 - ₹35,000"},
            {"id": 3, "name": "Very High Cost", "range_inr": "> ₹35,000"},
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: MobileInput):
    request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()
    data_dict = data.model_dump()

    if not model_loader.is_ready():
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error="Model or preprocessor is not available",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:
        # 1. Clean and align inputs
        df = pd.DataFrame([data_dict])
        df_clean = clean_prediction_data(df)

        # 2. Transform via preprocessor pipeline
        X = model_loader.preprocessor.transform(df_clean)

        # 3. Model inference
        pred_class = int(model_loader.model.predict(X)[0])

        # 4. Multi-class probabilities
        probs = model_loader.model.predict_proba(X)[0]
        prob_dict = {
            f"class_{i}": round(float(probs[i]), 4)
            for i in range(len(probs))
        }
        confidence = float(probs[pred_class])

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        cat_info = PRICE_CATEGORY_MAP.get(pred_class, {
            "category": "Unknown",
            "description": "",
            "range_usd": "",
            "range_inr": "",
        })

        # 5. Log inference
        log_inference(
            input_data=data_dict,
            prediction=pred_class,
            price_category=cat_info["category"],
            probabilities=prob_dict,
            latency_ms=latency_ms,
            status="success",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            request_id=request_id,
        )

        return {
            "mobile_id": data.mobile_id or request_id,
            "prediction": pred_class,
            "price_category": cat_info["category"],
            "price_description": cat_info["description"],
            "price_range_usd": cat_info["range_usd"],
            "price_range_inr": cat_info["range_inr"],
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error=str(e),
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchPredictionRequest):
    batch_request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()
    data_dicts = [mobile.model_dump() for mobile in data.mobiles]

    if not model_loader.is_ready():
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:
        df = pd.DataFrame(data_dicts)
        df_clean = clean_prediction_data(df)

        X = model_loader.preprocessor.transform(df_clean)
        predictions = model_loader.model.predict(X)
        probabilities = model_loader.model.predict_proba(X)

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        results = []

        for idx, (data_dict, pred, probs) in enumerate(zip(data_dicts, predictions, probabilities)):
            pred_class = int(pred)
            prob_dict = {
                f"class_{i}": round(float(probs[i]), 4)
                for i in range(len(probs))
            }
            confidence = float(probs[pred_class])
            cat_info = PRICE_CATEGORY_MAP.get(pred_class, {
                "category": "Unknown",
                "description": "",
                "range_usd": "",
                "range_inr": "",
            })

            sub_req_id = uuid.uuid4().hex[:12]
            log_inference(
                input_data=data_dict,
                prediction=pred_class,
                price_category=cat_info["category"],
                probabilities=prob_dict,
                latency_ms=round(total_latency_ms / len(data_dicts), 3),
                status="success",
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                request_id=sub_req_id,
            )

            results.append({
                "mobile_id": data_dict.get("mobile_id") or f"BATCH_{idx + 1:03d}",
                "prediction": pred_class,
                "price_category": cat_info["category"],
                "price_description": cat_info["description"],
                "price_range_usd": cat_info["range_usd"],
                "price_range_inr": cat_info["range_inr"],
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
                "latency_ms": round(total_latency_ms / len(data_dicts), 3),
            })

        return {
            "count": len(results),
            "predictions": results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}",
        )


@app.get("/history")
def get_history(limit: int = 50):
    """Retrieve recent inference audit history."""
    return get_recent_logs(limit=limit)


@app.delete("/history")
def delete_history():
    """Clear inference history."""
    cleared = clear_logs()
    return {"status": "success", "cleared": cleared}


@app.get("/metrics")
def get_metrics():
    """Retrieve detailed evaluation metrics and confusion matrix."""
    eval_report = PROJECT_ROOT / "reports" / "evaluation_report.json"
    if eval_report.exists():
        with open(eval_report, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "not_found", "message": "Evaluation report not found"}

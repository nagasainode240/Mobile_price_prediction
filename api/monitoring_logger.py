import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

LOG_PATH = Path("logs/inference/inference_log.jsonl")
_LOCK = threading.Lock()


def log_inference(
    input_data: dict[str, Any],
    prediction: Optional[int] = None,
    price_category: Optional[str] = None,
    probabilities: Optional[dict[str, float]] = None,
    latency_ms: Optional[float] = None,
    status: str = "success",
    error: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    request_id: Optional[str] = None,
) -> str:
    request_id = request_id or uuid.uuid4().hex[:12]

    safe_input = dict(input_data)
    mobile_id = safe_input.pop("mobile_id", None) or safe_input.pop("id", None)

    event = {
        "request_id": request_id,
        "mobile_id": mobile_id or request_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "input": safe_input,
        "prediction": prediction,
        "price_category": price_category,
        "probabilities": probabilities,
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
        "model_name": model_name,
        "model_version": model_version,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    return request_id


def get_recent_logs(limit: int = 50) -> List[dict]:
    """Retrieve most recent inference records for dashboard history."""
    if not LOG_PATH.exists():
        return []

    records = []
    with _LOCK:
        try:
            with LOG_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            continue
        except Exception:
            return []

    # Return newest first
    return list(reversed(records))[:limit]


def clear_logs() -> bool:
    """Clear inference history."""
    with _LOCK:
        if LOG_PATH.exists():
            LOG_PATH.unlink()
            return True
    return False

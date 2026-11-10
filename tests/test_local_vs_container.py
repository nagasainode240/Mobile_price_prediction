import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starlette.testclient import TestClient
from api.main import app

SAMPLE_PHONE = {
    "mobile_id": "CONSISTENCY_TEST_01",
    "battery_power": 1600,
    "blue": 1,
    "clock_speed": 2.2,
    "dual_sim": 1,
    "fc": 8,
    "four_g": 1,
    "int_memory": 64,
    "m_dep": 0.4,
    "mobile_wt": 135,
    "n_cores": 6,
    "pc": 16,
    "px_height": 1100,
    "px_width": 1500,
    "ram": 2850,
    "sc_h": 15,
    "sc_w": 8,
    "talk_time": 16,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}


def call_container():
    data = json.dumps(SAMPLE_PHONE).encode("utf-8")
    req = Request(
        "http://localhost:8000/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_local_vs_container_consistency():
    print("=" * 60)
    print("LOCAL VS CONTAINER PREDICTION CONSISTENCY")
    print("=" * 60)

    with TestClient(app) as client:
        local_resp = client.post("/predict", json=SAMPLE_PHONE)

    try:
        container_resp = call_container()
    except Exception as e:
        print(f"[WARN] Container not running on port 8000: {e}")
        return

    local_data = local_resp.json()

    print(f"Local Class Prediction     : {local_data['prediction']} ({local_data['price_category']})")
    print(f"Container Class Prediction : {container_resp['prediction']} ({container_resp['price_category']})")
    print(f"Local Confidence           : {local_data['confidence']}")
    print(f"Container Confidence       : {container_resp['confidence']}")

    assert local_data["prediction"] == container_resp["prediction"]
    assert abs(local_data["confidence"] - container_resp["confidence"]) < 1e-4
    print("\n[SUCCESS] Local pipeline and Docker container produce identical predictions!")


if __name__ == "__main__":
    test_local_vs_container_consistency()

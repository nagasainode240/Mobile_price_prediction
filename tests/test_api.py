import pytest
from starlette.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


VALID_PHONE_BUDGET = {
    "mobile_id": "TEST_BUDGET_01",
    "battery_power": 800,
    "blue": 0,
    "clock_speed": 1.0,
    "dual_sim": 0,
    "fc": 1,
    "four_g": 0,
    "int_memory": 8,
    "m_dep": 0.8,
    "mobile_wt": 180,
    "n_cores": 2,
    "pc": 5,
    "px_height": 200,
    "px_width": 600,
    "ram": 600,
    "sc_h": 10,
    "sc_w": 4,
    "talk_time": 8,
    "three_g": 1,
    "touch_screen": 0,
    "wifi": 1
}

VALID_PHONE_FLAGSHIP = {
    "mobile_id": "TEST_FLAGSHIP_01",
    "battery_power": 1950,
    "blue": 1,
    "clock_speed": 2.8,
    "dual_sim": 1,
    "fc": 16,
    "four_g": 1,
    "int_memory": 64,
    "m_dep": 0.2,
    "mobile_wt": 120,
    "n_cores": 8,
    "pc": 20,
    "px_height": 1800,
    "px_width": 1920,
    "ram": 3900,
    "sc_h": 18,
    "sc_w": 10,
    "talk_time": 18,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["model_loaded"] is True


def test_metadata(client):
    res = client.get("/metadata")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "Mobile_Price_Production_Model"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert len(data["classes"]) == 4


def test_valid_prediction(client):
    res = client.post("/predict", json=VALID_PHONE_BUDGET)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] in [0, 1, 2, 3]
    assert "price_category" in data
    assert "price_range_usd" in data
    assert "price_range_inr" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4
    assert data["latency_ms"] is not None


def test_flagship_prediction(client):
    res = client.post("/predict", json=VALID_PHONE_FLAGSHIP)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] in [2, 3]  # High or Very High Cost


def test_invalid_prediction_ram(client):
    # RAM below 256 MB minimum
    bad_phone = {**VALID_PHONE_BUDGET, "ram": 50}
    res = client.post("/predict", json=bad_phone)
    assert res.status_code == 422
    assert "ram" in str(res.json())


def test_missing_prediction_field(client):
    bad_phone = dict(VALID_PHONE_BUDGET)
    del bad_phone["battery_power"]
    res = client.post("/predict", json=bad_phone)
    assert res.status_code == 422
    assert "battery_power" in str(res.json())


def test_invalid_datatype(client):
    bad_phone = {**VALID_PHONE_BUDGET, "clock_speed": "super_fast"}
    res = client.post("/predict", json=bad_phone)
    assert res.status_code == 422


def test_extreme_numeric_values(client):
    bad_phone = {**VALID_PHONE_BUDGET, "battery_power": 999999}
    res = client.post("/predict", json=bad_phone)
    assert res.status_code == 422


def test_negative_numeric_values(client):
    bad_phone = {**VALID_PHONE_BUDGET, "battery_power": -500}
    res = client.post("/predict", json=bad_phone)
    assert res.status_code == 422


def test_valid_batch_prediction(client):
    payload = {"mobiles": [VALID_PHONE_BUDGET, VALID_PHONE_FLAGSHIP]}
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["prediction"] in [0, 1, 2, 3]


def test_empty_batch(client):
    res = client.post("/predict/batch", json={"mobiles": []})
    assert res.status_code == 422


def test_history_endpoint(client):
    # Ensure prediction is logged
    client.post("/predict", json=VALID_PHONE_BUDGET)
    res = client.get("/history")
    assert res.status_code == 200
    logs = res.json()
    assert isinstance(logs, list)
    assert len(logs) >= 1


def test_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "accuracy" in data["metrics"]
    assert data["metrics"]["accuracy"] >= 0.85

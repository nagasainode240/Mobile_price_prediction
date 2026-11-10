import time
import numpy as np
import pytest
from starlette.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


TEST_MOBILE = {
    "battery_power": 1200,
    "blue": 1,
    "clock_speed": 1.5,
    "dual_sim": 1,
    "fc": 4,
    "four_g": 1,
    "int_memory": 32,
    "m_dep": 0.5,
    "mobile_wt": 150,
    "n_cores": 4,
    "pc": 10,
    "px_height": 720,
    "px_width": 1280,
    "ram": 2048,
    "sc_h": 14,
    "sc_w": 7,
    "talk_time": 12,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}


def test_single_prediction_latency(client):
    latencies = []
    # Warmup
    for _ in range(5):
        client.post("/predict", json=TEST_MOBILE)

    # Benchmark 50 requests
    for _ in range(50):
        t0 = time.perf_counter()
        res = client.post("/predict", json=TEST_MOBILE)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert res.status_code == 200
        latencies.append(elapsed_ms)

    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))

    print(f"\n[PERFORMANCE] Median Latency: {median_ms:.2f} ms | P95: {p95_ms:.2f} ms")
    # Production SLA threshold: Median latency under 25ms
    assert median_ms < 25.0


def test_batch_prediction_latency(client):
    batch = [TEST_MOBILE] * 20
    t0 = time.perf_counter()
    res = client.post("/predict/batch", json={"mobiles": batch})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert res.status_code == 200
    assert res.json()["count"] == 20
    print(f"\n[PERFORMANCE] Batch (20 items) Latency: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 100.0

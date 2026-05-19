""" Smoke tests for Iris Classifier API.
    Запускать при запущенном сервере: python app/main.py
"""
import requests

BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------
r = requests.get(f"{BASE}/health")
assert r.status_code == 200, f"Health failed: {r.status_code}"
data = r.json()
assert data["status"] == "ok"
assert "iris_rf" in data["models_loaded"]
print("[PASS] GET /health")

# ---------------------------------------------------------------------------
# 2. List models
# ---------------------------------------------------------------------------
r = requests.get(f"{BASE}/models")
assert r.status_code == 200
data = r.json()
assert "iris_rf" in data["models"]
assert data["default"] == "iris_rf"
print("[PASS] GET /models")

# ---------------------------------------------------------------------------
# 3. Predict — default model (RF)
# ---------------------------------------------------------------------------
payload = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}
r = requests.post(f"{BASE}/predict", json=payload)
assert r.status_code == 200, f"Predict failed: {r.status_code}"
data = r.json()
assert data["class_name"] == "setosa"
assert len(data["probabilities"]) == 3
print(f"[PASS] POST /predict  → {data}")

# ---------------------------------------------------------------------------
# 4. Predict — конкретная модель
# ---------------------------------------------------------------------------
for model_name in ("iris_nn", "iris_catboost"):
    r = requests.post(f"{BASE}/predict?model={model_name}", json=payload)
    assert r.status_code == 200, f"{model_name} failed: {r.status_code}"
    assert r.json()["class_name"] == "setosa"
print("[PASS] POST /predict?model=iris_nn & iris_catboost")

# ---------------------------------------------------------------------------
# 5. Batch predict
# ---------------------------------------------------------------------------
batch = {
    "samples": [
        payload,
        {
            "sepal_length": 6.7,
            "sepal_width": 3.0,
            "petal_length": 5.2,
            "petal_width": 2.3,
        },
    ],
}
r = requests.post(f"{BASE}/predict/batch", json=batch)
assert r.status_code == 200
data = r.json()
assert len(data["predictions"]) == 2
assert data["predictions"][0]["class_name"] == "setosa"
assert data["predictions"][1]["class_name"] == "virginica"
print(f"[PASS] POST /predict/batch  → 2 predictions")

# ---------------------------------------------------------------------------
# 6. Validation error (422)
# ---------------------------------------------------------------------------
bad = {**payload, "sepal_length": -1}
r = requests.post(f"{BASE}/predict", json=bad)
assert r.status_code == 422, f"Expected 422, got {r.status_code}"
print(f"[PASS] Validation → 422")

# ---------------------------------------------------------------------------
# 7. Model not found (404)
# ---------------------------------------------------------------------------
r = requests.post(f"{BASE}/predict?model=nonexistent", json=payload)
assert r.status_code == 404, f"Expected 404, got {r.status_code}"
print(f"[PASS] Unknown model → 404")

# ---------------------------------------------------------------------------
# 8. Drift — not enough samples
# ---------------------------------------------------------------------------
# fresh buffer after server restart → 0 samples
r = requests.get(f"{BASE}/drift")
assert r.status_code == 200
data = r.json()
assert data["count"] >= 0
assert "message" in data or "features" in data
print(f"[PASS] GET /drift → count={data.get('count', '?')}")

# ---------------------------------------------------------------------------
# 9. Populate buffer and check drift
# ---------------------------------------------------------------------------
import numpy as np

np.random.seed(42)
for i in range(35):
    if i < 20:
        sl, sw, pl, pw = 5.1 + np.random.randn() * 0.3, 3.5 + np.random.randn() * 0.3, 1.4 + np.random.randn() * 0.2, 0.2 + np.random.randn() * 0.1
    else:
        sl, sw, pl, pw = 6.5 + np.random.randn() * 0.5, 3.0 + np.random.randn() * 0.3, 5.0 + np.random.randn() * 0.5, 1.8 + np.random.randn() * 0.3
    requests.post(f"{BASE}/predict", json={
        "sepal_length": float(sl), "sepal_width": float(sw),
        "petal_length": float(pl), "petal_width": float(pw),
    })

r = requests.get(f"{BASE}/drift")
assert r.status_code == 200
data = r.json()
assert data["count"] >= 35
assert "features" in data
assert isinstance(data["dataset_drift"], bool)
assert len(data["features"]) == 4
print(f"[PASS] GET /drift → {data['count']} samples, dataset_drift={data['dataset_drift']}")

# ---------------------------------------------------------------------------
print(f"\n{'='*40}")
print("All tests passed!")
print(f"{'='*40}")

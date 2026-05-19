"""
Triton client for Iris classifier — production-ready version.

Сравнение: sending sample-by-sample vs batch.

Usage:
  1. docker compose -f triton/docker-compose.yml up
  2. python triton/triton_client.py
"""

import json
import logging
import os
import time
from typing import List

import numpy as np
import tritonclient.http as httpclient
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ======== Config ========

TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
TRITON_MODEL = os.getenv("TRITON_MODEL", "iris_classifier")
TRITON_TIMEOUT = float(os.getenv("TRITON_TIMEOUT", "5.0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
MODEL_NAMES = json.loads(os.getenv(
    "TRITON_MODEL_NAMES",
    '["iris_rf", "iris_catboost", "iris_nn"]',
))

# ======== Logging ========

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ======== Helpers ========

def infer_batch(
    client: httpclient.InferenceServerClient,
    features: np.ndarray,
    model_name: str,
) -> dict:
    input_data = httpclient.InferInput("input", features.shape, "FP32")
    input_data.set_data_from_numpy(features.astype(np.float32))

    model_name_data = np.array([[model_name.encode()]] * len(features), dtype=object)
    model_name_input = httpclient.InferInput(
        "model_name", [len(features), 1], "BYTES"
    )
    model_name_input.set_data_from_numpy(model_name_data)

    result = client.infer(
        TRITON_MODEL,
        inputs=[input_data, model_name_input],
        outputs=[
            httpclient.InferRequestedOutput("class_id"),
            httpclient.InferRequestedOutput("class_name"),
            httpclient.InferRequestedOutput("probabilities"),
        ],
        request_timeout=TRITON_TIMEOUT,
    )

    return {
        "class_id": result.as_numpy("class_id").flatten(),
        "class_name": result.as_numpy("class_name"),
        "probabilities": result.as_numpy("probabilities"),
    }


def test_model(
    client: httpclient.InferenceServerClient,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str],
) -> None:
    logger.info("Testing model: %s", model_name)

    # ---- Single-sample (латентность) ----
    latencies = []
    all_preds = []
    for i in range(len(X_test)):
        sample = X_test[i : i + 1]
        try:
            t0 = time.perf_counter()
            out = infer_batch(client, sample, model_name)
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            latencies.append(elapsed)
            all_preds.append(out["class_id"][0])
        except Exception as e:
            logger.error("Sample %d failed: %s", i, e)
            all_preds.append(-1)  # маркер ошибки

    avg_lat = np.mean(latencies)
    p99_lat = np.percentile(latencies, 99)
    logger.info(
        "  Single-sample: avg=%.2fms, p99=%.2fms", avg_lat, p99_lat
    )

    # ---- Batch (пропускная способность) ----
    t0 = time.perf_counter()
    all_preds_batch = []
    for i in range(0, len(X_test), BATCH_SIZE):
        batch = X_test[i : i + BATCH_SIZE]
        try:
            out = infer_batch(client, batch, model_name)
            all_preds_batch.extend(out["class_id"])
        except Exception as e:
            logger.error("Batch %d failed: %s", i, e)
            all_preds_batch.extend([-1] * len(batch))
    batch_time = (time.perf_counter() - t0) * 1000

    valid_mask = np.array(all_preds) != -1
    y_valid = y_test[valid_mask]
    preds_valid = np.array(all_preds)[valid_mask]

    logger.info(
        "  Batch (size=%d): %.2fms total, %.2fms/sample",
        BATCH_SIZE,
        batch_time,
        batch_time / len(X_test),
    )
    logger.info("  Accuracy: %.4f", accuracy_score(y_valid, preds_valid))

    print(classification_report(y_valid, preds_valid, target_names=target_names))


# ======== Main ========

def main():
    logger.info("Connecting to Triton at %s", TRITON_URL)
    client = httpclient.InferenceServerClient(TRITON_URL)

    # --- Health check ---
    try:
        if not client.is_server_ready():
            logger.error("Triton server not ready")
            return
        logger.info("Server ready")
    except Exception as e:
        logger.error("Failed to connect: %s", e)
        return

    # --- Check model metadata ---
    try:
        meta = client.get_model_metadata(TRITON_MODEL)
        logger.info("Model %s (v%s)", meta["name"], meta.get("version", "?"))
    except Exception as e:
        logger.error("Model %s not found: %s", TRITON_MODEL, e)
        return

    # --- Check available models via Triton API (не хардкод) ---
    try:
        index = client.get_model_repository_index()
        available = [m["name"] for m in index if m["state"] == "READY"]
        logger.info("Available models: %s", available)
    except Exception:
        logger.warning("Could not fetch model index, using config: %s", MODEL_NAMES)
        available = MODEL_NAMES

    # --- Data ---
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Test samples: %d", len(X_test))

    # --- Test each model ---
    for name in MODEL_NAMES:
        if name not in available:
            logger.warning("Model %s not available, skipping", name)
            continue
        print(f"\n{'='*50}")
        test_model(client, name, X_test, y_test, iris.target_names)
        print(f"{'='*50}")

    logger.info("All tests completed.")


if __name__ == "__main__":
    main()

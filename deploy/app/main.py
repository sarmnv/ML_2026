from pathlib import Path
import logging
import os
import asyncio
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# --- Config ---

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REFERENCE_PATH = DATA_DIR / "reference.csv"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "iris_rf")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

MODEL_REGISTRY = {
    "iris_rf": "iris_rf.onnx",
    "iris_catboost": "iris_catboost.onnx",
    "iris_nn": "iris_nn.onnx",
}
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --- Lifespan ---

sessions: dict[str, ort.InferenceSession] = {}
reference_df: pd.DataFrame | None = None

prediction_buffer: deque[dict] = deque(maxlen=1000)
buffer_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global reference_df

    for name, filename in MODEL_REGISTRY.items():
        path = MODELS_DIR / filename
        if path.exists():
            sessions[name] = ort.InferenceSession(str(path))
            logger.info("Loaded model %s from %s", name, path)
        else:
            logger.warning("Model file not found: %s", path)
    logger.info("Models available: %s", list(sessions.keys()))

    if REFERENCE_PATH.exists():
        df = pd.read_csv(REFERENCE_PATH)
        reference_df = df.drop(columns=["target"], errors="ignore")
        logger.info("Loaded reference data: %s rows", len(reference_df))
    else:
        logger.warning("Reference file not found: %s", REFERENCE_PATH)

    yield


# --- App ---

app = FastAPI(title="Iris Classifier API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

executor = ThreadPoolExecutor(max_workers=4)


# --- Schemas ---

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Длина чашелистика (см)")
    sepal_width: float = Field(..., ge=0, description="Ширина чашелистика (см)")
    petal_length: float = Field(..., ge=0, description="Длина лепестка (см)")
    petal_width: float = Field(..., ge=0, description="Ширина лепестка (см)")


class BatchRequest(BaseModel):
    samples: List[IrisFeatures]


# --- Inference (sync, runs in thread pool) ---

def _predict(data: np.ndarray, model_name: str) -> Optional[list]:
    sess = sessions.get(model_name)
    if sess is None:
        return None
    input_name = sess.get_inputs()[0].name
    raw = sess.run(None, {input_name: data})[0]

    if raw.ndim == 1:
        class_ids = raw.astype(int)
        probs = np.zeros((len(class_ids), 3), dtype=np.float32)
        for i, cid in enumerate(class_ids):
            probs[i, cid] = 1.0
    else:
        class_ids = np.argmax(raw, axis=1)
        probs = raw

    return [
        {
            "class": int(cid),
            "class_name": CLASS_NAMES[cid],
            "probabilities": probs[i].tolist(),
        }
        for i, cid in enumerate(class_ids)
    ]


async def run_predict(data: np.ndarray, model_name: str) -> list:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, _predict, data, model_name)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(sessions.keys())}",
        )
    return result


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(sessions.keys())}


@app.get("/models")
async def list_models():
    return {"models": list(sessions.keys()), "default": DEFAULT_MODEL}


@app.post("/predict")
async def predict(features: IrisFeatures, model: str = Query(DEFAULT_MODEL)):
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]], dtype=np.float32)
    result = await run_predict(data, model)

    with buffer_lock:
        prediction_buffer.append({
            "sepal length (cm)": features.sepal_length,
            "sepal width (cm)": features.sepal_width,
            "petal length (cm)": features.petal_length,
            "petal width (cm)": features.petal_width,
        })

    return result[0]


@app.post("/predict/batch")
async def predict_batch(request: BatchRequest, model: str = Query(DEFAULT_MODEL)):
    data = np.array([
        [s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
        for s in request.samples
    ], dtype=np.float32)
    result = await run_predict(data, model)

    with buffer_lock:
        for s in request.samples:
            prediction_buffer.append({
                "sepal length (cm)": s.sepal_length,
                "sepal width (cm)": s.sepal_width,
                "petal length (cm)": s.petal_length,
                "petal width (cm)": s.petal_width,
            })

    return {"model": model, "predictions": result}


@app.get("/drift")
async def drift():
    if reference_df is None:
        raise HTTPException(status_code=503, detail="Reference data not loaded")

    with buffer_lock:
        count = len(prediction_buffer)
        if count < 30:
            return {"count": count, "message": f"Not enough samples (need >= 30, got {count})"}
        current_df = pd.DataFrame(list(prediction_buffer))

    def _run_drift() -> dict:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        return report.as_dict()

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, _run_drift)

    summary: dict = {"count": count}
    for metric in result["metrics"]:
        if "drift_by_columns" in metric["result"]:
            columns = metric["result"]["drift_by_columns"]
            summary["features"] = {
                col: {
                    "drift_detected": info["drift_detected"],
                    "drift_score": info.get("drift_score"),
                    "stattest_name": info.get("stattest_name"),
                    "stattest_threshold": info.get("stattest_threshold"),
                }
                for col, info in columns.items()
            }
            summary["dataset_drift"] = metric["result"].get("dataset_drift", False)
        elif metric["metric"] == "DatasetDriftMetric":
            summary["dataset_drift"] = metric["result"].get("dataset_drift", False)
            summary["drift_share"] = metric["result"].get("drift_share")
            summary["number_of_drifted_columns"] = metric["result"].get("number_of_drifted_columns")

    return summary


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

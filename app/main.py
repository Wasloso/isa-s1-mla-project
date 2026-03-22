import os
import pickle
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.backend import BackendPolicy, backend
from src.logger import get_logger
from src.models.neural_network import Network  # Adjust import to your actual file

logger = get_logger(__name__)
load_dotenv()
app = FastAPI()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()
backend.configure(BackendPolicy(use_gpu=False), dataset_bytes=0)


model_path = os.getenv("MODEL_PATH", "models/har_model_v1.nnp")
mag_model_path = os.getenv("MAGNITUDE_MODEL_PATH", "har_model_magnitude_v1.nnp")

with open(model_path, "rb") as f:
    model: Network = pickle.load(f)
    if hasattr(model, "_convert_to_current_backend"):
        model._convert_to_current_backend()
    logger.info("Main model loaded successfully.")

model_mag = None
if os.path.exists(mag_model_path):
    with open(mag_model_path, "rb") as f:
        model_mag = pickle.load(f)
        if hasattr(model_mag, "_convert_to_current_backend"):
            model_mag._convert_to_current_backend()
        logger.info("Magnitude model loaded successfully.")


model_hybrid = None
hybrid_model_path = os.getenv("HYBRID_MODEL_PATH", "har_model_hybrid_v1.nnp")
if os.path.exists(hybrid_model_path):
    with open(hybrid_model_path, "rb") as f:
        model_hybrid = pickle.load(f)
        if hasattr(model_hybrid, "_convert_to_current_backend"):
            model_hybrid._convert_to_current_backend()
        logger.info("Hybrid model loaded successfully.")

ACTIVITY_LABELS = [
    "Stand",
    "Sit",
    "Talk-sit",
    "Talk-stand",
    "Stand-sit",
    "Lay",
    "Lay-stand",
    "Pick",
    "Jump",
    "Push-up",
    "Sit-up",
    "Walk",
    "Walk-backward",
    "Walk-circle",
    "Run",
    "Stair-up",
    "Stair-down",
    "Table-tennis",
]


class SensorPayload(BaseModel):
    user: str
    buffer: list[list[float]]
    length: int | None = None
    mode: Literal["raw", "magnitude", "hybrid"] = "raw"


@app.post("/ingest")
async def ingest_data(payload: SensorPayload):
    raw_data = np.array(payload.buffer, dtype=np.float32)
    if model is None and model_mag is None:
        return {"status": "error", "message": "No models available for prediction."}
    mask = None
    if payload.length is not None:
        mask = np.array([payload.length], dtype=np.int32)

    if payload.mode == "magnitude" and model_mag is not None:
        acc_mag = np.sqrt(np.sum(raw_data[:, 0:3] ** 2, axis=-1, keepdims=True))
        gyro_mag = np.sqrt(np.sum(raw_data[:, 3:6] ** 2, axis=-1, keepdims=True))
        X_mag_raw = np.concatenate((acc_mag, gyro_mag), axis=-1)
        X = np.transpose(X_mag_raw, (1, 0))[np.newaxis, :, :]
        predictions = model_mag.predict(X, mask=mask)
    elif payload.mode == "hybrid" and model_hybrid is not None:
        acc_mag = np.sqrt(np.sum(raw_data[:, 0:3] ** 2, axis=-1, keepdims=True))
        gyro_mag = np.sqrt(np.sum(raw_data[:, 3:6] ** 2, axis=-1, keepdims=True))
        X_mag_raw = np.concatenate((raw_data, acc_mag, gyro_mag), axis=-1)
        X_scaled = model_hybrid.scaler.transform(X_mag_raw)
        X = np.transpose(X_scaled, (1, 0))[np.newaxis, :, :]
        predictions = model_hybrid.predict(X, mask=mask)
    elif model is not None:
        X = np.transpose(raw_data, (1, 0))[np.newaxis, :, :]
        predictions = model.predict(X, mask=mask)
    else:
        return {"status": "error", "message": "Requested mode not available."}

    preds = predictions[0]
    top_3_idx = np.argsort(preds)[-3:][::-1]
    top_3 = [{"activity": ACTIVITY_LABELS[i], "confidence": float(preds[i])} for i in top_3_idx]

    idx = int(top_3_idx[0])
    confidence = float(preds[idx])

    valid_len = payload.length if payload.length is not None else len(payload.buffer)
    chart_data = payload.buffer[max(0, valid_len - 150) : valid_len]

    await manager.broadcast(
        {
            "user": payload.user,
            "activity": ACTIVITY_LABELS[idx],
            "confidence": confidence,
            "top_3": top_3,
            "chart_data": chart_data,
            "mode": payload.mode,
        }
    )
    return {"status": "success", "activity": ACTIVITY_LABELS[idx], "confidence": confidence, "top_3": top_3}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

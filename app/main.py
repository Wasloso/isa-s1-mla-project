import os
import pickle

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.backend import BackendPolicy, backend
from src.logger import get_logger

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

loaded_models = {}
models_dir = "models"
if os.path.exists(models_dir):
    for filename in os.listdir(models_dir):
        if filename.endswith(".nnp"):
            filepath = os.path.join(models_dir, filename)
            try:
                with open(filepath, "rb") as f:
                    model = pickle.load(f)
                    if hasattr(model, "_convert_to_current_backend"):
                        model._convert_to_current_backend()
                    loaded_models[filename] = model
                    logger.info(f"Model {filename} loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model {filename}: {e}")

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
    model_id: str


@app.get("/models")
async def get_models():
    return {"status": "success", "models": list(loaded_models.keys())}


@app.post("/ingest")
async def ingest_data(payload: SensorPayload):
    raw_data = np.array(payload.buffer, dtype=np.float32)

    if payload.model_id not in loaded_models:
        return {"status": "error", "message": f"Model '{payload.model_id}' not found on server."}

    use_model = loaded_models[payload.model_id]

    mask = None
    if payload.length is not None:
        mask = np.array([payload.length], dtype=np.int32)

    X = None
    if "magnitude" in payload.model_id.lower():
        acc_mag = np.sqrt(np.sum(raw_data[:, 0:3] ** 2, axis=-1, keepdims=True))
        gyro_mag = np.sqrt(np.sum(raw_data[:, 3:6] ** 2, axis=-1, keepdims=True))
        X_mag_raw = np.concatenate((acc_mag, gyro_mag), axis=-1)
        if hasattr(use_model, "scaler") and use_model.scaler is not None:
            X_mag_raw = use_model.scaler.transform(X_mag_raw)
        X = np.transpose(X_mag_raw, (1, 0))[np.newaxis, :, :]
    elif "hybrid" in payload.model_id.lower():
        acc_mag = np.sqrt(np.sum(raw_data[:, 0:3] ** 2, axis=-1, keepdims=True))
        gyro_mag = np.sqrt(np.sum(raw_data[:, 3:6] ** 2, axis=-1, keepdims=True))
        X_mag_raw = np.concatenate((raw_data, acc_mag, gyro_mag), axis=-1)
        if hasattr(use_model, "scaler") and use_model.scaler is not None:
            X_mag_raw = use_model.scaler.transform(X_mag_raw)
            X_mag_raw = np.clip(X_mag_raw, -4.0, 4.0)
        X = np.transpose(X_mag_raw, (1, 0))[np.newaxis, :, :]
    else:
        if hasattr(use_model, "scaler") and use_model.scaler is not None:
            X = use_model.scaler.transform(raw_data)
        else:
            X = raw_data
        X = np.transpose(X, (1, 0))[np.newaxis, :, :]

    predictions = use_model.predict(X, training=False, mask=mask)

    preds = predictions[0]
    current_labels = getattr(use_model, "activity_labels", None)
    if not current_labels:
        current_labels = [f"Class_{i}" for i in range(len(preds))]
    top_3_idx = np.argsort(preds)[-3:][::-1]
    top_3 = [
        {"activity": current_labels[i], "confidence": float(preds[i])} for i in top_3_idx if i < len(current_labels)
    ]
    idx = int(top_3_idx[0])
    label = current_labels[idx]
    confidence = float(preds[idx])

    valid_len = payload.length if payload.length is not None else len(payload.buffer)
    chart_data = payload.buffer[max(0, valid_len - 150) : valid_len]

    response_payload = {
        "user": payload.user,
        "activity": label,
        "confidence": confidence,
        "top_3": top_3,
        "chart_data": chart_data,
        "model_id": payload.model_id,
    }

    await manager.broadcast(response_payload)
    return {"status": "success", **response_payload}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

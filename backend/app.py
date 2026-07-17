"""
Health Monitor API — FastAPI Backend
Deployable on Hugging Face Spaces (free CPU tier: 2 vCPU, 16 GB RAM)
Auto-detected by HF Spaces when app_file: backend/app.py is set in README.md
"""
import sys
import os
import json
import time
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ──
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent if _HERE.name == "backend" else _HERE
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import MODELS_DIR, THRESHOLDS, WINDOW_CONFIG, SAMPLING_RATES, CONDITIONS
from src.inference import Predictor, TemporalSmoother, ResponseBuilder
from data.synthetic.generator import SyntheticDataGenerator

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

# ── FastAPI App ──
app = FastAPI(
    title="Health Monitor API",
    version="2.0.0",
    description="Real-time health monitoring inference API — detects 7+ conditions from wearable sensor data",
    contact={"name": "Health Monitor", "url": "https://github.com/XORNOTFOUND02/health-monitor"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──
predictor: Optional[Predictor] = None
smoother: Optional[TemporalSmoother] = None
builder: Optional[ResponseBuilder] = None
generator: Optional[SyntheticDataGenerator] = None

alert_history: list[dict] = []
MAX_ALERT_HISTORY = 200
inference_times: list[float] = []
startup_time: float = 0.0
ws_connections: set[WebSocket] = set()


# ── Pydantic Models ──
class SensorData(BaseModel):
    """Single window of sensor data for inference."""
    accelerometer: list[list[float]] | dict[str, list[float]]
    gyroscope: list[list[float]] | dict[str, list[float]]
    heart_rate: list[float]
    spo2: Optional[list[float]] = None
    temperature: list[float]
    ppg: Optional[list[float]] = None
    metadata: Optional[dict[str, Any]] = None


class ThresholdUpdate(BaseModel):
    condition: str
    threshold: str
    value: float


# ═══════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    global predictor, smoother, builder, generator, startup_time
    startup_time = time.time()
    models_path = str(_PROJECT_ROOT / "models")
    log.info("Loading models from %s", models_path)
    try:
        predictor = Predictor(models_dir=models_path)
        smoother = TemporalSmoother(window_buffer_size=5, min_detections=3, cooldown_seconds=60)
        builder = ResponseBuilder()
        generator = SyntheticDataGenerator(seed=42)
        log.info("✅ Backend ready — %d conditions available", len(CONDITIONS))
    except Exception as e:
        log.error("❌ Failed to load models: %s", e)
        raise


# ═══════════════════════════════════════════════════════════════
# REST Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "message": "Health Monitor API",
        "docs": "/docs",
        "version": "2.0.0",
        "status": "running" if predictor and predictor.is_loaded else "loading",
    }


@app.get("/health")
@app.get("/api/v1/health")
async def health():
    """Health check with model versions and inference stats."""
    avg_ms = round(float(np.mean(inference_times[-100:])), 1) if inference_times else 0
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_versions": predictor.model_versions if predictor else {},
        "conditions": CONDITIONS,
        "uptime_seconds": round(time.time() - startup_time, 1),
        "total_inferences": len(inference_times),
        "avg_inference_ms": avg_ms,
        "models_loaded": bool(predictor and predictor.is_loaded),
    }


@app.get("/conditions")
@app.get("/api/v1/conditions")
async def list_conditions():
    """List all detectable conditions with thresholds."""
    thresholds = {}
    for field in dir(THRESHOLDS):
        if field.isupper() and not field.startswith("_"):
            thresholds[field.lower()] = getattr(THRESHOLDS, field)

    return {
        "conditions": [
            {
                "id": c,
                "name": c.replace("_", " ").title(),
                "severity": (
                    "critical" if c in ("low_spo2", "fall_detected")
                    else "warning" if c in ("tachycardia", "irregular_rhythm", "fever")
                    else "info"
                ),
                "detection_method": "ML" if c != "fever" else "rule",
            }
            for c in CONDITIONS
        ],
        "thresholds": thresholds,
        "window_config": {
            "duration_sec": WINDOW_CONFIG.WINDOW_DURATION_SEC,
            "stride_sec": WINDOW_CONFIG.STRIDE_SEC,
            "sampling_rates": {
                "accelerometer": SAMPLING_RATES.ACCEL,
                "gyroscope": SAMPLING_RATES.GYRO,
                "heart_rate": SAMPLING_RATES.HR,
                "spo2": SAMPLING_RATES.SPO2,
                "temperature": SAMPLING_RATES.TEMP,
            },
        },
    }


@app.post("/predict")
@app.post("/api/v1/predict")
async def predict(data: SensorData):
    """Run ML inference on a 30-second window of sensor data."""
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    start_t = time.perf_counter()
    try:
        raw = data.model_dump(exclude_none=True)
        
        # 1. Raw ML predictions
        raw_predictions = predictor.predict(raw)
        
        # 2. Data quality
        data_quality = predictor._compute_data_quality(raw)
        
        # 3. Temporal smoothing (N-of-M voting)
        smoothed = smoother.update(raw_predictions, timestamp=time.time())
        
        # 4. Build response envelope
        metadata = {"model_versions": predictor.model_versions}
        final = builder.build_response(smoothed, data_quality, metadata)
        
        elapsed_ms = (time.perf_counter() - start_t) * 1000
        inference_times.append(elapsed_ms)
        final["processing_time_ms"] = round(elapsed_ms, 1)

        # Store alerts & broadcast
        _store_alerts(final)
        asyncio.create_task(_broadcast(final))

        return final

    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/demo/generate")
@app.get("/api/v1/demo/generate")
async def generate_demo(condition: str = "normal", severity: int = 1):
    """Generate synthetic sensor data for demo/testing.
    
    condition: normal, tachycardia, bradycardia, hypoxia, hyperventilation,
               atrial_fibrillation, bradycardia_hypoxia, fatigue
    severity: 1 (mild), 2 (moderate), 3 (severe)
    """
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    sev_map = {1: "mild", 2: "moderate", 3: "severe"}
    sev = sev_map.get(severity, "mild")

    cond_map = {
        "normal": None,
        "tachycardia": "tachycardia",
        "bradycardia": "bradycardia",
        "hypoxia": "low_spo2",
        "hyperventilation": "hyperventilation",
        "atrial_fibrillation": "irregular_rhythm",
        "bradycardia_hypoxia": "bradycardia_hypoxia",
        "fatigue": "fatigue",
    }
    gen_cond = cond_map.get(condition)

    try:
        if gen_cond is None:
            session = generator.generate_normal_session(duration_sec=30)
        else:
            session = generator.generate_condition_session(gen_cond, duration_sec=30, severity=sev)

        windows = session.get("windows", [])
        if not windows:
            raise HTTPException(status_code=500, detail="No windows generated")
        
        sensor_data = windows[0].get("sensor_data", {})
        return {"sensor_data": sensor_data, "condition": condition, "severity": sev}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/history")
@app.get("/api/v1/history")
async def get_history(limit: int = 50, severity: Optional[str] = None):
    """Get recent alert history."""
    filtered = alert_history
    if severity and severity in ("critical", "warning", "info"):
        filtered = [a for a in filtered if a.get("severity") == severity]
    return {"alerts": filtered[-limit:], "total": len(filtered), "filter": severity or "all"}


@app.get("/config/thresholds")
@app.get("/api/v1/config/thresholds")
async def get_thresholds():
    """Get current clinical thresholds."""
    thresholds = {}
    for field in dir(THRESHOLDS):
        if field.isupper() and not field.startswith("_"):
            thresholds[field.lower()] = getattr(THRESHOLDS, field)
    return {"thresholds": thresholds}


@app.put("/config/thresholds")
@app.put("/api/v1/config/thresholds")
async def update_thresholds(update: ThresholdUpdate):
    """Update a clinical threshold (in-memory only)."""
    field = f"{update.condition.upper()}_{update.threshold.upper()}"
    if hasattr(THRESHOLDS, field):
        setattr(THRESHOLDS, field, update.value)
        log.info("Threshold %s → %s", field, update.value)
        return {"status": "updated", "field": field, "value": update.value}
    raise HTTPException(status_code=404, detail=f"Unknown threshold. Use /conditions to see valid names.")


# ═══════════════════════════════════════════════════════════════
# WebSocket — Real-time streaming inference
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws")
@app.websocket("/api/v1/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_connections.add(ws)
    log.info("🟢 WebSocket connected (%d total)", len(ws_connections))
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await ws.send_json({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            if msg.get("type") == "predict" and predictor and predictor.is_loaded:
                sensor = msg.get("data", {})
                start_t = time.perf_counter()
                try:
                    raw_preds = predictor.predict(sensor)
                    quality = predictor._compute_data_quality(sensor)
                    smoothed = smoother.update(raw_preds, timestamp=time.time())
                    final = builder.build_response(smoothed, quality)
                    elapsed_ms = (time.perf_counter() - start_t) * 1000
                    final["processing_time_ms"] = round(elapsed_ms, 1)
                    _store_alerts(final)

                    await ws.send_json({"type": "prediction", "data": final})
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})

            if msg.get("type") == "subscribe_alerts":
                await ws.send_json({
                    "type": "alert_history",
                    "data": alert_history[-50:],
                })

    except WebSocketDisconnect:
        ws_connections.discard(ws)
        log.info("🔴 WebSocket disconnected (%d remaining)", len(ws_connections))
    except Exception as e:
        ws_connections.discard(ws)
        log.error("WebSocket error: %s", e)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _store_alerts(response: dict):
    """Store alerts from a prediction response."""
    severity_map = {}
    for sev_key in ("critical_alerts", "warnings", "info"):
        conds = response.get("alert_summary", {}).get(sev_key, [])
        alert_sev = sev_key.replace("_alerts", "").replace("s", "") or "info"
        for c in conds:
            severity_map[c] = alert_sev

    for condition, info in response.get("predictions", {}).items():
        if info.get("detected"):
            alert_history.append({
                "condition": condition,
                "name": condition.replace("_", " ").title(),
                "severity": severity_map.get(condition, "info"),
                "probability": round(float(info.get("probability", 0)), 3),
                "confidence": round(float(info.get("confidence", 0)), 3),
                "timestamp": response.get("timestamp", datetime.now(timezone.utc).isoformat()),
            })

    while len(alert_history) > MAX_ALERT_HISTORY:
        alert_history.pop(0)


async def _broadcast(data: dict):
    """Broadcast prediction to all WebSocket clients."""
    dead: set[WebSocket] = set()
    for ws in ws_connections:
        try:
            await ws.send_json({"type": "prediction", "data": data})
        except Exception:
            dead.add(ws)
    ws_connections -= dead


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    log.info("Starting server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from collections import deque
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# Initialize FastAPI application
app = FastAPI(
    title="SmartMill ESP32 Predictive Maintenance Server",
    description="Real-time ML inference ingestion API for ESP32 sensor telemetry",
    version="2.0.0"
)

# Enable CORS for frontends / local dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory circular buffer for recent telemetry (last 100 entries)
TELEMETRY_LOG_FILE = "live_telemetry.json"
telemetry_buffer = deque(maxlen=100)

# Load existing telemetry if available
if os.path.exists(TELEMETRY_LOG_FILE):
    try:
        with open(TELEMETRY_LOG_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                telemetry_buffer.extend(data[-100:])
    except Exception:
        pass

# Load ML artifacts
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
MODES_PATH = os.path.join(MODEL_DIR, "failure_mode_models.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics_summary.json")

best_model = None
scaler = None
mode_models = None
feature_columns = []

try:
    best_model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    mode_models = joblib.load(MODES_PATH)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        feature_columns = metrics.get("feature_columns", [])
    print(f"Loaded ML model: {metrics.get('best_model_name', 'Unknown')}")
    print(f"Feature columns: {feature_columns}")
except Exception as e:
    print(f"Warning: Model artifacts could not be fully loaded ({e}). Run train_model.py first.")

# Pydantic schema for incoming ESP32 JSON payload
class SensorPayload(BaseModel):
    temperature: Optional[float] = None
    vibration: Optional[int] = 0
    voltage: Optional[int] = 1966
    current: Optional[float] = None
    power: Optional[float] = None
    rpm: Optional[float] = None

class PredictionResponse(BaseModel):
    status: str
    machine_health: str
    failure_probability: float
    detected_fault: str
    alert: bool
    voltage_volts: float
    temperature_c: float
    timestamp: str

def compute_features(temperature: Optional[float], vibration: int, voltage_adc: int):
    # Impute missing temperature if sensor is disconnected
    temp_val = 25.0 if temperature is None or np.isnan(temperature) else float(temperature)
    vib_val = 1 if vibration else 0
    volt_adc = 1966 if voltage_adc is None else int(voltage_adc)
    
    # 0-25V module calculation
    voltage_volts = round((volt_adc / 4095.0) * 25.0, 2)
    voltage_dev = round(abs(voltage_volts - 12.0), 3)
    temp_anomaly = round(temp_val - 25.0, 2)
    temp_excess = max(0.0, temp_val - 50.0)
    stress_index = round((vib_val * 2.5) + (voltage_dev * 1.5) + (temp_excess * 0.1), 3)
    
    df_raw = pd.DataFrame([{
        'temperature': temp_val,
        'vibration': vib_val,
        'voltage': volt_adc,
        'voltage_volts': voltage_volts,
        'voltage_dev': voltage_dev,
        'temp_anomaly': temp_anomaly,
        'stress_index': stress_index
    }])
    
    return df_raw, voltage_volts, temp_val

@app.get("/")
def root():
    return {
        "service": "SmartMill ESP32 Predictive Maintenance Server",
        "status": "online",
        "models_loaded": best_model is not None,
        "features": feature_columns,
        "endpoint": "POST /sensor-data"
    }

@app.post("/sensor-data", response_model=PredictionResponse)
def receive_sensor_data(payload: SensorPayload):
    """
    Receives live sensor readings from the ESP32, computes physics features,
    runs ML inference, logs telemetry, and returns risk assessment.
    """
    if best_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML Models not initialized. Please run train_model.py first.")

    # 1. Engineer features
    df_features, v_volts, t_c = compute_features(
        temperature=payload.temperature,
        vibration=payload.vibration or 0,
        voltage_adc=payload.voltage or 1966
    )

    # 2. Scale features and run inference
    X_scaled = scaler.transform(df_features[feature_columns])
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    
    prob = float(best_model.predict_proba(X_scaled_df)[0][1])
    
    # 3. Diagnose specific failure mode if risk is elevated
    detected_fault = "Normal"
    if prob >= 0.25 and mode_models:
        mode_probs = {}
        for mode_name, clf in mode_models.items():
            mode_probs[mode_name] = float(clf.predict_proba(X_scaled_df)[0][1])
        top_mode = max(mode_probs, key=mode_probs.get)
        if mode_probs[top_mode] >= 0.35:
            mode_labels = {
                'VAF': 'Vibration Anomaly (Bearing/Loose)',
                'OHF': 'Overheating Fault',
                'VLF': 'Voltage Sag/Surge Fault',
                'CSF': 'Combined Stress Failure'
            }
            detected_fault = mode_labels.get(top_mode, top_mode)
    
    # Health category
    if prob < 0.25:
        health_status = "NORMAL"
        alert = False
    elif prob < 0.60:
        health_status = "WARNING"
        alert = False
    else:
        health_status = "CRITICAL"
        alert = True

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 4. Record to telemetry buffer
    record = {
        "timestamp": timestamp_str,
        "temperature": t_c,
        "vibration": payload.vibration or 0,
        "voltage_adc": payload.voltage or 1966,
        "voltage_volts": v_volts,
        "failure_probability": round(prob, 4),
        "machine_health": health_status,
        "detected_fault": detected_fault,
        "alert": alert
    }
    telemetry_buffer.append(record)
    
    # Persist latest records
    try:
        with open(TELEMETRY_LOG_FILE, "w") as f:
            json.dump(list(telemetry_buffer), f, indent=2)
    except Exception:
        pass

    print(f"[{timestamp_str}] ESP32 -> Temp: {t_c:.1f}°C | Vib: {payload.vibration} | Volt: {v_volts:.2f}V | Health: {health_status} (Risk: {prob*100:.1f}%)")

    return PredictionResponse(
        status="success",
        machine_health=health_status,
        failure_probability=round(prob, 4),
        detected_fault=detected_fault,
        alert=alert,
        voltage_volts=v_volts,
        temperature_c=t_c,
        timestamp=timestamp_str
    )

@app.get("/live-telemetry")
def get_live_telemetry():
    """Returns the most recent telemetry entries."""
    return list(telemetry_buffer)

@app.post("/clear-telemetry")
def clear_telemetry():
    telemetry_buffer.clear()
    if os.path.exists(TELEMETRY_LOG_FILE):
        try:
            os.remove(TELEMETRY_LOG_FILE)
        except Exception:
            pass
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)

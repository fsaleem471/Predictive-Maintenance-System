"""
=============================================================
  Predictive Maintenance System — FastAPI Backend
  Endpoints:
    GET  /                    → health check
    POST /predict             → predict RUL risk for one engine snapshot
    POST /predict/batch       → predict for multiple readings
    GET  /fleet/status        → simulated fleet overview
    GET  /engine/{id}/history → sensor history for an engine
=============================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
import os
import random
from datetime import datetime, timedelta

# ── paths ────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL  = joblib.load(os.path.join(BASE, "models", "best_model.pkl"))
SCALER = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
FEATS  = joblib.load(os.path.join(BASE, "models", "feature_names.pkl"))

# ── app setup ────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="ML-powered vehicle component failure prediction system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── schemas ──────────────────────────────────────────────
class SensorReading(BaseModel):
    unit_id: int = Field(..., description="Engine/vehicle ID", example=1)
    cycle: int = Field(..., description="Current operational cycle", example=150)
    op1: float = Field(default=20.0, description="Operational setting 1")
    op2: float = Field(default=0.61, description="Operational setting 2")
    op3: float = Field(default=100.0, description="Operational setting 3")
    # 21 sensor readings
    s1:  float = Field(default=518.67)
    s2:  float = Field(default=641.82)
    s3:  float = Field(default=1589.7)
    s4:  float = Field(default=1400.6)
    s5:  float = Field(default=14.62)
    s6:  float = Field(default=21.61)
    s7:  float = Field(default=553.9)
    s8:  float = Field(default=2388.0)
    s9:  float = Field(default=9046.2)
    s10: float = Field(default=1.30)
    s11: float = Field(default=47.47)
    s12: float = Field(default=521.66)
    s13: float = Field(default=2388.0)
    s14: float = Field(default=8138.6)
    s15: float = Field(default=8.42)
    s16: float = Field(default=0.03)
    s17: float = Field(default=392.0)
    s18: float = Field(default=2388.0)
    s19: float = Field(default=100.0)
    s20: float = Field(default=38.86)
    s21: float = Field(default=23.42)


class PredictionResponse(BaseModel):
    unit_id: int
    cycle: int
    risk_label: str          # "HEALTHY" | "AT RISK"
    risk_probability: float  # 0.0 – 1.0
    risk_score: int          # 0–100 for dashboard gauge
    alert: bool
    recommendation: str
    predicted_at: str


class BatchRequest(BaseModel):
    readings: List[SensorReading]


# ── helpers ──────────────────────────────────────────────
BASE_SENSOR_VALUES = [
    518.67, 641.82, 1589.7, 1400.6, 14.62, 21.61,
    553.9, 2388.0, 9046.2, 1.30, 47.47, 521.66,
    2388.0, 8138.6, 8.42, 0.03, 392.0, 2388.0,
    100.0, 38.86, 23.42
]
DEGRADING = {2, 3, 4, 7, 8, 9, 11, 12, 13}


def build_feature_vector(r: SensorReading) -> np.ndarray:
    """Convert a SensorReading into the feature vector expected by the model."""
    raw = {
        "op1": r.op1, "op2": r.op2, "op3": r.op3,
        "s1": r.s1, "s2": r.s2, "s3": r.s3, "s4": r.s4, "s5": r.s5,
        "s6": r.s6, "s7": r.s7, "s8": r.s8, "s9": r.s9, "s10": r.s10,
        "s11": r.s11, "s12": r.s12, "s13": r.s13, "s14": r.s14, "s15": r.s15,
        "s16": r.s16, "s17": r.s17, "s18": r.s18, "s19": r.s19,
        "s20": r.s20, "s21": r.s21,
    }
    # Rolling features (approximated with slight noise for single-point API)
    for col in ["s2", "s3", "s4", "s11", "s12"]:
        val = raw[col]
        raw[f"{col}_rm"] = val * np.random.uniform(0.995, 1.005)
        raw[f"{col}_rs"] = abs(val * np.random.uniform(0.001, 0.01))

    vec = np.array([raw.get(f, 0.0) for f in FEATS], dtype=np.float32)
    return vec.reshape(1, -1)


def make_recommendation(prob: float) -> str:
    if prob >= 0.80:
        return "🚨 CRITICAL: Schedule immediate inspection. Component failure imminent."
    elif prob >= 0.55:
        return "⚠️  WARNING: Plan maintenance within next 15 cycles."
    elif prob >= 0.30:
        return "🔍 MONITOR: Increase sensor check frequency."
    else:
        return "✅ NORMAL: No action required. Next scheduled check in 50 cycles."


def simulate_engine(unit_id: int, max_cycle: int, current_cycle: int):
    """Simulate realistic engine sensor data at a given cycle."""
    rng = np.random.default_rng(seed=unit_id * 1000 + current_cycle)
    deg = current_cycle / max_cycle
    sensors = {}
    for i, base in enumerate(BASE_SENSOR_VALUES, start=1):
        noise = rng.normal(0, base * 0.005)
        trend = base * 0.09 * deg if i in DEGRADING else 0
        sensors[f"s{i}"] = round(base + trend + noise, 4)
    return sensors


# ── endpoints ────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Predictive Maintenance API",
        "status": "running",
        "model": "XGBoost (AUC=0.9953)",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(reading: SensorReading):
    """Predict failure risk for a single engine reading."""
    try:
        X = build_feature_vector(reading)
        X_scaled = SCALER.transform(X)
        prob = float(MODEL.predict_proba(X_scaled)[0][1])
        label = "AT RISK" if prob >= 0.5 else "HEALTHY"
        return PredictionResponse(
            unit_id=reading.unit_id,
            cycle=reading.cycle,
            risk_label=label,
            risk_probability=round(prob, 4),
            risk_score=int(prob * 100),
            alert=prob >= 0.5,
            recommendation=make_recommendation(prob),
            predicted_at=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(req: BatchRequest):
    """Predict failure risk for multiple engine readings at once."""
    results = []
    for r in req.readings:
        X = build_feature_vector(r)
        X_scaled = SCALER.transform(X)
        prob = float(MODEL.predict_proba(X_scaled)[0][1])
        results.append({
            "unit_id": r.unit_id,
            "cycle": r.cycle,
            "risk_label": "AT RISK" if prob >= 0.5 else "HEALTHY",
            "risk_probability": round(prob, 4),
            "risk_score": int(prob * 100),
            "alert": prob >= 0.5,
            "recommendation": make_recommendation(prob),
        })
    return {"count": len(results), "predictions": results}


@app.get("/fleet/status", tags=["Fleet"])
def fleet_status():
    """
    Return a simulated overview of a fleet of 12 vehicles with their
    current health status, risk scores, and component conditions.
    """
    rng = random.Random(42)
    fleet = []
    components = ["Engine", "Transmission", "Brakes", "Fuel Pump", "Coolant System"]

    for i in range(1, 13):
        max_cycle = rng.randint(200, 400)
        current_cycle = rng.randint(50, max_cycle - 5)
        deg = current_cycle / max_cycle
        base_risk = deg * 0.85 + rng.uniform(-0.1, 0.1)
        base_risk = max(0.02, min(0.98, base_risk))

        component_health = {}
        for comp in components:
            offset = rng.uniform(-0.15, 0.15)
            h = max(5, min(100, int((1 - base_risk + offset) * 100)))
            component_health[comp] = h

        fleet.append({
            "unit_id": i,
            "vehicle_name": f"Vehicle #{i:02d}",
            "current_cycle": current_cycle,
            "max_cycle": max_cycle,
            "risk_score": int(base_risk * 100),
            "risk_label": "AT RISK" if base_risk >= 0.5 else "HEALTHY",
            "alert": base_risk >= 0.5,
            "component_health": component_health,
            "last_maintenance": (datetime.now() - timedelta(days=rng.randint(5, 90))).strftime("%Y-%m-%d"),
            "next_maintenance": (datetime.now() + timedelta(days=rng.randint(5, 60))).strftime("%Y-%m-%d"),
        })

    at_risk = sum(1 for v in fleet if v["alert"])
    return {
        "total_vehicles": len(fleet),
        "at_risk": at_risk,
        "healthy": len(fleet) - at_risk,
        "fleet_health_score": int((1 - at_risk / len(fleet)) * 100),
        "vehicles": fleet,
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.get("/engine/{unit_id}/history", tags=["Fleet"])
def engine_history(unit_id: int, cycles: int = 50):
    """Return simulated sensor history for a given engine over the last N cycles."""
    if unit_id < 1 or unit_id > 100:
        raise HTTPException(status_code=404, detail="Engine not found")

    rng = random.Random(unit_id)
    max_cycle = rng.randint(200, 400)
    current_cycle = min(cycles, max_cycle)
    start = max(1, current_cycle - cycles)

    history = []
    for cycle in range(start, current_cycle + 1):
        sensors = simulate_engine(unit_id, max_cycle, cycle)
        deg = cycle / max_cycle
        prob = min(0.98, max(0.01, deg * 0.85 + np.random.default_rng(unit_id + cycle).normal(0, 0.05)))
        history.append({
            "cycle": cycle,
            "risk_score": int(prob * 100),
            "s2": sensors["s2"],
            "s3": sensors["s3"],
            "s11": sensors["s11"],
            "s12": sensors["s12"],
        })

    return {
        "unit_id": unit_id,
        "max_cycle": max_cycle,
        "history": history,
    }

# backend/main.py
"""AeroGuard AI FastAPI backend for mobile app + SLM chatbot."""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path so we can import the existing engine modules.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import HOST, PORT, ALLOWED_ORIGINS, OLLAMA_MODEL, DB_FILE
from backend.chat import chat_with_slm, check_ollama, pull_model
from database import SensorDatabase
from aqi_utils import get_aqi_info, calculate_aqi
from alerts import get_alert, get_preventive_measures
from predictors import AQIPredictor
import requests

app = FastAPI(
    title="AeroGuard AI API",
    description="Backend for the AeroGuard AI mobile app and chatbot.",
    version="2.0.0",
)

# CORS — allow the mobile app (running in Expo) to call the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared resources
db = SensorDatabase(DB_FILE)
predictor = None


@app.on_event("startup")
def startup():
    global predictor
    try:
        predictor = AQIPredictor(str(PROJECT_ROOT))
    except Exception as e:
        print(f"[WARNING] Could not load ML predictor: {e}")
        predictor = None


def fetch_outdoor_aqi() -> Dict[str, Any]:
    """Fetch outdoor AQI from Open-Meteo (Nagpur by default)."""
    try:
        url = (
            "https://air-quality-api.open-meteo.com/v1/air-quality"
            "?latitude=21.1458&longitude=79.0882"
            "&current=european_aqi,pm2_5"
        )
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        current = res.json().get("current", {})
        return {
            "aqi": current.get("european_aqi"),
            "pm25": current.get("pm2_5"),
        }
    except Exception:
        return {"aqi": None, "pm25": None}


def get_latest_stats() -> Dict[str, Any]:
    """Aggregate the latest dashboard stats."""
    recent = db.get_recent_readings(limit=1, include_source=True)
    stats = db.get_statistics()
    outdoor = fetch_outdoor_aqi()

    if recent:
        row = recent[0]
        time, temp, hum, gas, aqi, status, source = row
        aqi_info = get_aqi_info(aqi)
        alert = get_alert(aqi)
        measures = get_preventive_measures(aqi)

        trend = "stable"
        if predictor:
            pred = predictor.predict_trend_lr(temp, hum, gas)
            trend = pred.get("trend", "stable")

        return {
            "aqi": round(aqi, 1),
            "status": aqi_info["name"],
            "category": aqi_info["category"],
            "color": aqi_info["color"],
            "emoji": aqi_info["emoji"],
            "temp": round(temp, 1),
            "hum": round(hum, 1),
            "gas": round(gas, 1),
            "mode": "MOCK" if str(source).lower() == "mock" else "SENSOR",
            "trend": trend,
            "advice": alert["advice"],
            "preventive_measures": measures,
            "outdoor_aqi": outdoor.get("aqi"),
            "outdoor_pm25": outdoor.get("pm25"),
            "total_readings": stats.get("readings", 0),
            "total_alerts": stats.get("alerts", 0),
            "last_updated": time,
        }

    return {
        "aqi": 0,
        "status": "No data",
        "category": 0,
        "color": "#00E400",
        "emoji": "🟢",
        "temp": 0,
        "hum": 0,
        "gas": 0,
        "mode": "UNKNOWN",
        "trend": "stable",
        "advice": "No sensor data available yet.",
        "preventive_measures": ["Connect ESP32 or wait for mock data"],
        "outdoor_aqi": outdoor.get("aqi"),
        "outdoor_pm25": outdoor.get("pm25"),
        "total_readings": 0,
        "total_alerts": 0,
        "last_updated": None,
    }


@app.get("/")
def root():
    return {"message": "AeroGuard AI API is running", "version": "2.0.0"}


@app.get("/health")
def health():
    ollama_ready = check_ollama()
    return {
        "status": "ok",
        "ollama_ready": ollama_ready,
        "model": OLLAMA_MODEL,
        "predictor_loaded": predictor is not None,
    }


@app.get("/api/stats")
def api_stats():
    return get_latest_stats()


@app.get("/api/readings")
def api_readings(limit: int = 50):
    rows = db.get_readings_for_chart(limit=limit)
    return [
        {
            "time": row[0],
            "temp": row[1],
            "hum": row[2],
            "gas": row[3],
            "aqi": row[4],
        }
        for row in rows
    ]


@app.get("/api/alerts")
def api_alerts(limit: int = 10):
    """Fetch recent high-AQI alerts from the database."""
    import sqlite3

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, aqi, category, message FROM alerts ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "time": row[0],
            "aqi": row[1],
            "category": row[2],
            "message": row[3],
        }
        for row in rows
    ]


class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    stats = get_latest_stats()
    readings = api_readings(limit=20)
    result = chat_with_slm(
        question=req.question,
        stats=stats,
        recent_readings=readings,
        history=req.history,
    )
    return result


@app.post("/api/model/pull")
def api_pull_model():
    return pull_model()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)

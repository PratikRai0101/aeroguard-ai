# backend/chat.py
"""Chatbot logic that injects dashboard context into a local SLM."""

import json
import requests
from typing import List, Dict, Any
from backend.config import OLLAMA_HOST, OLLAMA_MODEL, MAX_CHAT_HISTORY


SYSTEM_PROMPT = """You are AeroGuard AI, a helpful air-quality assistant.
You are answering questions about a real-time IoT air-quality monitoring system.

Use ONLY the data provided below to answer the user's question.
If the data is not enough to answer, say so and suggest what the user should check.
Keep answers concise, clear, and helpful. Use Celsius for temperature, percent for humidity, and AQI for air quality.

Role:
- The ML models and the risk module produce the prediction, the reliability measure and the risk level. Your job is to explain those results in plain language and answer questions about them.
- Never invent a different prediction, reliability value or risk level than the one in the context below.
- The risk level is an environmental indicator, not a medical diagnosis. Say so if the user asks about disease or personal health.
- When asked "why" a prediction or risk level was produced, use the SHAP contributing factors in the context.
- The system's air-quality value is "Predicted AQI (ML model)". The "sensor-estimate AQI" is a cruder raw figure. If both appear, use the predicted AQI as the answer.
- When you quote any number, copy it exactly from the context. Do not round, average or invent numbers.

Scope rules:
- Answer only questions about air quality, AQI, sensors, readings, trends, alerts, outdoor conditions, or health guidance related to these readings.
- If the user asks for unrelated content such as programming code, general knowledge, jokes, or creative writing, do not answer that request. Say: "I can only help with AeroGuard air-quality data and related health guidance."
- Do not provide Python code or instructions for unrelated tasks.

Current dashboard context:
{context}
"""


def build_context(stats: Dict[str, Any], recent_readings: List[Dict[str, Any]]) -> str:
    """Build a natural-language context block from the latest dashboard data."""
    lines = []
    lines.append(f"- Predicted AQI (ML model): {stats.get('aqi', 'N/A')} ({stats.get('status', 'N/A')})")
    if stats.get('sensor_aqi') is not None:
        lines.append(
            f"- Sensor-estimate AQI (raw gas formula, cruder): {stats.get('sensor_aqi')}"
        )
    lines.append(f"- Temperature: {stats.get('temp', 'N/A')} °C")
    lines.append(f"- Humidity: {stats.get('hum', 'N/A')} %")
    lines.append(f"- Gas/VOC: {stats.get('gas', 'N/A')}")
    lines.append(f"- Mode: {stats.get('mode', 'N/A')}")
    lines.append(f"- Trend: {stats.get('trend', 'N/A')}")
    lines.append(f"- Outdoor AQI: {stats.get('outdoor_aqi', 'N/A')}")
    lines.append(f"- Total readings stored: {stats.get('total_readings', 'N/A')}")
    lines.append(f"- Total alerts: {stats.get('total_alerts', 'N/A')}")

    prediction = stats.get('prediction') or {}
    if prediction:
        confidence = prediction.get('confidence')
        confidence_text = f"{confidence:.0f}%" if isinstance(confidence, (int, float)) else "N/A"
        lines.append(
            f"- ML predicted category: {prediction.get('label', 'N/A')} "
            f"(class probability {confidence_text})"
        )
    if stats.get('predicted_aqi') is not None:
        lines.append(f"- Predicted AQI (trend model): {stats.get('predicted_aqi')}")

    reliability = stats.get('reliability') or {}
    if reliability:
        interval = reliability.get('interval') or {}
        lines.append(
            f"- Prediction reliability: {reliability.get('label', 'N/A')} "
            f"(category probability {reliability.get('category_probability', 'N/A')}; "
            f"80% interval {interval.get('low', '?')}-{interval.get('high', '?')})"
        )

    risk = stats.get('risk') or {}
    if risk:
        lines.append(
            f"- Airborne disease risk: {risk.get('level', 'N/A')} "
            f"(score {risk.get('score', 'N/A')})"
        )
        for factor in risk.get('factors', [])[:4]:
            lines.append(f"    * {factor}")
        lines.append(f"- Risk advisory: {risk.get('advisory', 'N/A')}")
        lines.append(f"- Note: {risk.get('disclaimer', '')}")

    explanation = stats.get('explanation') or {}
    if explanation.get('top_factors'):
        lines.append("- Main contributing factors to the prediction (SHAP):")
        for factor in explanation['top_factors'][:3]:
            lines.append(
                f"    * {factor.get('description', factor.get('feature'))} = "
                f"{factor.get('value')}{factor.get('unit', '')} "
                f"({factor.get('direction')}, contribution {factor.get('contribution')})"
            )

    if recent_readings:
        lines.append("- Last few readings (most recent last):")
        for r in recent_readings[-5:]:
            lines.append(
                f"  {r.get('time', 'N/A')}: sensor-estimate AQI={r.get('aqi', 'N/A')}, "
                f"T={r.get('temp', 'N/A')}°C, H={r.get('hum', 'N/A')}%, G={r.get('gas', 'N/A')}"
            )

    if stats.get('alerts'):
        lines.append("- Recent alerts:")
        for alert in stats.get('alerts', [])[:3]:
            lines.append(f"  {alert.get('message', 'N/A')}")

    return "\n".join(lines)


def check_ollama() -> bool:
    """Check if Ollama is reachable and the target model is available."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        data = r.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        return any(OLLAMA_MODEL in name or name in OLLAMA_MODEL for name in models)
    except Exception as e:
        print(f"[Ollama check failed] {e}")
        return False


def chat_with_slm(
    question: str,
    stats: Dict[str, Any],
    recent_readings: List[Dict[str, Any]],
    history: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Send a question to the local SLM with injected dashboard context."""
    history = history or []
    context = build_context(stats, recent_readings)
    system_msg = {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}

    # Keep only last N exchanges to fit context
    trimmed_history = history[-MAX_CHAT_HISTORY:]
    messages = [system_msg] + trimmed_history + [{"role": "user", "content": question}]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        # This is a concise dashboard assistant; hidden reasoning can consume
        # the entire output budget without leaving a visible answer.
        "think": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 512,
            "num_ctx": 4096,
        },
    }

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            answer = "⚠️ The AI assistant returned an empty response. Please try again."
        return {
            "answer": answer,
            "model": OLLAMA_MODEL,
            "context": context,
        }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "⚠️ The local AI assistant is not running. Please start Ollama and make sure the model is downloaded.",
            "model": OLLAMA_MODEL,
            "context": context,
        }
    except Exception as e:
        return {
            "answer": f"⚠️ Error talking to the AI assistant: {str(e)}",
            "model": OLLAMA_MODEL,
            "context": context,
        }


def pull_model() -> Dict[str, Any]:
    """Pull the configured SLM via Ollama."""
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"name": OLLAMA_MODEL, "stream": False},
            timeout=600,
        )
        r.raise_for_status()
        return {"status": "success", "detail": r.json(), "model": OLLAMA_MODEL}
    except Exception as e:
        return {"status": "error", "detail": str(e), "model": OLLAMA_MODEL}

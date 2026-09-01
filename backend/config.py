# backend/config.py
"""Backend configuration for AeroGuard AI API."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")

# API server
HOST = os.getenv("AEROGUARD_HOST", "0.0.0.0")
PORT = int(os.getenv("AEROGUARD_PORT", "8000"))

# Database (reuse the project-root database unless explicitly overridden)
DB_FILE = os.getenv("AEROGUARD_DB", str(PROJECT_ROOT / "aeroguard.db"))

# CORS
ALLOWED_ORIGINS = os.getenv("AEROGUARD_ORIGINS", "*").split(",")

# Chat settings
MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", "10"))

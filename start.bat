@echo off
REM AeroGuard AI - Windows one-click start
REM This script sets up the Python venv, Ollama, and the FastAPI backend.

echo Starting AeroGuard AI...

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH. Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

python setup_and_run.py %*

pause

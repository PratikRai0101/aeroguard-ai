#!/usr/bin/env bash
# AeroGuard AI - macOS/Linux one-click start
# This script sets up the Python venv, Ollama, and the FastAPI backend.

set -e

echo "Starting AeroGuard AI..."

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.11+ and rerun."
    exit 1
fi

python3 setup_and_run.py "$@"

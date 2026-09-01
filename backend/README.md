# AeroGuard AI Backend

FastAPI backend that exposes the existing AeroGuard AI engine to the React Native mobile app and serves a local Qwen 3.5 2B SLM chatbot.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check + Ollama status |
| GET | `/api/stats` | Current AQI, temp, humidity, gas, trend, outdoor AQI |
| GET | `/api/readings?limit=50` | Recent sensor readings |
| GET | `/api/alerts?limit=10` | Recent critical alerts |
| POST | `/api/chat` | Ask the SLM about dashboard data |
| POST | `/api/model/pull` | Pull the configured SLM |

## One-run setup (Windows / macOS / Linux)

From the project root, run the platform script:

```bash
# Windows
start.bat

# macOS / Linux
./start.sh
```

Or run the Python script directly:

```bash
python setup_and_run.py
```

This script will:
1. Create a Python virtual environment (`venv/`)
2. Install backend dependencies
3. Check for Ollama and install it if needed
4. Pull the Qwen 3.5 2B model
5. Start Ollama
6. Start the FastAPI backend on `http://localhost:8000`

## Manual setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

Make sure Ollama is installed and running:

```bash
# Pull the model
ollama pull qwen3.5:2b

# Start Ollama (it usually auto-starts)
ollama serve
```

Start the backend:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration

Set environment variables to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen3.5:2b` | SLM model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `AEROGUARD_PORT` | `8000` | Backend port |
| `AEROGUARD_HOST` | `0.0.0.0` | Backend host |
| `AEROGUARD_DB` | project-root `aeroguard.db` | Path to SQLite database |

## Chat system prompt

The `/api/chat` endpoint injects the latest dashboard context into the model's system prompt:

- Current indoor AQI, temperature, humidity, gas
- Trend direction
- Outdoor AQI
- Recent readings
- Recent alerts

This ensures the assistant answers based on real data, not hallucinations.

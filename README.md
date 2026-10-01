# AeroGuard AI - Phase 2.0
## Complete Setup & User Guide

---

## 1. Project Overview

AeroGuard AI is a professional-grade **IoT Air Quality Monitoring System** with:

| Feature | Description |
|---------|-------------|
| **Sensors** | ESP32 + DHT22 (Temp, Humidity) + MQ-135 (Gas) |
| **ML Models** | Random Forest (classification), LSTM (predictions), Linear Regression (trend) |
| **AQI** | Real-time AQI calculation using India CPCB standards |
| **Health Alerts** | Category-based health recommendations |
| **Persistence** | SQLite database for all readings |
| **Auto-Detect** | Automatically detects ESP32 or falls back to mock mode |
| **Mobile App** | Cross-platform React Native app (iOS + Android) |
| **AI Chatbot** | Local Qwen 3.5 2B SLM that answers questions about your air-quality data |

---

## 2. Hardware Requirements

| Component | Details |
|-----------|---------|
| ESP32 | ESP32 Dev Module |
| DHT22 | Temperature & Humidity Sensor (GPIO 4) |
| MQ-135 | Gas/VOC Sensor (Analog pin) |
| USB Cable | For programming + power |

### Wiring Diagram

```
ESP32 GPIO 4 ────── DHT22 Data Pin
ESP32 3.3V ──────── DHT22 VCC
ESP32 GND ──────── DHT22 GND

MQ-135 A0 ──────── ESP32 ADC (GPIO 34 or any ADC)
MQ-135 VCC ───── ESP32 5V
MQ-135 GND ───── ESP32 GND
```

### ESP32 Code Format
The ESP32 should output data in this format:
```
temp,humidity,gas
```
Example: `25.3,55.2,185`

---

## 3. Python Environment Setup

### Quick Start (Copy-Paste)

```bash
# 1. Install pyenv (if not installed)
curl https://pyenv.run | bash

# 2. Install Python 3.10
pyenv install 3.10.13
pyenv local 3.10.13

# 3. Create virtual environment
~/.pyenv/versions/3.10.13/bin/python -m venv venv310

# 4. Activate (bash)
source venv310/bin/activate

# (OR) Activate (fish shell)
source venv310/bin/activate.fish

# 5. Install dependencies
pip install --upgrade pip
pip install numpy tensorflow scikit-learn joblib pandas requests pyserial
pip install streamlit plotly pyyaml
```

### Python Installation (Alternative)

If you don't want pyenv:
```bash
# Install Python 3.10+ from python.org, then:

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install numpy tensorflow scikit-learn joblib pandas requests pyserial
pip install streamlit plotly pyyaml
```

---

## 4. Running the Project

### Option A: Streamlit Dashboard (Recommended for Demo)

```bash
# Activate environment
source venv310/bin/activate  # (or venv/bin/activate)

# Run dashboard
streamlit run dashboard.py
```

Open browser at: **http://localhost:8501**

### Option B: CLI Dashboard

```bash
# Activate environment  
source venv310/bin/activate

# Run (auto-detects ESP32)
python dashboard_v2.py
```

---

## 5. Dashboard Values Explained

### Streamlit Dashboard Sidebar

| Value | Description |
|-------|-------------|
| **Mode** | 🔴 REAL SENSOR (ESP32 connected) or 🟡 MOCK MODE (using simulated data) |
| **Total Readings** | Number of sensor readings stored in database |
| **Total Alerts** | Number of health alerts generated |

### Main Dashboard Display

| Value | Description |
|-------|-------------|
| **Indoor AQI** | Current Air Quality Index (0-500) |
| **Temperature °C** | Current temperature from DHT22 |
| **Humidity %** | Current humidity from DHT22 |
| **Gas (VOC)** | Gas reading from MQ-135 |

### Health Alert Section

| Value | Description |
|-------|-------------|
| **Status** | Good / Moderate / Poor / Very Poor / Severe / Hazardous |
| **Advice** | Health-based recommendation |

### ML Predictions

| Value | Description |
|-------|-------------|
| **Current (RF)** | Current air quality classification (Random Forest) with class probability % (not a validated reliability measure) |
| **Future (LSTM)** | Predicted status for next 10 readings (LSTM) with class probability % (not a validated reliability measure) |
| **Trend (LR)** | Linear Regression AQI prediction + trend direction (rising/falling/stable) |

### Charts

| Chart | Shows |
|-------|-------|
| **Temp & Humidity** | Line chart of last 40 readings |
| **Gas & AQI** | Line chart showing correlation |

### Dev Mode (Toggle in Sidebar)

| Value | Description |
|-------|-------------|
| **Outlier Rejection** | % of readings rejected as impossible values |
| **Rejected/Total** | Count of rejected vs total readings |
| **Serial** | ✅ (connected) or ❌ (not connected) |

---

## 6. CLI Dashboard Values

```
============================================================
🛡️ AeroGuard AI - Phase 1.5 CLI Dashboard
============================================================

[1] Initializing database...
    ✓ Database ready

[2] Loading ML models...
  ✓ Random Forest loaded
  ✓ Linear Regression loaded
  ✓ LSTM loaded
  ✓ Scaler loaded
    ✓ All models loaded

[3] Auto-detecting ESP32...
    ⚠️ ESP32 not found - using mock data
    (Connect ESP32 and restart to use real sensors)

[4] Running in MOCK mode
    Scenarios: healthy → rising → spike → falling
```

### Real-Time Output

```
--------------------------------------------------
🕐 20:47:20 | 📊 T=24.8°C H=55.6% G=172
🌬️ AQI: 79 (Moderate)
🤖 RF: Moderate (79%) | LSTM: Poor (75%) | Trend: 77 (rising)
🏥 Moderate: Normal outdoor activities.
📦 Mode: MOCK | Reject: 0.0%
--------------------------------------------------
```

| Value | Description |
|-------|-------------|
| 🕐 Timestamp | Current time HH:MM:SS |
| 📊 Sensors | T (Temp °C), H (Humidity %), G (Gas raw value) |
| 🌬️ AQI | Air Quality Index + (Category) |
| 🤖 RF | Random Forest classification with class probability |
| 🤖 LSTM | LSTM prediction with class probability |
| 🤖 Trend | Linear Regression AQI + trend direction |
| 🏥 | Health status + recommendation |
| 📦 Mode | MOCK or REAL + outlier rejection % |

### ML feature contract & preprocessing

The models consume exactly three features, defined once in `preprocessing.py`:

```
FEATURE_COLS = ['temp', 'hum', 'gas']
```

PM2.5 is deliberately **not** a feature: the sensor node cannot measure it, so
training on it would be train/serve skew.

* `preprocessing.py` is the single source of truth for cleaning. Both
  `train_model.py` / `train_lstm.py` and the live `predictors.py` path call it,
  so train-time and serve-time transformations match.
* `AQI_LABELS` (0–5: Good → Hazardous) is the canonical class mapping. Do not
define label maps elsewhere.
* `model_metadata.json` records the feature schema, labels and honest metrics
  (accuracy, MAE, RMSE, R²). Prediction output labels it "class probability",
  not a validated reliability measure — a statistically justified reliability
  indicator is a later phase.

### Dataset, calibration & validation

Training uses the **UCI Air Quality dataset** (De Vito et al., 2008): 9,357
hourly records from a metal-oxide sensor array co-located with a certified
reference analyzer. Fetch it with `python fetch_datasets.py`.

* The gas channel is **calibrated** (`calibration.py`) against reference
  benzene with temperature/humidity compensation. Calibration R² = 0.985,
  MAE = 0.73 µg/m³. The model feature is a benzene-equivalent VOC
  concentration, not a raw ADC value.
* The predicted target is the **VOC air-quality sub-index** (benzene
  sub-index). The NO₂-driven composite AQI is not physically measurable with
  an MQ-135-class sensor (r = 0.09 with NO₂), so it is kept only as a
  reference column (`aqi_composite`).
* `validation.py` reports MAE, RMSE, R², Pearson r and AQI-category
  agreement against the certified reference. Current held-out results:
  RF 98.4% accuracy, Linear Regression R² 0.969 / r 0.991, category
  within-one 100%.

> **Live-node caveat:** the UCI calibration coefficients describe the UCI
> sensor array, not the MQ-135. The live node needs its own clean-air `R0`
> and a reference gas before deployment. The methodology is complete; only
> the per-device coefficients are outstanding.

> **Import order (macOS):** `bootstrap_tf.py` must be imported before pandas,
> streamlit or sklearn in any entry point. On some macOS setups (pandas 3.x +
> TensorFlow 2.21 on arm64), importing pandas first makes every `fit`/`predict`
> call deadlock. This is why `dashboard.py`, `dashboard_v2.py` and
> `backend/main.py` start with `import bootstrap_tf`.

---

## 7. Database

All data is saved to `aeroguard.db` (SQLite):

```bash
# View database
sqlite3 aeroguard.db

# Query examples
SELECT * FROM readings ORDER BY id DESC LIMIT 10;
SELECT * FROM predictions ORDER BY id DESC LIMIT 5;
SELECT * FROM alerts ORDER BY id DESC LIMIT 5;
```

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| **Serial port error** | Check ESP32 is connected; try different port in config.yaml |
| **Models not loading** | Run `python train_model.py` and `python train_lstm.py` |
| **No data showing** | Enable Dev Mode in sidebar to see connection status |
| **Port already in use** | Kill existing streamlit: `pkill -f streamlit` |

### Common Commands

```bash
# Kill existing processes
pkill -f streamlit

# Check serial ports
python -c "import serial; print([p.device for p in serial.tools.list_ports.comports()])"

# Reset database (delete all data)
rm aeroguard.db

# View errors
streamlit run dashboard.py 2>&1
```

---

## 9. Configuration

Edit `config.yaml` to customize:

```yaml
serial:
  port: /dev/ttyUSB0      # Change if different
  baudrate: 115200

models:
  rf_model: rf_air_model.pkl
  lstm_model: lstm_air_model.h5

aqi:
  good_max: 50
  moderate_max: 100

dashboard:
  refresh_rate: 2

api:
  outdoor_enabled: true
```

---

## 10. Project Files

```
AeroGuard AI/
├── README.md              # This file
├── config.yaml            # Configuration
├── requirements.txt       # Python dependencies
│
├── Core Modules/
│   ├── aqi_utils.py       # AQI calculation + outlier detection
│   ├── alerts.py         # Health alerts
│   ├── preprocessing.py  # Cleaning, validation, feature contract
│   ├── dataset.py        # UCI + legacy dataset loaders, reference AQI
│   ├── calibration.py    # Gas calibration (log-log + T/RH) + MQ-135 helpers
│   ├── validation.py     # MAE/RMSE/r/category-agreement vs reference
│   ├── reliability.py    # Prediction interval + category probability
│   ├── risk.py           # Airborne disease environmental risk (Low/Mod/High)
│   ├── explain.py        # SHAP (TreeSHAP) explanations
│   ├── training.py       # Shared dataset -> split -> calibration pipeline
│   ├── predictors.py      # ML pipeline
│   ├── database.py        # SQLite persistence
│   └── bootstrap_tf.py    # Import-order shim (see note below)
│
├── Scripts/
│   ├── fetch_datasets.py     # Download the UCI reference dataset
│   ├── collect_real_data.py  # Fetch real data from API
│   ├── train_model.py        # Train RF + LR
│   └── train_lstm.py         # Train LSTM
│
├── Dashboards/
│   ├── dashboard.py       # Streamlit UI
│   └── dashboard_v2.py   # CLI UI
│
├── Data/
│   ├── aeroguard.db      # SQLite database (auto-created)
│   └── real_air_data.csv # Training data
│
└── Models/
    ├── rf_air_model.pkl
    ├── lr_trend_model.pkl
    ├── lstm_air_model.h5
    ├── scaler.pkl
    ├── gas_calibrator.pkl   # fitted gas calibration
    └── model_metadata.json  # feature schema, labels, metrics, validation

Backend (Phase 2)
├── backend/
│   ├── main.py
│   ├── chat.py
│   ├── config.py
│   ├── requirements.txt
│   └── README.md

Mobile App (Phase 2)
├── mobile/
│   ├── app/
│   │   ├── (tabs)/
│   │   │   ├── index.tsx
│   │   │   └── chat.tsx
│   │   └── _layout.tsx
│   ├── lib/api.ts
│   └── README.md

Setup & Launch
├── setup_and_run.py
├── start.bat
└── start.sh
```

---

## 11. Phase 2: Mobile App + AI Chatbot

### What's new

- **FastAPI backend** (`backend/`) exposes sensor data and an AI chat endpoint.
- **Local SLM chatbot** uses **Qwen 3.5 2B** running via **Ollama**.
- **React Native mobile app** (`mobile/`) with two tabs:
  - **Dashboard** — live AQI, readings, alerts, outdoor AQI.
  - **AI Chat** — ask natural-language questions about the data.

### One-run setup (Windows, macOS, Linux)

```bash
# Windows — double-click or run in terminal
start.bat

# macOS / Linux
./start.sh
```

Or run the Python launcher directly:

```bash
python setup_and_run.py
```

This single script will:
1. Create a Python virtual environment.
2. Install backend dependencies.
3. Check for and install Ollama if missing.
4. Pull the `qwen3.5:2b` model.
5. Start Ollama and the FastAPI backend.

> **Tip:** If you already pulled the model (or want to pull it later), start with:
> ```bash
> ./start.sh --skip-model-pull
> ```
> On Windows: `start.bat --skip-model-pull`

### Mobile app setup

1. Start the backend with `setup_and_run.py` and keep that terminal open.
2. Open a **second terminal** and set the backend URL in `mobile/.env`:

   ```bash
   # Simulator / same machine
   echo "EXPO_PUBLIC_API_URL=http://localhost:8000" > mobile/.env

   # Real device on same WiFi
   echo "EXPO_PUBLIC_API_URL=http://192.168.1.XXX:8000" > mobile/.env
   ```

3. In that second terminal, install and run:

   ```bash
   cd mobile
   npm install
   npx expo start
   ```

---

## 12. First-Time Setup Commands

```bash
# Complete setup for a new machine:

# 1. Clone/get project
cd final_year_project-IT

# 2. Create environment
python3 -m venv venv
source venv/bin/activate

# 3. Install everything
pip install --upgrade pip
pip install numpy tensorflow scikit-learn joblib pandas requests pyserial
pip install streamlit plotly pyyaml

# 4. Train models (optional - already trained)
# python train_model.py
# python train_lstm.py

# 5. Run dashboard
streamlit run dashboard.py
```

---

## 13. Quick Reference Card

| Action | Command |
|--------|---------|
| One-click setup + backend | `start.bat` (Windows) / `./start.sh` (macOS/Linux) |
| Skip model download | `./start.sh --skip-model-pull` / `start.bat --skip-model-pull` |
| Start backend only | `python -m uvicorn backend.main:app --reload` |
| Start Streamlit | `streamlit run dashboard.py` |
| Start CLI | `python dashboard_v2.py` |
| Start mobile app | `cd mobile && npx expo start` |
| Dev Mode | Toggle in sidebar |
| Mock Mode | Auto-detected when no ESP32 |
| Data Location | `aeroguard.db` (SQLite) |

---

**For Support:** Check the troubleshooting section above or review `config.yaml` settings.

---

*Generated for client distribution - AeroGuard AI Phase 1.5*
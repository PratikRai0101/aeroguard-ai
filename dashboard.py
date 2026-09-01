# dashboard.py
# AeroGuard AI - Phase 1.5 Dashboard
# Real-time AQI Monitoring with LSTM Predictions, SQLite Persistence, and Dev Mode

import streamlit as st
import pandas as pd
import numpy as np
import serial
import time
import requests
import random
from datetime import datetime
import plotly.graph_objects as go
import streamlit.components.v1 as components

# Import custom modules
from aqi_utils import calculate_aqi, get_aqi_info, validate_reading, OutlierDetector
from alerts import get_alert, get_preventive_measures, get_alert_color
from predictors import AQIPredictor
from database import SensorDatabase
import yaml

# Load configuration
def load_config(filename='config.yaml'):
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except:
        return None

config = load_config()

# ==== SETTINGS ====
SERIAL_PORT = config.get('serial', {}).get('port', '/dev/ttyUSB0') if config else '/dev/ttyUSB0'
BAUD = config.get('serial', {}).get('baudrate', 115200) if config else 115200


def connect_serial_with_retry(port, baud, max_retries=3):
    """Auto-connect to serial with retries"""
    for attempt in range(max_retries):
        try:
            ser = serial.Serial(port, baud, timeout=1)
            return ser, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None, str(e)
    return None, "No connection"

st.set_page_config(
    page_title="AeroGuard AI - Phase 1.5", 
    layout="wide", 
    page_icon="🛡️"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ==== MODEL LOADING ====
@st.cache_resource
def load_ml_pipeline():
    try:
        return AQIPredictor('.')
    except Exception as e:
        print(f"Model loading error: {e}")
        return None

predictor = load_ml_pipeline()

# ==== DATABASE INITIALIZATION ====
@st.cache_resource
def init_database():
    return SensorDatabase('aeroguard.db')

db = init_database()

# ==== INITIALIZE SESSION STATE ====
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Temp', 'Hum', 'Gas', 'AQI', 'Status'])

if 'outlier_detector' not in st.session_state:
    st.session_state.outlier_detector = OutlierDetector(window_size=10)

if 'serial_connected' not in st.session_state:
    st.session_state.serial_connected = False

if 'dev_mode' not in st.session_state:
    st.session_state.dev_mode = False

# Initialize serial connection in session state if not already done
if 'serial_port' not in st.session_state:
    st.session_state.serial_port = None

# ==== AUTO-DETECT ESP32 ====
# Try to detect and open serial port
detected = False
detected_port = '/dev/ttyUSB0'

# Only try to detect if not already connected
if not st.session_state.serial_connected or st.session_state.serial_port is None:
    try:
        # Try the configured port first (for example, macOS /dev/cu.usbserial-*),
        # then common Linux ESP32 device names.
        ports_to_try = list(dict.fromkeys([
            SERIAL_PORT, '/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1'
        ]))
        for port in ports_to_try:
            try:
                test_ser = serial.Serial(port, BAUD, timeout=0.5)
                detected_port = port
                detected = True
                break
            except:
                continue
    except:
        detected = False
    
    if detected:
        st.session_state.serial_connected = True
        try:
            st.session_state.serial_port = serial.Serial(detected_port, BAUD, timeout=1)
            st.session_state.serial_port.flush()
            st.session_state.serial_port.flushInput()  # Clear stale buffer
        except:
            st.session_state.serial_connected = False
            st.session_state.serial_port = None
    else:
        st.session_state.serial_connected = False
        st.session_state.serial_port = None

# Mode indicator
mode_indicator = "🔴 REAL SENSOR" if st.session_state.serial_connected else "🟡 MOCK MODE"
detected_port = st.session_state.serial_port.port if st.session_state.serial_port else 'None'

# Debug info for dev mode
debug_info = {
    'serial_connected': st.session_state.serial_connected,
    'port': detected_port,
    'last_temp': st.session_state.get('last_temp'),
    'last_hum': st.session_state.get('last_hum'),
    'last_gas': st.session_state.get('last_gas'),
    'last_update': st.session_state.get('last_reading_time'),
}

# ==== HELPER FUNCTIONS ====
def create_gauge(value, title, max_val, color, suffix='', min_val=0):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': 'white'}},
        number={'font': {'color': 'white', 'size': 28}, 'suffix': suffix},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': "white", 'tickfont': {'size': 10}},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        height=220, 
        margin=dict(l=10, r=10, t=50, b=10), 
        paper_bgcolor="rgba(0,0,0,0)", 
        font={'color': "white"}
    )
    return fig

def create_aqi_gauge(aqi):
    aqi_info = get_aqi_info(aqi)
    return create_gauge(aqi, f"AQI<br>{aqi_info['name']}", 500, aqi_info['color'], min_val=0)

def play_alert_sound():
    beep_html = """
    <script>
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    osc.type = 'square';
    osc.frequency.setValueAtTime(600, ctx.currentTime);
    osc.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.3);
    </script>
    """
    components.html(beep_html, height=0)

def connect_serial(port, baud, max_retries=5):
    """Auto-reconnect serial with retries"""
    for attempt in range(max_retries):
        try:
            ser = serial.Serial(port, baud, timeout=1)
            return ser, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None, str(e)
    return None, "Max retries exceeded"

def get_status_from_db():
    """Fetch last 100 readings from database"""
    try:
        rows = db.get_readings(limit=100)
        if rows:
            df = pd.DataFrame(rows, columns=['Time', 'Temp', 'Hum', 'Gas', 'AQI', 'Status'])
            return df
    except:
        pass
    return None

# ==== OUTDOOR API ====
@st.cache_data(ttl=1800)
def get_outdoor_aqi():
    try:
        lat = config.get('api', {}).get('latitude', 21.1458) if config else 21.1458
        lon = config.get('api', {}).get('longitude', 79.0882) if config else 79.0882
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=european_aqi,pm2_5"
        res = requests.get(url).json()
        return res.get('current')
    except:
        return None

outdoor_data = get_outdoor_aqi()

# ==== UI HEADER ====
st.title("🛡️ AeroGuard AI - Phase 1.5")
st.markdown("*Real-time AQI Monitoring with LSTM Predictions*")

# ==== DEV MODE TOGGLE ====
dev_mode = st.sidebar.toggle("Dev Mode", value=False)
st.session_state.dev_mode = dev_mode

# Sidebar
with st.sidebar:
    st.header("⚙️ System Controls")
    st.markdown("---")
    
    # Mode indicator
    st.write(f"**Mode:** {mode_indicator}")
    
    # Port info (if connected)
    if detected_port != 'None':
        st.write(f"**Port:** {detected_port}")
    
    # Dev Mode toggle
    if dev_mode:
        st.write(f"**Serial:** {'✅ Connected' if st.session_state.serial_connected else '❌ Disconnected'}")
        st.write(f"**Port:** {detected_port}")
        st.write("---")
        st.write("**Raw vs Displayed:**")
        st.write(f"- Raw: T={st.session_state.get('last_raw_temp', 'N/A')}, H={st.session_state.get('last_raw_hum', 'N/A')}, G={st.session_state.get('last_raw_gas', 'N/A')}")
        st.write(f"- Disp: T={st.session_state.get('last_temp', 'N/A')}, H={st.session_state.get('last_hum', 'N/A')}, G={st.session_state.get('last_gas', 'N/A')}")
        st.write(f"- Updated: {st.session_state.get('last_reading_time', 'N/A')}")
        st.write(f"- Mode: {'SENSOR' if not st.session_state.get('using_mock_data') else 'MOCK'}")
        st.write("---")
    db_stats = db.get_statistics()
    st.metric("Total Readings", db_stats['readings'])
    st.metric("Total Alerts", db_stats['alerts'])
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Export data
    if not st.session_state.history.empty:
        csv = st.session_state.history.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download CSV", data=csv, file_name="session_log.csv", mime="text/csv")
    
    st.markdown("---")
    st.subheader("📊 Session Statistics")
    if not st.session_state.history.empty:
        st.write(f"**Max Temp:** {st.session_state.history['Temp'].max():.1f} °C")
        st.write(f"**Max Gas:** {st.session_state.history['Gas'].max():.0f}")
        st.write(f"**Max AQI:** {st.session_state.history['AQI'].max():.0f}")
        st.write(f"**Avg Hum:** {st.session_state.history['Hum'].mean():.1f} %")
    
    # Dev Mode Stats
    if dev_mode:
        st.markdown("---")
        st.subheader("🔧 Dev Mode")
        
        outlier_stats = st.session_state.outlier_detector.get_stats()
        st.write(f"**Outlier Rejection:** {outlier_stats['rejection_rate']:.1f}%")
        st.write(f"**Rejected:** {outlier_stats['rejected']}/{outlier_stats['total']}")
        
        # Connection status
        status_icon = "✅" if st.session_state.serial_connected else "❌"
        st.write(f"**Serial:** {status_icon}")
    
    st.markdown("---")
    st.subheader("🤖 Model Status")
    if predictor:
        st.success("✅ Models Loaded")
    else:
        st.error("⚠️ Model Error")

# Context Row
col_in, col_out = st.columns(2)
with col_out:
    if outdoor_data:
        st.info(f"🌍 **Outdoor AQI (Nagpur):** {outdoor_data.get('european_aqi', 'N/A')} | PM2.5: {outdoor_data.get('pm2_5', 'N/A')} µg/m³")
    else:
        st.info("🌍 Outdoor API Offline")

st.divider()

# ==== MAIN DATA LOOP ====
latest_line = None

DEBUG_READ = True  # Enable debug logging

# Always initialize t, h, g before any if/else
t = h = g = None
latest_line = None
raw_temp = raw_hum = raw_gas = None  # Raw sensor values
reading_valid = False
using_mock_data = False

if st.session_state.serial_connected and st.session_state.serial_port:
    try:
        ser_port = st.session_state.serial_port
        # BLOCKING READ: Wait up to 1 second for ESP32 to send fresh data
        # This ensures we always get the latest reading, not stale cached data
        raw = ser_port.readline()
        latest_line = raw.decode('utf-8', errors='ignore').replace('\x00', '').strip()
        
        if latest_line and ',' in latest_line and latest_line.count(',') == 2:
            # Check if valid non-zero reading
            parts = latest_line.split(',')
            try:
                vals = [float(p.strip()) for p in parts[:3]]
                # If ALL values are 0, sensors are disconnected - ignore
                if vals[0] == 0 and vals[1] == 0 and vals[2] == 0:
                    latest_line = None
                    if DEBUG_READ:
                        print("[SERIAL] Sensors disconnected (0,0,0)")
                else:
                    st.session_state.serial_connected = True
                    if DEBUG_READ:
                        print(f"[SERIAL READ] {latest_line}")
            except:
                latest_line = None
        elif latest_line:
            # Got data but wrong format
            if DEBUG_READ:
                print(f"[SERIAL SKIP] Bad format: {latest_line}")
            latest_line = None
        else:
            # Timeout - no data from ESP32
            if DEBUG_READ:
                print("[SERIAL] Timeout waiting for data")
            latest_line = None
            
    except Exception as e:
        # Serial error - try to reconnect
        print(f"[SERIAL ERROR] {e}")
        try:
            st.session_state.serial_port.close()
        except:
            pass
        st.session_state.serial_connected = False
        st.session_state.serial_port = None
        latest_line = None
    
    if latest_line and latest_line.startswith(('0','1','2','3','4','5','6','7','8','9','.')):
        parts = [p.strip().replace('\r', '').replace('\n', '') for p in latest_line.split(',')]
        
        # Validate: must have at least 3 parts AND each must be a valid number
        if len(parts) >= 3:
            valid_nums = []
            for i, p in enumerate(parts[:3]):
                try:
                    val = float(p)
                    # Check each value's proper range
                    if i == 0:  # temp: -20 to 60 °C
                        if -20 <= val <= 60:
                            valid_nums.append(val)
                        else:
                            valid_nums.append(None)
                    elif i == 1:  # humidity: 0 to 100 %
                        if 0 <= val <= 100:
                            valid_nums.append(val)
                        else:
                            valid_nums.append(None)
                    else:  # gas: 0 to 5000 (expanded for high readings)
                        if 0 <= val <= 5000:
                            valid_nums.append(val)
                        else:
                            valid_nums.append(None)
                except:
                    valid_nums.append(None)
            
            # All 3 values must be valid
            if all(v is not None for v in valid_nums):
                raw_temp, raw_hum, raw_gas = valid_nums[0], valid_nums[1], valid_nums[2]  # Store raw
                t, h, g = raw_temp, raw_hum, raw_gas
            else:
                t = h = g = None
                raw_temp = raw_hum = raw_gas = None
                print(f"[PARSE ERROR] Invalid values: {parts[:3]}")
            
            if t is not None and h is not None and g is not None:
                if DEBUG_READ:
                    print(f"[PARSED] raw_t={raw_temp}, raw_h={raw_hum}, raw_g={raw_gas} | disp_t={t}, disp_h={h}, disp_g={g}")
                # OUTLIER DETECTION
                outlier_result, outlier_errors = st.session_state.outlier_detector.add_reading(t, h, g)
                
                if outlier_result is None:
                    # Invalid reading - reject but DON'T fall back to mock
                    print(f"[OUTLIER REJECTED] {outlier_errors}")
                    if dev_mode:
                        st.sidebar.warning(f"Outlier rejected: {outlier_errors}")
                    # Clear values to avoid displaying invalid data
                    t = h = g = None
                    reading_valid = False
                else:
                    reading_valid = True
                    using_mock_data = False
                    t, h, g = outlier_result['temp'], outlier_result['hum'], outlier_result['gas']
                    
                    # Add small jitter for demo - sensors are very precise so add slight variation
                    if abs(g) > 0:  # Only add jitter if we have real data
                        t += random.uniform(-0.15, 0.15)
                        h += random.uniform(-0.5, 0.5)
                        g += random.uniform(-5, 5)
                    
                    if DEBUG_READ:
                        print(f"[VALID] raw=({raw_temp},{raw_hum},{raw_gas}) disp=({t:.1f},{h:.1f},{g:.0f})")
        else:
            print(f"[PARSE ERROR] Not enough parts: {parts}")
    else:
        # No valid serial data - keep previous values or wait
        if DEBUG_READ and latest_line:
            print(f"[SKIP] Not numeric: {latest_line}")
        t = h = g = None

# Store valid serial reading for next cycle
if t is not None:
    st.session_state.last_temp = t
    st.session_state.last_hum = h
    st.session_state.last_gas = g
    st.session_state.last_raw_temp = raw_temp
    st.session_state.last_raw_hum = raw_hum
    st.session_state.last_raw_gas = raw_gas
    st.session_state.last_reading_time = time.strftime("%H:%M:%S")
    if DEBUG_READ:
        print(f"[STORED] disp=({t},{h},{g})")

# If NO valid reading from serial, use last known values for display (NOT mock)
# This prevents showing fake data when sensors are just slow
if t is None:
    if st.session_state.get('last_temp') is not None:
        t = st.session_state.last_temp
        h = st.session_state.last_hum
        g = st.session_state.last_gas
        reading_valid = True
        if DEBUG_READ:
            print(f"[HOLD] Using last known values: ({t:.1f},{h:.1f},{g:.0f})")
    elif not st.session_state.serial_connected:
        # No serial hardware and no previous data - generate mock data
        if DEBUG_READ:
            print("[MOCK] Generating mock data")
        if 'mock_t' not in st.session_state:
            st.session_state.mock_t = 24.0
            st.session_state.mock_h = 55.0
            st.session_state.mock_g = 150.0
            st.session_state.mock_counter = 0
            st.session_state.mock_scenario = 'healthy'
        
        st.session_state.mock_counter += 1
        
        # Only change scenario every 30 readings (~60 seconds)
        if st.session_state.mock_counter % 30 == 0:
            scenarios = ['healthy', 'healthy', 'healthy', 'rising', 'spike', 'falling']
            st.session_state.mock_scenario = random.choice(scenarios)
        
        scenario = st.session_state.mock_scenario
        
        # Generate small realistic variations (not fully random)
        if scenario == 'healthy':
            # Small jitter around steady values
            t = st.session_state.mock_t + random.uniform(-0.3, 0.3)
            h = st.session_state.mock_h + random.uniform(-1, 1)
            g = st.session_state.mock_g + random.uniform(-5, 5)
            # Gradually drift back to baseline
            st.session_state.mock_t = st.session_state.mock_t * 0.95 + 24.5 * 0.05
            st.session_state.mock_h = st.session_state.mock_h * 0.95 + 55 * 0.05
            st.session_state.mock_g = st.session_state.mock_g * 0.95 + 150 * 0.05
        elif scenario == 'rising':
            # Slowly increasing
            st.session_state.mock_g += random.uniform(0, 3)
            st.session_state.mock_t += random.uniform(0, 0.2)
            t = st.session_state.mock_t + random.uniform(-0.2, 0.2)
            h = st.session_state.mock_h + random.uniform(-0.5, 0.5)
            g = st.session_state.mock_g + random.uniform(-3, 3)
        elif scenario == 'spike':
            # Sudden jump
            if st.session_state.mock_g < 300:
                st.session_state.mock_g += random.uniform(10, 20)
            t = st.session_state.mock_t + random.uniform(-0.2, 0.2)
            h = st.session_state.mock_h + random.uniform(-0.5, 0.5)
            g = st.session_state.mock_g + random.uniform(-5, 5)
        else:  # falling
            # Slowly decreasing
            if st.session_state.mock_g > 100:
                st.session_state.mock_g -= random.uniform(3, 6)
            st.session_state.mock_t = st.session_state.mock_t * 0.98 + 24.5 * 0.02
            t = st.session_state.mock_t + random.uniform(-0.2, 0.2)
            h = st.session_state.mock_h + random.uniform(-0.5, 0.5)
            g = st.session_state.mock_g + random.uniform(-5, 5)
        
        # Clamp realistic ranges
        t = max(15, min(45, t))
        h = max(20, min(90, h))
        g = max(50, min(500, g))
        
        reading_valid = True
        using_mock_data = True
        st.session_state.using_mock_data = True
        
        if DEBUG_READ:
            print(f"[MOCK] disp=({t:.1f},{h:.1f},{g:.0f})")
    else:
        # No previous data at all - show zeros/waiting
        t = h = g = 0
        reading_valid = False
        if DEBUG_READ:
            print("[WAIT] No sensor data yet")

# ==== DISPLAY ====
if reading_valid:
    curr_time = datetime.now().strftime("%H:%M:%S")
    
    # Estimate AQI from gas
    est_pm25 = g * 0.15
    aqi = calculate_aqi(pm25=est_pm25)
    aqi_info = get_aqi_info(aqi)
    
    # ML Predictions
    current_pred = {'label': 'Unknown', 'confidence': 0}
    future_pred = {'label': 'Unknown', 'confidence': 0}
    trend_pred = {'aqi': aqi, 'trend': 'stable'}
    
    if predictor:
        # First add data to buffer
        predictor.add_reading(t, h, g)
        
        # Get predictions
        ml_result = predictor.predict_all(t, h, g)
        current_pred = ml_result.get('current', {})
        future_pred = ml_result.get('future', {})
        trend_pred = ml_result.get('trend', {})
        
        # Debug output
        print(f"[ML] RF={current_pred.get('label')} {current_pred.get('confidence',0):.0f}% | LSTM={future_pred.get('label')} {future_pred.get('confidence',0):.0f}% | Buffer={len(predictor.buffer)}")
    
    # Health Alert
    alert = get_alert(aqi)
    
    # Save to database
    db.add_reading(t, h, g, aqi, aqi_info['name'], source='mock' if using_mock_data else 'sensor')
    if predictor:
        db.add_prediction(
            current_pred.get('label', 'Unknown'),
            current_pred.get('confidence', 0),
            future_pred.get('label', 'Unknown'),
            future_pred.get('confidence', 0),
            trend_pred.get('trend', 'stable'),
            trend_pred.get('aqi', aqi)
        )
    
    if alert['category'] >= 2:
        db.add_alert(aqi, alert['name'], alert['advice'], alert['category'])
    
    # Update history
    new_row = pd.DataFrame([{
        'Time': curr_time,
        'Temp': t,
        'Hum': h,
        'Gas': g,
        'AQI': aqi,
        'Status': aqi_info['name']
    }])
    st.session_state.history = pd.concat([
        st.session_state.history, 
        new_row
    ], ignore_index=True).tail(40)
    
    # ==== DISPLAY ====
    with col_in:
        st.success(f"🏠 **Indoor AQI:** {int(aqi)} ({aqi_info['name']})")
    
    # Gauges
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(create_gauge(t, "Temperature<br>°C", 50, "#FF5A5F"), width='stretch')
    with g2:
        st.plotly_chart(create_gauge(h, "Humidity<br>%", 100, "#00A699"), width='stretch')
    with g3:
        alert_color = get_alert_color(aqi)
        st.plotly_chart(create_gauge(g, "Gas<br>(VOC)", 1000, alert_color), width='stretch')
    with g4:
        st.plotly_chart(create_aqi_gauge(aqi), width='stretch')
    
    # ==== ML PREDICTIONS ====
    st.subheader("🤖 ML Predictions")
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    rf_conf = current_pred.get('confidence', 0)
    rf_label = current_pred.get('label', 'Unknown')
    rf_status = "⚠️" if rf_conf < 50 else "✅"
    
    with pred_col1:
        st.metric(
            f"Current RF", 
            f"{rf_label} {rf_status}", 
            f"{rf_conf:.1f}%"
        )
    
    # LSTM - show if loaded or buffering
    lstm_conf = future_pred.get('confidence', 0)
    lstm_label = future_pred.get('label', 'Loading...')
    if lstm_conf == 0 and lstm_label == 'Moderate':
        lstm_label = 'Not Loaded'
    
    with pred_col2:
        st.metric("Future LSTM", lstm_label, f"{lstm_conf:.1f}%")
    
    with pred_col3:
        trend_dir = trend_pred.get('trend', 'stable')
        trend_aqi = trend_pred.get('aqi', aqi)
        emoji = "📈" if trend_dir == "rising" else "📉" if trend_dir == "falling" else "➡️"
        st.metric("Trend (LR)", f"AQI {int(trend_aqi)} {emoji}", trend_dir)
    
    # ==== HEALTH ALERTS ====
    st.subheader("🏥 Health Alerts")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.info(f"**Status:** {alert['name']}")
        st.write(f"**Advice:** {alert['advice']}")
    
    with alert_col2:
        measures = get_preventive_measures(aqi)
        for measure in measures:
            st.write(f"• {measure}")
    
    # Critical alerts
    if alert['category'] >= 3:
        st.error("⚠️ **CRITICAL AIR QUALITY**")
        play_alert_sound()
    
    # ==== CHARTS ====
    st.subheader("📈 Trends")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.line_chart(st.session_state.history.set_index('Time')[['Temp', 'Hum']], height=250)
    
    with chart_col2:
        st.line_chart(st.session_state.history.set_index('Time')[['Gas', 'AQI']], height=250)
        
else:
    st.error(f"❌ Hardware Disconnected. Check {SERIAL_PORT}.")
    st.info("💡 Switch to Mock Mode in dashboard_v2.py for testing without hardware.")

# ==== REFRESH ====
time.sleep(2)
st.rerun()

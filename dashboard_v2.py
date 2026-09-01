# dashboard_v2.py
# AeroGuard AI - Phase 1.5 CLI Dashboard
# Real-time AQI Monitoring with Auto-Detect ESP32

import os
import time
import numpy as np
import random
from collections import deque

# Import custom modules
from aqi_utils import calculate_aqi, get_aqi_info, OutlierDetector
from alerts import get_alert
from predictors import AQIPredictor
from database import SensorDatabase
import yaml
import serial

# Load configuration
def load_config(filename='config.yaml'):
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except:
        return None

config = load_config('config.yaml')

# ==== CONFIGURATION ====
FORCE_MOCK = os.getenv("AEROGUARD_FORCE_MOCK", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
SERIAL_PORT = config.get('serial', {}).get('port', '/dev/ttyUSB0') if config else '/dev/ttyUSB0'
BAUD = config.get('serial', {}).get('baudrate', 115200) if config else 115200


def connect_serial_with_retry(port, baud, max_retries=3):
    """Connect to serial with auto-retry"""
    for attempt in range(max_retries):
        try:
            ser = serial.Serial(port, baud, timeout=1)
            return ser
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def process_line(line):
    """Process serial line into sensor data"""
    try:
        parts = line.strip().split(',')
        if len(parts) == 3:
            return [float(p) for p in parts]
    except:
        pass
    return None


def mock_data_generator():
    """Generate realistic mock sensor data"""
    scenarios = ['healthy', 'rising', 'spike', 'falling']
    idx = 0
    while True:
        scenario = scenarios[idx % 4]
        if scenario == 'healthy':
            for _ in range(15):
                yield [random.uniform(23, 26), random.uniform(50, 60), random.uniform(100, 180)]
        elif scenario == 'rising':
            gas_start = random.uniform(150, 200)
            for g in np.linspace(gas_start, gas_start + 100, 15):
                yield [random.uniform(25, 28), random.uniform(55, 65), g + random.uniform(-5, 5)]
        elif scenario == 'spike':
            for g in np.concatenate([np.linspace(150, 180, 5), np.linspace(200, 350, 5), np.linspace(380, 450, 5)]):
                yield [random.uniform(27, 30), random.uniform(60, 70), g + random.uniform(-10, 10)]
        elif scenario == 'falling':
            for g in np.linspace(400, 120, 15):
                yield [random.uniform(24, 26), random.uniform(50, 55), g + random.uniform(-10, 10)]
        idx += 1


def main():
    print("\n" + "=" * 60)
    print("🛡️ AeroGuard AI - Phase 1.5 CLI Dashboard")
    print("=" * 60)
    
    # Initialize database
    print("\n[1] Initializing database...")
    db = SensorDatabase('aeroguard.db')
    print(f"    ✓ Database ready")
    
    # Load ML models
    print("\n[2] Loading ML models...")
    predictor = AQIPredictor('.')
    print("    ✓ All models loaded")
    
    # Initialize components
    outlier_detector = OutlierDetector(window_size=10)
    buffer = deque(maxlen=10)
    
    # Auto-detect ESP32
    print(f"\n[3] Auto-detecting ESP32...")
    ser = connect_serial_with_retry(SERIAL_PORT, BAUD, max_retries=2)
    
    if ser and not FORCE_MOCK:
        print(f"    ✓ ESP32 detected on {SERIAL_PORT}")
        run_real_mode(ser, db, predictor, outlier_detector, buffer)
    else:
        if FORCE_MOCK:
            print("    ⚠️ Mock mode forced by AEROGUARD_FORCE_MOCK")
        else:
            print("    ⚠️ ESP32 not found - using mock data")
            print("    (Connect ESP32 and restart to use real sensors)")
        run_mock_mode(db, predictor, outlier_detector, buffer)


def run_mock_mode(db, predictor, outlier_detector, buffer):
    """Run in mock/simulation mode"""
    print("\n[4] Running in MOCK mode")
    print("    Scenarios: healthy → rising → spike → falling\n")
    
    data_gen = mock_data_generator()
    
    try:
        while True:
            reading = next(data_gen)
            result, errors = outlier_detector.add_reading(reading[0], reading[1], reading[2])
            
            if result is None:
                continue
            
            buffer.append(reading)
            t, h, g = reading[0], reading[1], reading[2]
            
            if len(buffer) < 10:
                print(f"    Buffer: {len(buffer)}/10...")
                time.sleep(0.8)
                continue
            
            # Process reading
            aqi = calculate_aqi(pm25=g * 0.15)
            aqi_info = get_aqi_info(aqi)
            predictor.add_reading(t, h, g)
            ml = predictor.predict_all(t, h, g)
            alert = get_alert(aqi)
            
            # Save to DB
            db.add_reading(t, h, g, aqi, aqi_info['name'], source='mock')
            db.add_prediction(ml['current']['label'], ml['current']['confidence'],
                          ml['future']['label'], ml['future']['confidence'],
                          ml['trend']['trend'], ml['trend']['aqi'])
            
            outlier_stats = outlier_detector.get_stats()
            print(f"\n{'-'*50}")
            print(f"🕐 {time.strftime('%H:%M:%S')} | 📊 T={t:.1f}°C H={h:.1f}% G={g:.0f}")
            print(f"🌬️ AQI: {int(aqi)} ({aqi_info['name']})")
            print(f"🤖 RF: {ml['current']['label']} ({ml['current']['confidence']:.0f}%) | "
                  f"LSTM: {ml['future']['label']} ({ml['future']['confidence']:.0f}%)")
            print(f"🏥 {alert['name']}: {alert['advice']}")
            print(f"📦 Mode: MOCK | Reject: {outlier_stats['rejection_rate']:.1f}%")
            print(f"{'-'*50}")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print_final_stats(db, outlier_detector)


def run_real_mode(ser, db, predictor, outlier_detector, buffer):
    """Run in real sensor mode"""
    print("\n[4] Running in REAL SENSOR mode")
    print("    Waiting for data from ESP32...\n")
    
    try:
        while True:
            line = ser.readline()
            reading = process_line(line)
            
            if reading is None:
                continue
            
            result, errors = outlier_detector.add_reading(reading[0], reading[1], reading[2])
            
            if result is None:
                continue
            
            buffer.append(reading)
            t, h, g = reading[0], reading[1], reading[2]
            
            if len(buffer) < 10:
                print(f"    Buffer: {len(buffer)}/10...")
                continue
            
            # Process reading
            aqi = calculate_aqi(pm25=g * 0.15)
            aqi_info = get_aqi_info(aqi)
            predictor.add_reading(t, h, g)
            ml = predictor.predict_all(t, h, g)
            alert = get_alert(aqi)
            
            # Save to DB
            db.add_reading(t, h, g, aqi, aqi_info['name'])
            db.add_prediction(ml['current']['label'], ml['current']['confidence'],
                          ml['future']['label'], ml['future']['confidence'],
                          ml['trend']['trend'], ml['trend']['aqi'])
            
            outlier_stats = outlier_detector.get_stats()
            print(f"\n{'-'*50}")
            print(f"🕐 {time.strftime('%H:%M:%S')} | 📊 T={t:.1f}°C H={h:.1f}% G={g:.0f}")
            print(f"🌬️ AQI: {int(aqi)} ({aqi_info['name']})")
            print(f"🤖 RF: {ml['current']['label']} ({ml['current']['confidence']:.0f}%) | "
                  f"LSTM: {ml['future']['label']} ({ml['future']['confidence']:.0f}%)")
            print(f"🏥 {alert['name']}: {alert['advice']}")
            print(f"📦 Mode: REAL | Reject: {outlier_stats['rejection_rate']:.1f}%")
            print(f"{'-'*50}")
    
    except serial.SerialException:
        print("    ⚠️ Disconnected, reconnecting...")
        time.sleep(2)
        main()  # Restart to re-detect


def print_final_stats(db, outlier_detector):
    """Print final statistics"""
    stats = db.get_statistics()
    outlier = outlier_detector.get_stats()
    print(f"\n\n📊 Final Stats:")
    print(f"  Readings: {stats['readings']}")
    print(f"  Alerts: {stats['alerts']}")
    print(f"  Outlier Rejection: {outlier['rejection_rate']:.1f}%")
    print("\n👋 Shutting down...")


if __name__ == '__main__':
    main()
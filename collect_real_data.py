# collect_real_data.py
"""
Real Data Collection Script
Fetches air quality and weather data from Open-Meteo API
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

NAGPUR_LAT = 21.1458
NAGPUR_LON = 79.0882

def fetch_air_quality_data(days_past=30, days_forecast=2):
    """Fetch historical and forecasted air quality data"""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": NAGPUR_LAT,
        "longitude": NAGPUR_LON,
        "hourly": "pm2_5,pm10,carbon_dioxide,nitrogen_dioxide,ozone,temperature_2m,relative_humidity_2m",
        "timezone": "auto",
        "past_days": days_past,
        "forecast_days": days_forecast
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}")

def fetch_weather_data(days_past=30):
    """Fetch weather data (temperature, humidity)"""
    url = "https://open-meteo.com/v1/forecast"
    params = {
        "latitude": NAGPUR_LAT,
        "longitude": NAGPUR_LON,
        "hourly": "temperature_2m,relative_humidity_2m",
        "past_days": days_past,
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}")

def calculate_aqi_pm25(pm25):
    """Calculate AQI from PM2.5 using India CPCB formula"""
    if pm25 <= 12:
        return pm25 * 50 / 12
    elif pm25 <= 35.4:
        return 50 + (pm25 - 12) * 50 / 23.4
    elif pm25 <= 55.4:
        return 100 + (pm25 - 35.4) * 50 / 20
    elif pm25 <= 150.4:
        return 150 + (pm25 - 55.4) * 100 / 95
    elif pm25 <= 250.4:
        return 200 + (pm25 - 150.4) * 100 / 100
    elif pm25 <= 350.4:
        return 300 + (pm25 - 250.4) * 100 / 100
    elif pm25 <= 500.4:
        return 400 + (pm25 - 350.4) * 100 / 150
    else:
        return 500

def calculate_aqi_pm10(pm10):
    """Calculate AQI from PM10 using India CPCB formula"""
    if pm10 <= 54:
        return pm10 * 50 / 54
    elif pm10 <= 154:
        return 50 + (pm10 - 54) * 50 / 100
    elif pm10 <= 254:
        return 100 + (pm10 - 154) * 100 / 100
    elif pm10 <= 354:
        return 200 + (pm10 - 254) * 100 / 100
    elif pm10 <= 424:
        return 300 + (pm10 - 354) * 100 / 70
    elif pm10 <= 604:
        return 400 + (pm10 - 424) * 100 / 180
    else:
        return 500

def calculate_composite_aqi(pm25, pm10):
    """Calculate composite AQI (worst of PM2.5 or PM10)"""
    aqi_pm25 = calculate_aqi_pm25(pm25)
    aqi_pm10 = calculate_aqi_pm10(pm10)
    return max(aqi_pm25, aqi_pm10)

def get_aqi_category(aqi):
    """Get AQI category label"""
    if aqi <= 50:
        return 0  # Good
    elif aqi <= 100:
        return 1  # Moderate
    elif aqi <= 200:
        return 2  # Poor
    elif aqi <= 300:
        return 3  # Very Poor
    elif aqi <= 400:
        return 4  # Severe
    else:
        return 5  # Hazardous

def estimate_gas_from_pollutants(pm25, pm10, co2, no2, ozone):
    """Estimate MQ-135 gas reading from pollutants"""
    gas = (pm25 * 2.5) + (pm10 * 1.5) + (co2 * 0.3) + (no2 * 2.0) + (ozone * 0.8)
    gas = gas / 5
    return max(50, min(1200, gas + np.random.normal(0, 20)))

def process_hourly_data(data):
    """Process hourly data into DataFrame"""
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    pm25 = hourly.get('pm2_5', [np.nan] * len(times))
    pm10 = hourly.get('pm10', [np.nan] * len(times))
    co2 = hourly.get('carbon_dioxide', [np.nan] * len(times))
    no2 = hourly.get('nitrogen_dioxide', [np.nan] * len(times))
    ozone_vals = hourly.get('ozone', [np.nan] * len(times))
    temp = hourly.get('temperature_2m', [np.nan] * len(times))
    hum = hourly.get('relative_humidity_2m', [np.nan] * len(times))
    
    rows = []
    for i in range(len(times)):
        pm25_val = pm25[i] if pm25[i] is not None else np.nan
        pm10_val = pm10[i] if pm10[i] is not None else np.nan
        
        if pd.isna(pm25_val) or pd.isna(pm10_val):
            continue
        
        co2_val = co2[i] if co2[i] is not None else 420
        no2_val = no2[i] if no2[i] is not None else 15
        ozone_val = ozone_vals[i] if ozone_vals[i] is not None else 50
        temp_val = temp[i] if temp[i] is not None else 25.0
        hum_val = hum[i] if hum[i] is not None else 50.0
            
        aqi = calculate_composite_aqi(pm25_val, pm10_val)
        status = get_aqi_category(aqi)
        gas = estimate_gas_from_pollutants(pm25_val, pm10_val, co2_val, no2_val, ozone_val)
        
        row = {
            'timestamp': times[i],
            'temp': temp_val,
            'hum': hum_val,
            'gas': gas,
            'pm25': pm25_val,
            'pm10': pm10_val,
            'co2': co2_val,
            'no2': no2_val,
            'ozone': ozone_val,
            'aqi': aqi,
            'status': status
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def collect_and_save(output_file='real_air_data.csv', days_past=30):
    """Main function to collect and save data"""
    print("Fetching air quality data from Open-Meteo API...")
    
    try:
        data = fetch_air_quality_data(days_past=days_past, days_forecast=2)
        print(f"Retrieved {len(data.get('hourly', {}).get('time', []))} hourly records")
        
        df = process_hourly_data(data)
        
        print(f"Processed {len(df)} valid records")
        
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
        
        print("\n=== Data Summary ===")
        print(f"Temperature: {df['temp'].min():.1f} - {df['temp'].max():.1f} °C")
        print(f"Humidity: {df['hum'].min():.1f} - {df['hum'].max():.1f} %")
        print(f"Gas (simulated): {df['gas'].min():.0f} - {df['gas'].max():.0f}")
        print(f"PM2.5: {df['pm25'].min():.1f} - {df['pm25'].max():.1f} μg/m³")
        print(f"PM10: {df['pm10'].min():.1f} - {df['pm10'].max():.1f} μg/m³")
        print(f"AQI: {df['aqi'].min():.0f} - {df['aqi'].max():.0f}")
        
        status_counts = df['status'].value_counts().sort_index()
        print("\nStatus Distribution:")
        status_labels = {0: 'Good', 1: 'Moderate', 2: 'Poor', 3: 'Very Poor', 4: 'Severe', 5: 'Hazardous'}
        for status, count in status_counts.items():
            print(f"  {status_labels.get(status, status)}: {count}")
        
        return df
        
    except Exception as e:
        print(f"Error collecting data: {e}")
        return None

if __name__ == "__main__":
    df = collect_and_save('real_air_data.csv', days_past=30)
    if df is not None:
        print("\n✅ Data collection complete!")
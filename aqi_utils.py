# aqi_utils.py
"""
AQI Calculation Module
Based on India CPCB / US EPA standards
"""

import numpy as np

# AQI Breakpoints (India CPCB Standard)
AQI_BREAKPOINTS = {
    'pm25': [
        (0, 12, 0, 50),
        (12, 35.4, 51, 100),
        (35.4, 55.4, 101, 150),
        (55.4, 150.4, 151, 200),
        (150.4, 250.4, 201, 300),
        (250.4, 500, 301, 500),
    ],
    'pm10': [
        (0, 54, 0, 50),
        (54, 154, 51, 100),
        (154, 254, 101, 200),
        (254, 354, 201, 300),
        (354, 424, 301, 400),
        (424, 604, 401, 500),
    ]
}

# AQI Categories
AQI_CATEGORIES = {
    0: {'name': 'Good', 'color': '#00E400', 'emoji': '🟢'},
    1: {'name': 'Moderate', 'color': '#FFFF00', 'emoji': '🟡'},
    2: {'name': 'Poor', 'color': '#FF7E00', 'emoji': '🟠'},
    3: {'name': 'Very Poor', 'color': '#FF0000', 'emoji': '🔴'},
    4: {'name': 'Severe', 'color': '#8F3F97', 'emoji': '🟣'},
    5: {'name': 'Hazardous', 'color': '#7E0023', 'emoji': '⚫'},
}


def calculate_aqi_from_concentration(concentration, pollutant_type='pm25'):
    """Calculate AQI from pollutant concentration"""
    if pollutant_type not in AQI_BREAKPOINTS:
        raise ValueError(f"Unknown pollutant: {pollutant_type}")
    
    breakpoints = AQI_BREAKPOINTS[pollutant_type]
    
    if concentration <= breakpoints[0][1]:
        return concentration * 50 / breakpoints[0][1]
    
    for i, (c_low, c_high, aqi_low, aqi_high) in enumerate(breakpoints):
        if c_low < concentration <= c_high:
            return aqi_low + (concentration - c_low) * (aqi_high - aqi_low) / (c_high - c_low)
    
    return 500


def calculate_aqi(pm25=None, pm10=None):
    """Calculate composite AQI from PM2.5 and/or PM10"""
    aqi_values = []
    
    if pm25 is not None and pm25 > 0:
        aqi_pm25 = calculate_aqi_from_concentration(pm25, 'pm25')
        aqi_values.append(aqi_pm25)
    
    if pm10 is not None and pm10 > 0:
        aqi_pm10 = calculate_aqi_from_concentration(pm10, 'pm10')
        aqi_values.append(aqi_pm10)
    
    if not aqi_values:
        return 0
    
    return max(aqi_values)


def get_aqi_category(aqi):
    """Get AQI category from AQI value"""
    if aqi <= 50:
        return 0
    elif aqi <= 100:
        return 1
    elif aqi <= 200:
        return 2
    elif aqi <= 300:
        return 3
    elif aqi <= 400:
        return 4
    else:
        return 5


def get_aqi_info(aqi):
    """Get full AQI information"""
    category = get_aqi_category(aqi)
    info = AQI_CATEGORIES.get(category, AQI_CATEGORIES[5])
    return {
        'aqi': round(aqi, 1),
        'category': category,
        'name': info['name'],
        'color': info['color'],
        'emoji': info['emoji']
    }


def estimate_gas_from_sensors(temp, hum, raw_gas=None):
    """Estimate gas level from temperature and humidity
    MQ-135 sensor simulation based on environmental factors
    """
    base_gas = 100
    
    temp_factor = (temp - 20) * 2 if temp > 20 else (temp - 20) * 0.5
    hum_factor = (hum - 50) * 0.5 if hum > 50 else (hum - 50) * 0.3
    
    estimated_gas = base_gas + temp_factor + hum_factor + np.random.normal(0, 10)
    
    if raw_gas:
        estimated_gas = (estimated_gas + raw_gas) / 2
    
    return max(50, min(1000, estimated_gas))


def predict_aqi_from_trend(current_aqi, trend_direction, hours=1):
    """Predict future AQI based on trend"""
    if trend_direction == 'rising':
        return current_aqi + (hours * 5)
    elif trend_direction == 'falling':
        return max(0, current_aqi - (hours * 3))
    else:
        return current_aqi


def get_air_quality_status(temp, hum, gas):
    """Get air quality status from sensor readings
    Simplified estimation without PM sensor
    """
    score = 0
    
    if temp < 20 or temp > 30:
        score += 20
    elif temp < 22 or temp > 28:
        score += 10
    
    if hum < 40 or hum > 70:
        score += 20
    elif hum < 45 or hum > 65:
        score += 10
    
    if gas > 400:
        score += 40
    elif gas > 300:
        score += 30
    elif gas > 200:
        score += 20
    elif gas > 150:
        score += 10
    
    if score <= 20:
        return 0
    elif score <= 40:
        return 1
    else:
        return 2


class AQIAnalyzer:
    """AQI Analyzer for real-time analysis"""
    
    def __init__(self):
        self.history = []
        self.max_history = 50
    
    def add_reading(self, temp, hum, gas):
        """Add a sensor reading"""
        reading = {
            'temp': temp,
            'hum': hum,
            'gas': gas,
            'pm25_estimated': self._estimate_pm25(gas)
        }
        self.history.append(reading)
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def _estimate_pm25(self, gas):
        """Estimate PM2.5 from gas reading"""
        return gas * 0.15 + np.random.normal(0, 5)
    
    def get_current_aqi(self):
        """Get current AQI"""
        if not self.history:
            return 0
        pm25 = self.history[-1].get('pm25_estimated', 0)
        return calculate_aqi(pm25=pm25)
    
    def get_trend(self):
        """Get AQI trend direction"""
        if len(self.history) < 5:
            return 'stable'
        
        recent = [r.get('pm25_estimated', 0) for r in self.history[-5:]]
        diff = (recent[-1] - recent[0]) / 5
        
        if diff > 3:
            return 'rising'
        elif diff < -3:
            return 'falling'
        else:
            return 'stable'
    
    def reset(self):
        """Reset history"""
        self.history = []


# ============================================================
# OUTLIER DETECTION MODULE
# ============================================================

# Physical limits for sensors (impossible values)
SENSOR_LIMITS = {
    'temp': {'min': -10, 'max': 85, 'min_valid': 10, 'max_valid': 55},  # expanded for heat source
    'hum': {'min': 0, 'max': 100, 'min_valid': 15, 'max_valid': 95},
    'gas': {'min': 0, 'max': 2000, 'min_valid': 30, 'max_valid': 1500},
}

# Rate of change limits (per reading) - relaxed for real sensor spikes
RATE_LIMITS = {
    'temp': {'max': 15.0},   # max 15°C change per reading (was 5)
    'hum': {'max': 30.0},   # max 30% change per reading (was 15)
    'gas': {'max': 500.0},  # max 500 units change per reading (was 100)
}


def validate_reading(temp, hum, gas):
    """
    Validate sensor readings for impossible values.
    Returns (is_valid, errors)
    """
    errors = []
    
    # Check physical limits
    if temp < SENSOR_LIMITS['temp']['min'] or temp > SENSOR_LIMITS['temp']['max']:
        errors.append(f"Temp {temp}°C out of physical range")
    
    if hum < SENSOR_LIMITS['hum']['min'] or hum > SENSOR_LIMITS['hum']['max']:
        errors.append(f"Humidity {hum}% out of physical range")
    
    if gas < SENSOR_LIMITS['gas']['min'] or gas > SENSOR_LIMITS['gas']['max']:
        errors.append(f"Gas {gas} out of physical range")
    
    return len(errors) == 0, errors


def validate_rate_of_change(current, previous):
    """
    Validate rate of change between readings.
    Returns (is_valid, warnings)
    """
    warnings = []
    
    for key in ['temp', 'hum', 'gas']:
        if key in current and key in previous:
            diff = abs(current[key] - previous[key])
            limit = RATE_LIMITS[key]['max']
            
            if diff > limit:
                warnings.append(f"{key} changed by {diff:.1f} (max: {limit})")
    
    return len(warnings) == 0, warnings


class OutlierDetector:
    """Detect and filter sensor outliers"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []
        self.rejected_count = 0
        self.total_count = 0
    
    def add_reading(self, temp, hum, gas):
        """Add reading and check for outliers"""
        self.total_count += 1
        
        # Check absolute limits
        valid, errors = validate_reading(temp, hum, gas)
        if not valid:
            self.rejected_count += 1
            return None, errors
        
        # Check rate of change if we have history
        if self.history:
            prev = self.history[-1]
            valid, warnings = validate_rate_of_change(
                {'temp': temp, 'hum': hum, 'gas': gas},
                prev
            )
            if not valid:
                # Warning but don't reject - just log
                pass
        
        # Add to history
        reading = {'temp': temp, 'hum': hum, 'gas': gas}
        self.history.append(reading)
        
        # Trim history
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        return reading, []
    
    def get_stats(self):
        """Get outlier detection statistics"""
        if self.total_count == 0:
            return {'rejection_rate': 0, 'rejected': 0, 'total': 0}
        
        return {
            'rejection_rate': self.rejected_count / self.total_count * 100,
            'rejected': self.rejected_count,
            'total': self.total_count
        }
    
    def reset(self):
        """Reset detector"""
        self.history.clear()
        self.rejected_count = 0
        self.total_count = 0
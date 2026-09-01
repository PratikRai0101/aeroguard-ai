# predictors.py
"""
Unified ML Pipeline for Air Quality Prediction
Orchestrates RF, LSTM, and Linear Regression models
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from collections import deque

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class AQIPredictor:
    """Unified AQI Predictor using Multiple Models"""
    
    def __init__(self, model_dir='.'):
        self.model_dir = model_dir
        self.rf_model = None
        self.lr_model = None
        self.lstm_model = None
        self.scaler = None
        self.buffer = deque(maxlen=10)
        self.feature_cols = ['temp', 'hum', 'gas']
        
        self._load_models()
    
    def _load_models(self):
        """Load all models"""
        print("Loading ML models...")
        
        # Load Random Forest
        try:
            self.rf_model = joblib.load(f'{self.model_dir}/rf_air_model.pkl')
            print("  ✓ Random Forest loaded")
        except Exception as e:
            print(f"  ✗ RF load error: {e}")
        
        # Load Linear Regression
        try:
            self.lr_model = joblib.load(f'{self.model_dir}/lr_trend_model.pkl')
            print("  ✓ Linear Regression loaded")
        except Exception as e:
            print(f"  ✗ LR load error: {e}")
        
        # Load LSTM
        if TF_AVAILABLE:
            try:
                self.lstm_model = tf.keras.models.load_model(f'{self.model_dir}/lstm_air_model.h5')
                print("  ✓ LSTM loaded")
            except Exception as e:
                print(f"  ✗ LSTM load error: {e}")
        
        # Load Scaler
        try:
            with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("  ✓ Scaler loaded")
        except Exception as e:
            print(f"  ✗ Scaler load error: {e}")
    
    def add_reading(self, temp, hum, gas):
        """Add a reading to the buffer"""
        self.buffer.append([temp, hum, gas])
    
    def predict_current(self, temp, hum, gas):
        """Predict current air quality using Random Forest"""
        if self.rf_model is None:
            return {'status': 1, 'label': 'Moderate', 'confidence': 0}
        
        X = pd.DataFrame([[temp, hum, gas, 25.0]], 
                     columns=['temp', 'hum', 'gas', 'pm25'])
        
        pred = self.rf_model.predict(X)[0]
        prob = self.rf_model.predict_proba(X)[0]
        
        labels = {0: 'Good', 1: 'Moderate', 2: 'Poor', 3: 'Very Poor'}
        
        return {
            'status': int(pred),
            'label': labels.get(pred, 'Unknown'),
            'confidence': float(max(prob)) * 100
        }
    
    def predict_future_lstm(self):
        """Predict future status using LSTM"""
        if self.lstm_model is None or self.scaler is None:
            return {'status': 1, 'label': 'Moderate', 'confidence': 0}
        
        if len(self.buffer) < 10:
            return {'status': None, 'label': 'Buffer filling...', 'confidence': 0}
        
        arr = np.array(list(self.buffer))
        
        try:
            arr_scaled = self.scaler.transform(arr)
            arr_scaled = arr_scaled.reshape(1, 10, 3)
            probs = self.lstm_model.predict(arr_scaled, verbose=0)[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred]) * 100
            
            labels = {0: 'Good', 1: 'Moderate', 2: 'Poor'}
            
            return {
                'status': pred,
                'label': labels.get(pred, 'Unknown'),
                'confidence': conf
            }
        except Exception as e:
            return {'status': 1, 'label': 'Error', 'confidence': 0}
    
    def predict_trend_lr(self, temp, hum, gas):
        """Predict AQI trend using Linear Regression"""
        if self.lr_model is None:
            return {'aqi': 0, 'trend': 'stable'}
        
        X = pd.DataFrame([[temp, hum, gas, 25.0]], 
                     columns=['temp', 'hum', 'gas', 'pm25'])
        
        aqi = self.lr_model.predict(X)[0]
        
        if len(self.buffer) >= 3:
            recent = np.array(list(self.buffer)[-3:])
            if recent[-1][2] > recent[0][2] + 10:
                trend = 'rising'
            elif recent[-1][2] < recent[0][2] - 10:
                trend = 'falling'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'aqi': max(0, round(aqi, 1)),
            'trend': trend
        }
    
    def predict_all(self, temp, hum, gas):
        """Run all predictions and return comprehensive result"""
        current = self.predict_current(temp, hum, gas)
        future = self.predict_future_lstm()
        trend = self.predict_trend_lr(temp, hum, gas)
        
        self.add_reading(temp, hum, gas)
        
        return {
            'current': current,
            'future': future,
            'trend': trend,
            'buffer_size': len(self.buffer)
        }
    
    def reset(self):
        """Reset buffer"""
        self.buffer.clear()


def create_predictor(model_dir='.'):
    """Create a predictor instance"""
    return AQIPredictor(model_dir)


# Test function
def test_predictor():
    """Test the predictor"""
    predictor = AQIPredictor('.')
    
    test_readings = [
        (25, 55, 150),
        (26, 56, 155),
        (25, 54, 160),
        (27, 55, 165),
        (26, 56, 170),
        (25, 55, 175),
        (26, 54, 180),
        (27, 55, 185),
        (26, 56, 190),
        (25, 55, 195),
    ]
    
    for temp, hum, gas in test_readings:
        predictor.add_reading(temp, hum, gas)
        
        if len(predictor.buffer) >= 10:
            result = predictor.predict_all(temp, hum, gas)
            
            print(f"\nInput: T={temp}°C, H={hum}%, G={gas}")
            print(f"  Current: {result['current']['label']} ({result['current']['confidence']:.1f}%)")
            print(f"  Future: {result['future']['label']} ({result['future']['confidence']:.1f}%)")
            print(f"  Trend: {result['trend']['aqi']} ({result['trend']['trend']})")


if __name__ == '__main__':
    test_predictor()
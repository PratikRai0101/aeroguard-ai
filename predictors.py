# predictors.py
"""
Unified ML Pipeline for Air Quality Prediction.

Orchestrates the Random Forest (classification), LSTM (time-series) and
Linear Regression (trend) models.

Feature contract
----------------
Every model consumes exactly the features in :data:`preprocessing.FEATURE_COLS`
(``temp``, ``hum``, ``gas``). The sensor node cannot measure PM2.5, so it is
deliberately not a model input. Training and inference both go through this
module so the feature order can never drift.
"""

import json
import pickle
from collections import deque

import numpy as np

# IMPORT ORDER MATTERS: on macOS, importing pandas/sklearn before TensorFlow
# causes TensorFlow's fit/predict paths to deadlock. TensorFlow must be
# imported first. Do not reorder these imports.
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:  # pragma: no cover - tensorflow is optional
    TF_AVAILABLE = False

import pandas as pd
import joblib

from preprocessing import (
    AQI_LABELS,
    FEATURE_COLS,
    features_from_reading,
    label_for,
    transform_features,
)


METADATA_FILE = 'model_metadata.json'


class AQIPredictor:
    """Unified AQI predictor using multiple models."""

    def __init__(self, model_dir='.'):
        self.model_dir = model_dir
        self.rf_model = None
        self.lr_model = None
        self.lstm_model = None
        self.scaler = None
        self.buffer = deque(maxlen=10)

        # Defaults; overridden by model_metadata.json when present.
        self.feature_cols = list(FEATURE_COLS)
        self.labels = dict(AQI_LABELS)
        self.metadata = {}

        self._load_metadata()
        self._load_models()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_metadata(self):
        """Load the training metadata so feature order cannot drift."""
        try:
            with open(f'{self.model_dir}/{METADATA_FILE}', 'r') as handle:
                self.metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            self.metadata = {}
            return

        meta_features = self.metadata.get('feature_cols')
        if meta_features:
            self.feature_cols = list(meta_features)

        meta_labels = self.metadata.get('labels')
        if meta_labels:
            # JSON keys are strings; normalise back to ints.
            self.labels = {int(k): v for k, v in meta_labels.items()}

    def _load_models(self):
        """Load all models from disk."""
        print("Loading ML models...")

        try:
            self.rf_model = joblib.load(f'{self.model_dir}/rf_air_model.pkl')
            print("  ✓ Random Forest loaded")
        except Exception as exc:
            print(f"  ✗ RF load error: {exc}")

        try:
            self.lr_model = joblib.load(f'{self.model_dir}/lr_trend_model.pkl')
            print("  ✓ Linear Regression loaded")
        except Exception as exc:
            print(f"  ✗ LR load error: {exc}")

        if TF_AVAILABLE:
            try:
                self.lstm_model = tf.keras.models.load_model(
                    f'{self.model_dir}/lstm_air_model.h5'
                )
                print("  ✓ LSTM loaded")
            except Exception as exc:
                print(f"  ✗ LSTM load error: {exc}")

        try:
            with open(f'{self.model_dir}/scaler.pkl', 'rb') as handle:
                self.scaler = pickle.load(handle)
            print("  ✓ Scaler loaded")
        except Exception as exc:
            print(f"  ✗ Scaler load error: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _features(self, temp, hum, gas):
        """Validate a reading and build a model-ready feature frame."""
        frame, errors = features_from_reading(
            temp, hum, gas, feature_cols=self.feature_cols
        )
        return frame, errors

    def _expected_feature_count(self, model):
        """Return the feature count a fitted sklearn model expects."""
        count = getattr(model, 'n_features_in_', None)
        return int(count) if count is not None else None

    def _feature_count_ok(self, model):
        expected = self._expected_feature_count(model)
        return expected is None or expected == len(self.feature_cols)

    def add_reading(self, temp, hum, gas):
        """Append a reading to the time-series buffer."""
        frame, _ = features_from_reading(
            temp, hum, gas, feature_cols=self.feature_cols
        )
        if frame is not None:
            self.buffer.append([float(temp), float(hum), float(gas)])

    def reset(self):
        """Clear the time-series buffer."""
        self.buffer.clear()

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def predict_current(self, temp, hum, gas):
        """Classify current air quality with the Random Forest."""
        if self.rf_model is None:
            return {'status': 1, 'label': 'Moderate', 'confidence': 0,
                    'class_probabilities': {}, 'error': 'Random Forest not loaded'}

        X, errors = self._features(temp, hum, gas)
        if X is None:
            return {'status': None, 'label': 'Invalid reading', 'confidence': 0,
                    'class_probabilities': {}, 'error': '; '.join(errors)}

        if not self._feature_count_ok(self.rf_model):
            return {
                'status': None,
                'label': 'Model mismatch',
                'confidence': 0,
                'class_probabilities': {},
                'error': (
                    f"RF expects {self._expected_feature_count(self.rf_model)} features "
                    f"but {len(self.feature_cols)} were supplied. Retrain the models."
                ),
            }

        prediction = int(self.rf_model.predict(X)[0])
        probabilities = self.rf_model.predict_proba(X)[0]
        classes = [int(c) for c in self.rf_model.classes_]

        # Map the predicted class to its own probability column. This is
        # robust to class sets that are not contiguous or do not start at 0.
        if prediction in classes:
            confidence = float(probabilities[classes.index(prediction)])
        else:
            confidence = float(np.max(probabilities))

        return {
            'status': prediction,
            'label': label_for(prediction),
            'confidence': confidence * 100,
            # NB: this is a class probability, not a validated reliability
            # measure. Phase 3 adds a residual-based reliability indicator.
            'class_probabilities': {
                label_for(cls): round(float(prob), 4)
                for cls, prob in zip(classes, probabilities)
            },
        }

    def predict_future_lstm(self):
        """Predict the next status with the LSTM."""
        if self.lstm_model is None or self.scaler is None:
            return {'status': None, 'label': 'Moderate', 'confidence': 0,
                    'error': 'LSTM or scaler not loaded'}

        if len(self.buffer) < self.buffer.maxlen:
            return {'status': None, 'label': 'Buffer filling...', 'confidence': 0}

        arr = np.array(list(self.buffer), dtype=float)

        try:
            arr_scaled = self.scaler.transform(arr)
            arr_scaled = arr_scaled.reshape(1, arr.shape[0], arr.shape[1])
            probs = self.lstm_model.predict(arr_scaled, verbose=0)[0]
            predicted = int(np.argmax(probs))
            return {
                'status': predicted,
                'label': label_for(predicted),
                'confidence': float(probs[predicted]) * 100,
            }
        except Exception as exc:
            return {'status': None, 'label': 'Error', 'confidence': 0,
                    'error': str(exc)}

    def predict_trend_lr(self, temp, hum, gas):
        """Predict the AQI value and direction with Linear Regression."""
        if self.lr_model is None:
            return {'aqi': 0, 'trend': 'stable', 'error': 'Linear Regression not loaded'}

        X, errors = self._features(temp, hum, gas)
        if X is None:
            return {'aqi': 0, 'trend': 'stable', 'error': '; '.join(errors)}

        if not self._feature_count_ok(self.lr_model):
            return {
                'aqi': 0,
                'trend': 'stable',
                'error': (
                    f"LR expects {self._expected_feature_count(self.lr_model)} features "
                    f"but {len(self.feature_cols)} were supplied. Retrain the models."
                ),
            }

        aqi = float(self.lr_model.predict(X)[0])

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

        return {'aqi': max(0.0, round(aqi, 1)), 'trend': trend}

    def predict_all(self, temp, hum, gas):
        """Run all models and return a combined result."""
        current = self.predict_current(temp, hum, gas)
        future = self.predict_future_lstm()
        trend = self.predict_trend_lr(temp, hum, gas)

        self.add_reading(temp, hum, gas)

        return {
            'current': current,
            'future': future,
            'trend': trend,
            'buffer_size': len(self.buffer),
        }


def create_predictor(model_dir='.'):
    """Create a predictor instance."""
    return AQIPredictor(model_dir)


def test_predictor():  # pragma: no cover - manual smoke test
    """Smoke-test the predictor with a rising gas sequence."""
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
        result = predictor.predict_all(temp, hum, gas)

        if len(predictor.buffer) >= predictor.buffer.maxlen:
            print(f"\nInput: T={temp}°C, H={hum}%, G={gas}")
            print(f"  Current: {result['current']['label']} "
                  f"({result['current']['confidence']:.1f}%)")
            print(f"  Future:  {result['future']['label']} "
                  f"({result['future']['confidence']:.1f}%)")
            print(f"  Trend:   {result['trend']['aqi']} "
                  f"({result['trend']['trend']})")


if __name__ == '__main__':
    test_predictor()

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

from calibration import GasCalibrator
from explain import SHAP_AVAILABLE, ModelExplainer
from preprocessing import (
    AQI_LABELS,
    FEATURE_COLS,
    features_from_reading,
    label_for,
    transform_features,
)
import reliability
import risk as risk_assessment


METADATA_FILE = 'model_metadata.json'


class AQIPredictor:
    """Unified AQI predictor using multiple models."""

    def __init__(self, model_dir='.'):
        self.model_dir = model_dir
        self.rf_model = None
        self.lr_model = None
        self.lstm_model = None
        self.scaler = None
        self.calibrator = None
        self.buffer = deque(maxlen=10)

        # Defaults; overridden by model_metadata.json when present.
        self.feature_cols = list(FEATURE_COLS)
        self.labels = dict(AQI_LABELS)
        self.metadata = {}
        self.residual_std = None
        self.explainer = None

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

        # Residual std from validation powers the reliability estimate.
        residual_std = self.metadata.get('lr', {}).get('residual_std')
        if residual_std is not None:
            self.residual_std = float(residual_std)

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

        try:
            self.calibrator = GasCalibrator.load(f'{self.model_dir}/gas_calibrator.pkl')
            print("  ✓ Gas calibrator loaded")
        except Exception:
            self.calibrator = None
            print("  • No gas calibrator found (raw gas will be used)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _calibrate_gas(self, temp, hum, gas):
        """Apply the fitted gas calibration (raw response -> VOC concentration)."""
        if self.calibrator is not None and self.calibrator.is_fitted():
            try:
                return float(self.calibrator.transform(gas, temp, hum)[0])
            except Exception:
                return float(gas)
        return float(gas)

    def _features(self, temp, hum, gas):
        """Validate a reading, calibrate the gas channel, build a feature frame."""
        calibrated_gas = self._calibrate_gas(temp, hum, gas)
        frame, errors = features_from_reading(
            temp, hum, calibrated_gas, feature_cols=self.feature_cols
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
        """Append a reading to the time-series buffer (calibrated gas)."""
        calibrated_gas = self._calibrate_gas(temp, hum, gas)
        frame, _ = features_from_reading(
            temp, hum, calibrated_gas, feature_cols=self.feature_cols
        )
        if frame is not None:
            self.buffer.append([float(temp), float(hum), calibrated_gas])

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

    def explain_current(self, temp, hum, gas, top_k=3):
        """Explain the current Random Forest prediction with TreeSHAP."""
        if self.rf_model is None or not SHAP_AVAILABLE:
            return None

        X, errors = self._features(temp, hum, gas)
        if X is None:
            return None

        predicted = int(self.rf_model.predict(X)[0])
        if self.explainer is None:
            self.explainer = ModelExplainer(self.rf_model, self.feature_cols)
        return self.explainer.explain(X.values, predicted, top_k=top_k)

    def predict_future_from(self, rows):
        """
        Predict the next status from an explicit list of raw readings.

        ``rows`` is a chronological list of ``(temp, hum, gas)`` tuples. This
        is the stateless counterpart of :meth:`predict_future_lstm`; it does
        not touch the internal buffer, so API callers can poll safely.
        """
        if self.lstm_model is None or self.scaler is None:
            return {'status': None, 'label': 'Moderate', 'confidence': 0,
                    'error': 'LSTM or scaler not loaded'}

        window = self.buffer.maxlen
        if len(rows) < window:
            return {'status': None, 'label': 'Buffer filling...', 'confidence': 0}

        recent = rows[-window:]
        calibrated = [
            [float(temp), float(hum), self._calibrate_gas(temp, hum, gas)]
            for temp, hum, gas in recent
        ]
        arr = np.array(calibrated, dtype=float)

        try:
            arr_scaled = self.scaler.transform(arr).reshape(1, window, arr.shape[1])
            probs = self.lstm_model.predict(arr_scaled, verbose=0)[0]
            predicted = int(np.argmax(probs))
            return {
                'status': predicted,
                'label': label_for(predicted),
                'confidence': float(probs[predicted]) * 100,
            }
        except Exception as exc:
            return {'status': None, 'label': 'Error', 'confidence': 0, 'error': str(exc)}

    def _compose_assessment(self, temp, hum, gas, current, future, trend):
        """Add reliability, risk and SHAP explanation to model outputs."""
        reliability_report = None
        if self.residual_std:
            reliability_report = reliability.assess(trend['aqi'], self.residual_std)

        risk_report = risk_assessment.assess_risk(
            trend['aqi'],
            temp=temp,
            hum=hum,
            trend=trend['trend'],
            reliability=reliability_report,
        )

        return {
            'current': current,
            'future': future,
            'trend': trend,
            'reliability': reliability_report,
            'risk': risk_report,
            'explanation': self.explain_current(temp, hum, gas),
        }

    def predict_all(self, temp, hum, gas):
        """Run all models (stateful streaming variant) and return a result."""
        current = self.predict_current(temp, hum, gas)
        future = self.predict_future_lstm()
        trend = self.predict_trend_lr(temp, hum, gas)

        result = self._compose_assessment(temp, hum, gas, current, future, trend)

        self.add_reading(temp, hum, gas)
        result['buffer_size'] = len(self.buffer)
        return result

    def assess(self, temp, hum, gas, history_rows=None):
        """
        Stateless full assessment, for API callers that poll.

        Parameters
        ----------
        temp, hum, gas : float
            The latest reading.
        history_rows : list[tuple], optional
            Chronological ``(temp, hum, gas)`` rows used for the LSTM window.

        Returns the same keys as :meth:`predict_all`, without mutating state.
        """
        history_rows = list(history_rows or [])
        current = self.predict_current(temp, hum, gas)
        trend = self.predict_trend_lr(temp, hum, gas)
        future = self.predict_future_from(history_rows + [(temp, hum, gas)])

        result = self._compose_assessment(temp, hum, gas, current, future, trend)
        result['buffer_size'] = len(history_rows) + 1
        return result


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

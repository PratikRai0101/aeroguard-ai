# train_model.py
"""
Train the Random Forest (classification) and Linear Regression (trend) models.

Pipeline
--------
    dataset -> chronological split -> gas calibration -> preprocessing
            -> train -> validate against reference -> metadata

The target is the VOC air-quality sub-index (benzene-equivalent), which is
what the MQ-135-class sensor can actually measure. See dataset.py.
"""

import json
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import AQI_LABELS, FEATURE_COLS, preprocess_dataframe
from training import (
    build_metadata,
    chronological_split,
    fit_and_apply_calibration,
    prepare_dataset,
)
from validation import format_report, validate_against_reference

METADATA_FILE = 'model_metadata.json'
TRAIN_FRACTION = 0.8

print("=" * 60)
print("Training ML Models")
print("=" * 60)

# 1. Dataset -------------------------------------------------------------
print("\n[1] Loading dataset...")
frame, source_name = prepare_dataset()
print(f"    Source: {source_name} | rows: {len(frame)}")
print(f"    Range: {frame['timestamp'].min()} -> {frame['timestamp'].max()}")

# 2. Split + calibration -------------------------------------------------
print("\n[2] Chronological split + gas calibration...")
train, test = chronological_split(frame, TRAIN_FRACTION)
train, test, calibrator = fit_and_apply_calibration(train, test, source_name)

if calibrator is not None:
    metrics = calibrator.metrics
    print(f"    Calibration R2(log): {metrics['r2_log']:.4f} | "
          f"RMSE: {metrics['rmse']:.3f} | MAE: {metrics['mae']:.3f}")
else:
    print("    No reference target: using raw gas (legacy dataset)")

# 3. Preprocess ----------------------------------------------------------
train, train_report = preprocess_dataframe(train, feature_cols=FEATURE_COLS)
test, test_report = preprocess_dataframe(test, feature_cols=FEATURE_COLS)
print(f"    Train rows: {len(train)} | Test rows: {len(test)}")
if train_report['constant_features']:
    print(f"    WARNING constant features: {train_report['constant_features']}")

X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]
y_status_train = train['status'].astype(int)
y_status_test = test['status'].astype(int)
y_aqi_train, y_aqi_test = train['aqi'], test['aqi']

# 4. Random Forest -------------------------------------------------------
print("\n[3] Training Random Forest classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1,
)
rf_model.fit(X_train, y_status_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_status_test, y_pred)
print(f"    Accuracy: {accuracy * 100:.2f}%")

present = sorted(set(y_status_test).union(set(y_pred)))
target_names = [AQI_LABELS.get(int(label), str(label)) for label in present]
print(classification_report(
    y_status_test, y_pred, labels=present,
    target_names=target_names, zero_division=0,
))
print("    Feature importance:")
for feature, importance in zip(FEATURE_COLS, rf_model.feature_importances_):
    print(f"      {feature}: {importance * 100:.1f}%")

joblib.dump(rf_model, 'rf_air_model.pkl')
print("    Saved: rf_air_model.pkl")

# 5. Linear Regression ---------------------------------------------------
print("\n[4] Training Linear Regression for AQI...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_aqi_train)
lr_report = validate_against_reference(
    y_aqi_test, lr_model.predict(X_test), 'LinearRegression'
)
print(format_report(lr_report))

joblib.dump(lr_model, 'lr_trend_model.pkl')
print("    Saved: lr_trend_model.pkl")

# 6. Metadata ------------------------------------------------------------
metadata = {}
try:
    with open(METADATA_FILE) as handle:
        metadata = json.load(handle)
except (OSError, json.JSONDecodeError):
    metadata = {}

# Drop keys from the previous (legacy) dataset so metadata is not stale.
for stale in ('data_file', 'data_rows', 'preprocessing'):
    metadata.pop(stale, None)

metadata.update(build_metadata(frame, source_name, calibrator))
metadata.update({
    'trained_at': datetime.now().isoformat(timespec='seconds'),
    'labels': {str(k): v for k, v in AQI_LABELS.items()},
    'rf': {
        'accuracy': float(accuracy),
        'classes': [int(c) for c in rf_model.classes_],
    },
    'lr': {
        'metrics': lr_report['regression'],
        'validation': lr_report,
        'residual_std': float(np.std(y_aqi_test - lr_model.predict(X_test))),
    },
    'notes': [
        'Target is the VOC air-quality sub-index (benzene-equivalent), which '
        'the MQ-135-class sensor can measure; the NO2-driven composite AQI is '
        'not physically measurable with this hardware.',
        'Gas feature is a calibrated VOC concentration, not a raw ADC value.',
        'Split is chronological; the test set is unseen future data.',
    ],
})

with open(METADATA_FILE, 'w') as handle:
    json.dump(metadata, handle, indent=2)
print(f"\n[5] Saved metadata: {METADATA_FILE}")

print("\n" + "=" * 60)
print(f"RF accuracy: {accuracy * 100:.1f}% | "
      f"LR MAE: {lr_report['regression']['mae']:.2f} | "
      f"RMSE: {lr_report['regression']['rmse']:.2f} | "
      f"R2: {lr_report['regression']['r2']:.3f}")
print("=" * 60)

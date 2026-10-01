# train_model.py
"""
Train the Random Forest (classification) and Linear Regression (trend) models.

Improvements over the previous version
--------------------------------------
* Training data is cleaned by ``preprocessing.preprocess_dataframe`` — the
  same pipeline used at inference.
* PM2.5 is no longer a feature (the sensor node cannot measure it).
* The split is chronological, not random, so the test set is genuinely
  unseen future data instead of a shuffled sample.
* Metrics and feature schema are written to ``model_metadata.json``.
"""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from preprocessing import AQI_LABELS, FEATURE_COLS, preprocess_dataframe

DATA_FILE = 'real_air_data.csv'
METADATA_FILE = 'model_metadata.json'
TRAIN_FRACTION = 0.8

print("=" * 55)
print("Training ML Models on Real Data")
print("=" * 55)

# 1. Load and clean -----------------------------------------------------
print("\n[1] Loading real air quality data...")
df = pd.read_csv(DATA_FILE)
print(f"    Loaded {len(df)} records")

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.sort_values('timestamp').reset_index(drop=True)

df, report = preprocess_dataframe(df, feature_cols=FEATURE_COLS)
print(f"    After preprocessing: {report['rows_out']} records")
print(f"    Out-of-range values corrected: {report['out_of_range_values']}")
if report['constant_features']:
    print(f"    WARNING constant features (no information): {report['constant_features']}")

# 2. Chronological split -------------------------------------------------
print("\n[2] Splitting chronologically (no shuffling)...")
split = int(len(df) * TRAIN_FRACTION)
train, test = df.iloc[:split], df.iloc[split:]

X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]
y_status_train, y_status_test = train['status'].astype(int), test['status'].astype(int)
y_aqi_train, y_aqi_test = train['aqi'], test['aqi']

print(f"    Train: {len(train)} rows ({train['timestamp'].min()} -> {train['timestamp'].max()})")
print(f"    Test:  {len(test)} rows ({test['timestamp'].min()} -> {test['timestamp'].max()})")

# 3. Random Forest -------------------------------------------------------
print("\n[3] Training Random Forest classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_status_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_status_test, y_pred)
print(f"    Accuracy: {accuracy * 100:.2f}%")

present = sorted(set(y_status_test).union(set(y_pred)))
target_names = [AQI_LABELS.get(int(label), str(label)) for label in present]
print("\n    Classification report:")
print(classification_report(
    y_status_test, y_pred, labels=present,
    target_names=target_names, zero_division=0,
))

print("    Feature importance:")
for feature, importance in zip(FEATURE_COLS, rf_model.feature_importances_):
    print(f"      {feature}: {importance * 100:.1f}%")

joblib.dump(rf_model, 'rf_air_model.pkl')
print("    Saved: rf_air_model.pkl")

# 4. Linear Regression ---------------------------------------------------
print("\n[4] Training Linear Regression for AQI...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_aqi_train)

y_aqi_pred = lr_model.predict(X_test)
mae = mean_absolute_error(y_aqi_test, y_aqi_pred)
rmse = float(np.sqrt(mean_squared_error(y_aqi_test, y_aqi_pred)))
r2 = r2_score(y_aqi_test, y_aqi_pred)
residual_std = float(np.std(y_aqi_test - y_aqi_pred))

print(f"    MAE:  {mae:.2f}")
print(f"    RMSE: {rmse:.2f}")
print(f"    R²:   {r2:.4f}")

joblib.dump(lr_model, 'lr_trend_model.pkl')
print("    Saved: lr_trend_model.pkl")

# 5. Write metadata ------------------------------------------------------
metadata = {}
try:
    with open(METADATA_FILE) as handle:
        metadata = json.load(handle)
except (OSError, json.JSONDecodeError):
    metadata = {}

metadata.update({
    'feature_cols': list(FEATURE_COLS),
    'labels': {str(k): v for k, v in AQI_LABELS.items()},
    'trained_at': datetime.now().isoformat(timespec='seconds'),
    'data_file': DATA_FILE,
    'data_rows': int(len(df)),
    'preprocessing': report,
    'rf': {
        'accuracy': float(accuracy),
        'classes': [int(c) for c in rf_model.classes_],
    },
    'lr': {
        'mae': float(mae),
        'rmse': rmse,
        'r2': float(r2),
        'residual_std': residual_std,
    },
    'notes': [
        'PM2.5 is not a model feature: the sensor node cannot measure it.',
        'Split is chronological; the test set is unseen future data.',
        'The gas feature in real_air_data.csv is synthetic (see collect_real_data.py).',
        'Phase 2 will replace synthetic gas with calibrated MQ-135 readings.',
    ],
})

with open(METADATA_FILE, 'w') as handle:
    json.dump(metadata, handle, indent=2)
print(f"\n[5] Saved metadata: {METADATA_FILE}")

print("\n" + "=" * 55)
print("Training complete.")
print(f"RF accuracy: {accuracy * 100:.1f}% | LR RMSE: {rmse:.2f} | R²: {r2:.3f}")
print("=" * 55)

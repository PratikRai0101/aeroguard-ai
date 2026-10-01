# train_lstm.py
"""
Train the LSTM time-series model.

Improvements over the previous version
--------------------------------------
* Training data is cleaned by ``preprocessing.preprocess_dataframe``.
* The scaler is fitted on the **training split only** (previously it was
  fitted on all data, leaking test statistics into training).
* The train/validation split is chronological.
* The output layer size follows the classes actually present in the data.
* Metrics are merged into ``model_metadata.json``.
"""

import json
import pickle
from datetime import datetime

import numpy as np

# IMPORT ORDER MATTERS: on macOS, importing pandas/sklearn before TensorFlow
# makes ``model.fit`` deadlock. TensorFlow must be imported first. Do not
# reorder these imports.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import pandas as pd

from preprocessing import AQI_LABELS, FEATURE_COLS, preprocess_dataframe

DATA_FILE = 'real_air_data.csv'
METADATA_FILE = 'model_metadata.json'
WINDOW = 10
TRAIN_FRACTION = 0.85

# Reproducibility for a saved artifact.
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 55)
print("Training LSTM on Real Time-Series Data")
print("=" * 55)

# 1. Load and clean ------------------------------------------------------
print("\n[1] Loading real air quality data...")
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.sort_values('timestamp').reset_index(drop=True)
df, report = preprocess_dataframe(df, feature_cols=FEATURE_COLS)
print(f"    Loaded {report['rows_out']} cleaned records")

# 2. Sequences -----------------------------------------------------------
print("\n[2] Building sequences...")
X_data = df[FEATURE_COLS].values
y_data = df['status'].values.astype(int)


def create_sequences(features, targets, window):
    sequences, labels = [], []
    for index in range(len(features) - window):
        sequences.append(features[index:index + window])
        labels.append(targets[index + window])
    return np.array(sequences), np.array(labels)


X_seq, y_seq = create_sequences(X_data, y_data, WINDOW)
print(f"    {len(X_seq)} windows of {WINDOW} steps")

present = np.unique(y_seq)
if list(present) != list(range(len(present))):
    raise ValueError(
        f"Class labels must be contiguous from 0, got {list(present)}. "
        "Remap the status column before training."
    )
n_classes = len(present)
print(f"    Classes present: {[AQI_LABELS.get(int(c), c) for c in present]}")

# 3. Chronological split + leakage-free scaling --------------------------
print("\n[3] Chronological split and scaling...")
split = int(len(X_seq) * TRAIN_FRACTION)
X_train_raw, X_val_raw = X_seq[:split], X_seq[split:]
y_train, y_val = y_seq[:split], y_seq[split:]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train_raw.reshape(-1, len(FEATURE_COLS)))

X_train = scaler.transform(
    X_train_raw.reshape(-1, len(FEATURE_COLS))
).reshape(X_train_raw.shape)
X_val = scaler.transform(
    X_val_raw.reshape(-1, len(FEATURE_COLS))
).reshape(X_val_raw.shape)

with open('scaler.pkl', 'wb') as handle:
    pickle.dump(scaler, handle)
print("    Saved: scaler.pkl")

print(f"    Train sequences: {len(X_train)} | Validation sequences: {len(X_val)}")

# 4. Model ---------------------------------------------------------------
print("\n[4] Building LSTM model...")
model = Sequential([
    LSTM(64, input_shape=(WINDOW, len(FEATURE_COLS))),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Train ---------------------------------------------------------------
print("\n[5] Training LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1,
)

val_loss = float(history.history['val_loss'][-1])
val_accuracy = float(history.history['val_accuracy'][-1])
print(f"\n[6] Validation loss: {val_loss:.4f} | accuracy: {val_accuracy * 100:.2f}%")

model.save('lstm_air_model.h5')
print("    Saved: lstm_air_model.h5")

# 6. Metadata ------------------------------------------------------------
try:
    with open(METADATA_FILE) as handle:
        metadata = json.load(handle)
except (OSError, json.JSONDecodeError):
    metadata = {}

metadata.setdefault('feature_cols', list(FEATURE_COLS))
metadata.setdefault('labels', {str(k): v for k, v in AQI_LABELS.items()})
metadata['trained_at'] = datetime.now().isoformat(timespec='seconds')
metadata['lstm'] = {
    'window': WINDOW,
    'val_loss': val_loss,
    'val_accuracy': val_accuracy,
    'validated_classes': [int(c) for c in present],
    'train_sequences': int(len(X_train)),
    'val_sequences': int(len(X_val)),
}

with open(METADATA_FILE, 'w') as handle:
    json.dump(metadata, handle, indent=2)
print(f"[7] Saved metadata: {METADATA_FILE}")

print("\n" + "=" * 55)
print("LSTM training complete.")
print("=" * 55)

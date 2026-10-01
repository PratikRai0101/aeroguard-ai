# train_lstm.py
"""
Train the LSTM time-series model on the calibrated VOC pipeline.

The scaler is fitted on the training split only (no leakage), the split is
chronological, and the output layer size follows the classes present.
"""

import json
import pickle
from datetime import datetime

import bootstrap_tf  # noqa: F401  (must precede pandas/sklearn imports)
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import MinMaxScaler

from preprocessing import AQI_LABELS, FEATURE_COLS, preprocess_dataframe
from training import (
    build_metadata,
    chronological_split,
    fit_and_apply_calibration,
    prepare_dataset,
)

METADATA_FILE = 'model_metadata.json'
WINDOW = 10
TRAIN_FRACTION = 0.85

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 60)
print("Training LSTM")
print("=" * 60)

# 1. Dataset -------------------------------------------------------------
print("\n[1] Loading dataset...")
frame, source_name = prepare_dataset()
print(f"    Source: {source_name} | rows: {len(frame)}")

# 2. Split + calibration -------------------------------------------------
print("\n[2] Chronological split + gas calibration...")
train, test = chronological_split(frame, TRAIN_FRACTION)
train, test, calibrator = fit_and_apply_calibration(train, test, source_name)

# 3. Preprocess ----------------------------------------------------------
train, _ = preprocess_dataframe(train, feature_cols=FEATURE_COLS)
test, _ = preprocess_dataframe(test, feature_cols=FEATURE_COLS)

# Sequences are built on the concatenated series, but the scaler and the
# train/validation split respect time order.
combined = np.concatenate([train[FEATURE_COLS].values, test[FEATURE_COLS].values])
labels = np.concatenate([train['status'].values, test['status'].values]).astype(int)


def create_sequences(features, targets, window):
    sequences, sequence_labels = [], []
    for index in range(len(features) - window):
        sequences.append(features[index:index + window])
        sequence_labels.append(targets[index + window])
    return np.array(sequences), np.array(sequence_labels)


X_seq, y_seq = create_sequences(combined, labels, WINDOW)
print(f"\n[3] {len(X_seq)} windows of {WINDOW} steps")

present = np.unique(y_seq)
if list(present) != list(range(len(present))):
    raise ValueError(f"Class labels must be contiguous from 0, got {list(present)}.")
n_classes = len(present)
print(f"    Classes: {[AQI_LABELS.get(int(c), c) for c in present]}")

# 4. Chronological split + leakage-free scaling --------------------------
split = int(len(X_seq) * TRAIN_FRACTION)
X_train_raw, X_val_raw = X_seq[:split], X_seq[split:]
y_train, y_val = y_seq[:split], y_seq[split:]

scaler = MinMaxScaler()
scaler.fit(X_train_raw.reshape(-1, len(FEATURE_COLS)))

X_train = scaler.transform(X_train_raw.reshape(-1, len(FEATURE_COLS))).reshape(X_train_raw.shape)
X_val = scaler.transform(X_val_raw.reshape(-1, len(FEATURE_COLS))).reshape(X_val_raw.shape)

with open('scaler.pkl', 'wb') as handle:
    pickle.dump(scaler, handle)
print(f"    Train sequences: {len(X_train)} | Validation: {len(X_val)} | Saved scaler.pkl")

# 5. Model ---------------------------------------------------------------
print("\n[4] Building LSTM model...")
model = Sequential([
    LSTM(64, input_shape=(WINDOW, len(FEATURE_COLS))),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\n[5] Training LSTM...")
history = model.fit(
    X_train, y_train, epochs=20, batch_size=32,
    validation_data=(X_val, y_val), verbose=1,
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

for key, value in build_metadata(frame, source_name, calibrator).items():
    metadata.setdefault(key, value)
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

print("\n" + "=" * 60)
print("LSTM training complete.")
print("=" * 60)

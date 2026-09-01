# train_lstm.py
"""
Train LSTM model on real time-series air quality data
Uses data from Open-Meteo API
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("Training LSTM on Real Time-Series Data")
print("=" * 50)

# 1. Load real data
print("\n[1] Loading real air quality data...")
df = pd.read_csv('real_air_data.csv')
print(f"    Loaded {len(df)} hourly records")

# 2. Prepare time-series data
print("\n[2] Preparing time-series data...")
WINDOW = 10

# Use temp, hum, gas as features
feature_cols = ['temp', 'hum', 'gas']
X_data = df[feature_cols].values
y_data = df['status'].values

# Create sequences
def create_sequences(X, y, window):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_data, y_data, WINDOW)
print(f"    Sequences: {len(X_seq)} windows of {WINDOW} steps")

# 3. Scale features
print("\n[3] Scaling features...")
scaler = MinMaxScaler()
X_flat = X_seq.reshape(-1, 3)
scaler.fit(X_flat)
X_scaled = scaler.transform(X_flat).reshape(-1, WINDOW, 3)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"    Saved: scaler.pkl")

# 4. Build LSTM model
print("\n[4] Building LSTM model...")
model = Sequential([
    LSTM(64, input_shape=(WINDOW, 3), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Good, Moderate, Poor
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Train
print("\n[5] Training LSTM...")
history = model.fit(
    X_scaled, y_seq,
    epochs=20,
    batch_size=32,
    validation_split=0.15,
    verbose=1
)

# Evaluate
print("\n[6] Model Evaluation:")
val_loss = history.history['val_loss'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"    Validation Loss: {val_loss:.4f}")
print(f"    Validation Accuracy: {val_acc*100:.2f}%")

# Save model
model.save('lstm_air_model.h5')
print(f"    Saved: lstm_air_model.h5")

# 7. Test predictions
print("\n[7] Sample Predictions:")
sample = X_scaled[:5]
preds = model.predict(sample, verbose=0)
labels = ['Good', 'Moderate', 'Poor']
for i, pred in enumerate(preds):
    pred_class = np.argmax(pred)
    conf = pred[pred_class] * 100
    print(f"    Sample {i+1}: {labels[pred_class]} ({conf:.1f}%)")

print("\n" + "=" * 50)
print("LSTM Training Complete!")
print("=" * 50)
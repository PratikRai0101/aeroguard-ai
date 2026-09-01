# train_model.py
"""
Train ML models on real air quality data
Uses data from Open-Meteo API
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("Training ML Models on Real Data")
print("=" * 50)

# 1. Load real data
print("\n[1] Loading real air quality data...")
df = pd.read_csv('real_air_data.csv')
print(f"    Loaded {len(df)} records")

# Features for models
feature_cols = ['temp', 'hum', 'gas', 'pm25']

# 2. Prepare training data
print("\n[2] Preparing training data...")
X = df[feature_cols].copy()
y_status = df['status'].values
y_aqi = df['aqi'].values

# Split data
X_train, X_test, y_status_train, y_status_test = train_test_split(
    X, y_status, test_size=0.2, random_state=42
)
_, _, y_aqi_train, y_aqi_test = train_test_split(
    X, y_aqi, test_size=0.2, random_state=42
)

print(f"    Training samples: {len(X_train)}")
print(f"    Test samples: {len(X_test)}")

# 3. Train Random Forest (Classifier)
print("\n[3] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_status_train)

# Evaluate
y_pred = rf_model.predict(X_test)
accuracy = rf_model.score(X_test, y_status_test)
print(f"    Accuracy: {accuracy*100:.2f}%")

# Detailed report
print("\n    Classification Report:")
print(classification_report(y_status_test, y_pred, 
      target_names=['Good', 'Moderate', 'Poor', 'Very Poor', 'Severe', 'Hazardous'][:len(np.unique(y_status))]))

# Save
joblib.dump(rf_model, 'rf_air_model.pkl')
print(f"    Saved: rf_air_model.pkl")

# 4. Train Linear Regression (AQI Predictor)
print("\n[4] Training Linear Regression for AQI prediction...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_aqi_train)

# Evaluate
y_aqi_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_aqi_test, y_aqi_pred)
r2 = r2_score(y_aqi_test, y_aqi_pred)
print(f"    MSE: {mse:.2f}")
print(f"    R² Score: {r2:.4f}")

# Save
joblib.dump(lr_model, 'lr_trend_model.pkl')
print(f"    Saved: lr_trend_model.pkl")

# 5. Feature Importance
print("\n[5] Feature Importance (Random Forest):")
for feat, imp in zip(feature_cols, rf_model.feature_importances_):
    print(f"    {feat}: {imp*100:.1f}%")

# 6. Model Summary
print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)
print(f"RF Model: {accuracy*100:.1f}% accuracy")
print(f"LR Model: R² = {r2:.3f}")
print(f"Data range: AQI {df['aqi'].min():.0f} - {df['aqi'].max():.0f}")
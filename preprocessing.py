# preprocessing.py
"""
Data preprocessing for AeroGuard AI.

This module is the single source of truth for how raw sensor readings are
cleaned before they reach the ML models. Both the training scripts and the
live inference path (`predictors.py`) call into it, so the transformations
applied at train time match the ones applied at serve time.

Pipeline
--------
    raw reading
        -> coerce to numeric
        -> reject physically impossible values
        -> fill missing values (interpolate)
        -> smooth noise (rolling median)
        -> model-ready features

Design notes
------------
* The sensor node measures exactly three quantities: temperature, humidity
  and the MQ-135 gas response. ``FEATURE_COLS`` is therefore the canonical
  model input schema.
* ``AQI_LABELS`` is the canonical integer -> category mapping used by every
  model and every dashboard. Do not redefine label maps elsewhere.
* Physical limits and AQI categories are imported from ``aqi_utils`` so
  there is only one definition in the codebase.
"""

import numpy as np
import pandas as pd

from aqi_utils import SENSOR_LIMITS, AQI_CATEGORIES

# Canonical model input schema. The node cannot measure PM2.5 directly, so it
# must not appear here: training on a feature that is unavailable at inference
# is train/serve skew.
FEATURE_COLS = ['temp', 'hum', 'gas']

# Canonical AQI category labels. Index == integer class used by the models.
AQI_LABELS = {index: meta['name'] for index, meta in AQI_CATEGORIES.items()}


def label_for(class_index):
    """Return the canonical human-readable label for an integer class."""
    try:
        return AQI_LABELS.get(int(class_index), 'Unknown')
    except (TypeError, ValueError):
        return 'Unknown'


def validate_reading(temp, hum, gas):
    """
    Validate a single raw reading against physical limits.

    Returns
    -------
    (is_valid, errors) : (bool, list[str])
    """
    values = {'temp': temp, 'hum': hum, 'gas': gas}
    errors = []

    for key, value in values.items():
        if value is None:
            errors.append(f"{key} is missing")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"{key}={value!r} is not numeric")
            continue
        if not np.isfinite(numeric):
            errors.append(f"{key} is not finite")
            continue
        low = SENSOR_LIMITS[key]['min']
        high = SENSOR_LIMITS[key]['max']
        if numeric < low or numeric > high:
            errors.append(f"{key}={numeric} outside physical range [{low}, {high}]")

    return len(errors) == 0, errors


def preprocess_dataframe(df, feature_cols=None, smooth_window=5):
    """
    Clean a dataframe of sensor readings.

    Steps
    -----
    1. Coerce feature columns to numeric.
    2. Replace physically impossible values with NaN.
    3. Fill missing values by linear interpolation, then ffill/bfill for the
       edges. Rows that remain empty are dropped.
    4. Smooth remaining noise with a centered rolling median.
    5. Detect constant features (a warning sign for the dataset, not a crash).

    Parameters
    ----------
    df : pandas.DataFrame
    feature_cols : list[str], optional
        Defaults to :data:`FEATURE_COLS`.
    smooth_window : int
        Rolling median window. ``<= 1`` disables smoothing.

    Returns
    -------
    (clean_df, report) : (pandas.DataFrame, dict)
    """
    feature_cols = list(feature_cols or FEATURE_COLS)
    report = {
        'rows_in': int(len(df)),
        'rows_out': 0,
        'out_of_range_values': 0,
        'missing_values_filled': 0,
        'rows_dropped': 0,
        'constant_features': [],
        'smoothing_window': int(smooth_window),
    }

    clean = df.copy()

    # 1. Coerce to numeric.
    for col in feature_cols:
        clean[col] = pd.to_numeric(clean[col], errors='coerce')

    # 2. Impossible values become NaN.
    for col in feature_cols:
        low = SENSOR_LIMITS[col]['min']
        high = SENSOR_LIMITS[col]['max']
        invalid = clean[col].notna() & ((clean[col] < low) | (clean[col] > high))
        report['out_of_range_values'] += int(invalid.sum())
        clean.loc[invalid, col] = np.nan

    # 3. Missing-value handling.
    report['missing_values_filled'] = int(clean[feature_cols].isna().sum().sum())
    clean[feature_cols] = (
        clean[feature_cols]
        .interpolate(method='linear', limit_direction='both')
    )
    before_drop = len(clean)
    clean = clean.dropna(subset=feature_cols).reset_index(drop=True)
    report['rows_dropped'] = int(before_drop - len(clean))

    # 4. Noise smoothing.
    if smooth_window and smooth_window > 1 and len(clean) > 0:
        clean[feature_cols] = (
            clean[feature_cols]
            .rolling(window=smooth_window, min_periods=1, center=True)
            .median()
        )

    # 5. Constant-feature detection.
    for col in feature_cols:
        if clean[col].nunique(dropna=True) <= 1:
            report['constant_features'].append(col)

    report['rows_out'] = int(len(clean))
    return clean, report


def fit_scaler(df, feature_cols=None):
    """Fit a MinMax scaler on cleaned feature columns (training data only)."""
    from sklearn.preprocessing import MinMaxScaler

    feature_cols = list(feature_cols or FEATURE_COLS)
    scaler = MinMaxScaler()
    scaler.fit(df[feature_cols].values)
    return scaler


def transform_features(df, scaler, feature_cols=None):
    """Scale feature columns using an already-fitted scaler."""
    feature_cols = list(feature_cols or FEATURE_COLS)
    return scaler.transform(df[feature_cols].values)


def features_from_reading(temp, hum, gas, feature_cols=None):
    """
    Build a single-row, model-ready feature frame from a raw reading.

    Returns
    -------
    (DataFrame, errors) : (pandas.DataFrame or None, list[str])
        The frame is ``None`` when the reading is physically invalid.
    """
    feature_cols = list(feature_cols or FEATURE_COLS)
    is_valid, errors = validate_reading(temp, hum, gas)
    if not is_valid:
        return None, errors

    return pd.DataFrame(
        [[float(temp), float(hum), float(gas)]],
        columns=feature_cols,
    ), []

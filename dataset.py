# dataset.py
"""
Dataset loaders for AeroGuard AI.

Primary dataset
---------------
UCI Air Quality (De Vito et al., 2008), 9,357 hourly records from a
metal-oxide sensor array co-located with a certified reference analyzer.
This is the dataset used for gas calibration and reference validation.

    Sensors:  PT08.S1..S5 (metal oxide), T, RH, AH
    Reference: CO, NO2, C6H6 (certified analyzer)

The gas proxy is ``PT08.S2(NMHC)`` (best VOC-like channel; r = 0.98 with
reference benzene) and the calibration target is reference benzene (C6H6).

Legacy dataset
--------------
``real_air_data.csv`` (768 rows). Kept as a fallback. Its gas column is
synthetic and its temperature/humidity are constant, so it is only used
when the UCI dataset is not present.
"""

import os

import numpy as np
import pandas as pd

from aqi_utils import calculate_aqi_from_concentration, get_aqi_category

UCI_PATH = os.path.join('data', 'AirQualityUCI.csv')
LEGACY_PATH = 'real_air_data.csv'

# Canonical training-frame columns shared by every dataset.
CANONICAL_COLUMNS = [
    'timestamp', 'temp', 'hum', 'gas_raw', 'gas_target', 'aqi', 'status',
]

# UCI specifics.
UCI_GAS_COLUMN = 'PT08.S2(NMHC)'
UCI_GAS_TARGET = 'C6H6(GT)'
UCI_REFERENCE_POLLUTANTS = {
    'co': 'CO(GT)',      # mg/m^3
    'no2': 'NO2(GT)',    # ug/m^3
    'c6h6': 'C6H6(GT)',  # ug/m^3
}


def _sub_index_array(values, pollutant):
    """Vectorised CPCB sub-index for one pollutant."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    mask = np.isfinite(values) & (values >= 0)
    if mask.any():
        out[mask] = [
            calculate_aqi_from_concentration(float(v), pollutant)
            for v in values[mask]
        ]
    return out


def _reference_aqi(frame):
    """Worst CPCB sub-index across the available reference pollutants."""
    sub_indices = []
    for pollutant, column in UCI_REFERENCE_POLLUTANTS.items():
        if column in frame.columns:
            sub_indices.append(_sub_index_array(frame[column], pollutant))

    if not sub_indices:
        return np.full(len(frame), np.nan)

    stacked = np.vstack(sub_indices)
    all_nan = np.all(np.isnan(stacked), axis=0)
    worst = np.full(stacked.shape[1], np.nan)
    if (~all_nan).any():
        worst[~all_nan] = np.nanmax(stacked[:, ~all_nan], axis=0)
    return worst


def _voc_aqi(frame):
    """
    VOC air-quality sub-index: the CPCB benzene (C6H6) sub-index.

    This is the target the MQ-135-class sensor can actually measure. The
    composite AQI is driven mostly by NO2 (about 81% of hours in this
    dataset), which a VOC sensor does not respond to (r = 0.09), so the
    composite index is kept only as a reference column.
    """
    return _sub_index_array(frame['C6H6(GT)'], 'c6h6')


def load_uci_air_quality(path=UCI_PATH):
    """
    Load and normalise the UCI Air Quality dataset.

    Returns a canonical frame: timestamp, temp, hum, gas_raw, gas_target,
    aqi, status (plus the raw reference columns).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"UCI dataset not found at {path}. Run `python fetch_datasets.py`."
        )

    frame = pd.read_csv(path, sep=';', decimal=',')
    frame = frame.dropna(axis=1, how='all').dropna(axis=0, how='all')
    frame = frame.replace(-200, np.nan)

    timestamps = pd.to_datetime(
        frame['Date'] + ' ' + frame['Time'].str.replace('.', ':', regex=False),
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce',
    )
    frame['timestamp'] = timestamps
    frame = frame.dropna(subset=['timestamp'])

    frame['temp'] = pd.to_numeric(frame['T'], errors='coerce')
    frame['hum'] = pd.to_numeric(frame['RH'], errors='coerce')
    frame['gas_raw'] = pd.to_numeric(frame[UCI_GAS_COLUMN], errors='coerce')
    frame['gas_target'] = pd.to_numeric(frame[UCI_GAS_TARGET], errors='coerce')

    # Primary target: VOC air quality (benzene sub-index) — what this
    # sensor class can measure. Composite AQI kept for reference only.
    frame['aqi'] = _voc_aqi(frame)
    frame['status'] = frame['aqi'].apply(
        lambda value: get_aqi_category(value) if np.isfinite(value) else np.nan
    )
    frame['aqi_composite'] = _reference_aqi(frame)
    frame['status_composite'] = frame['aqi_composite'].apply(
        lambda value: get_aqi_category(value) if np.isfinite(value) else np.nan
    )

    frame = frame.sort_values('timestamp').reset_index(drop=True)

    keep = CANONICAL_COLUMNS + [
        'aqi_composite', 'status_composite',
        'CO(GT)', 'NO2(GT)', 'C6H6(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)',
        'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
    ]
    keep = [column for column in keep if column in frame.columns]
    return frame[keep]


def load_legacy_csv(path=LEGACY_PATH):
    """Load the legacy 768-row dataset into the canonical schema."""
    frame = pd.read_csv(path)
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce')
    frame = frame.sort_values('timestamp').reset_index(drop=True)

    frame['gas_raw'] = pd.to_numeric(frame['gas'], errors='coerce')
    frame['gas_target'] = np.nan  # no reference concentration available
    frame['status'] = pd.to_numeric(frame['status'], errors='coerce')
    return frame[CANONICAL_COLUMNS]


def load_training_data(source='auto'):
    """
    Load the best available training dataset.

    Returns
    -------
    (frame, source_name)
    """
    if source in ('auto', 'uci') and os.path.exists(UCI_PATH):
        return load_uci_air_quality(UCI_PATH), 'uci'

    if source == 'uci':
        raise FileNotFoundError(f"UCI dataset not found at {UCI_PATH}")

    return load_legacy_csv(LEGACY_PATH), 'legacy'


def dataset_summary(frame):
    """Small dict describing a canonical frame, for metadata/reporting."""
    return {
        'rows': int(len(frame)),
        'start': str(frame['timestamp'].min()),
        'end': str(frame['timestamp'].max()),
        'has_reference_target': bool(frame['gas_target'].notna().any()),
        'temp_unique': int(frame['temp'].nunique(dropna=True)),
        'hum_unique': int(frame['hum'].nunique(dropna=True)),
        'aqi_min': float(np.nanmin(frame['aqi'])) if frame['aqi'].notna().any() else None,
        'aqi_max': float(np.nanmax(frame['aqi'])) if frame['aqi'].notna().any() else None,
    }

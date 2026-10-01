# training.py
"""
Shared training helpers: dataset loading, gas calibration and splitting.

Both ``train_model.py`` and ``train_lstm.py`` go through this module so the
feature pipeline (raw gas -> calibrated VOC concentration) is identical and
the fitted calibrator is saved once for inference.
"""

import os

import numpy as np

from calibration import GasCalibrator
from dataset import dataset_summary, load_training_data

CALIBRATOR_FILE = 'gas_calibrator.pkl'

# Model features after calibration. ``gas`` holds the calibrated VOC
# concentration, not the raw sensor response.
MODEL_FEATURES = ['temp', 'hum', 'gas']


def prepare_dataset(source='auto', require_target=True):
    """
    Load the training dataset and drop rows missing required fields.

    Returns
    -------
    (frame, source_name)
    """
    frame, source_name = load_training_data(source)

    required = ['timestamp', 'temp', 'hum', 'gas_raw', 'aqi', 'status']
    frame = frame.dropna(subset=required).reset_index(drop=True)
    frame = frame.sort_values('timestamp').reset_index(drop=True)

    return frame, source_name


def chronological_split(frame, train_fraction=0.8):
    """Split by time so the test set is genuinely unseen future data."""
    split = int(len(frame) * train_fraction)
    return frame.iloc[:split].copy(), frame.iloc[split:].copy()


def fit_and_apply_calibration(train, test, source_name):
    """
    Fit the gas calibrator on the training split and add the ``gas`` column.

    When no reference target is available (legacy dataset), the raw sensor
    value is passed through unchanged so the pipeline still runs.

    Returns
    -------
    (train, test, calibrator_or_None)
    """
    valid_targets = train['gas_target'].notna().sum() if 'gas_target' in train else 0

    if valid_targets >= 10:
        calibrator = GasCalibrator().fit(
            train['gas_raw'], train['temp'], train['hum'], train['gas_target'],
            trained_on=f"{source_name}:benzene-equivalent",
        )
        calibrator.save(CALIBRATOR_FILE)

        train = train.assign(
            gas=calibrator.transform(train['gas_raw'], train['temp'], train['hum'])
        )
        test = test.assign(
            gas=calibrator.transform(test['gas_raw'], test['temp'], test['hum'])
        )
        return train, test, calibrator

    # Legacy fallback: no reference concentration, use raw gas.
    train = train.assign(gas=train['gas_raw'])
    test = test.assign(gas=test['gas_raw'])
    return train, test, None


def build_metadata(frame, source_name, calibrator):
    """Common metadata block describing the dataset and calibration."""
    metadata = {
        'dataset_source': source_name,
        'dataset': dataset_summary(frame),
        'feature_cols': list(MODEL_FEATURES),
    }

    if calibrator is not None:
        metadata['calibration'] = {
            'trained_on': calibrator.trained_on,
            'input_domain': 'MQ-135-class metal-oxide response',
            'metrics': calibrator.metrics,
            'note': (
                'Calibrated VOC concentration (benzene-equivalent ug/m^3). '
                'The live MQ-135 must be re-fitted with a clean-air R0 and a '
                'reference gas before deployment; these coefficients describe '
                'the UCI sensor array.'
            ),
        }
    return metadata

"""Tests for the Phase 1 preprocessing pipeline and model feature contract."""

import json
import os
import unittest

import bootstrap_tf  # noqa: F401  (must precede pandas/ML imports)
import pandas as pd

from preprocessing import (
    AQI_LABELS,
    FEATURE_COLS,
    features_from_reading,
    preprocess_dataframe,
    validate_reading,
)


class PreprocessingTests(unittest.TestCase):
    def test_feature_schema_is_the_three_sensor_channels(self):
        # The node measures temp, humidity and MQ-135 gas. PM2.5 is not
        # available at inference and must not be a model input.
        self.assertEqual(FEATURE_COLS, ['temp', 'hum', 'gas'])

    def test_labels_cover_all_six_cpcb_categories(self):
        self.assertEqual(set(AQI_LABELS), {0, 1, 2, 3, 4, 5})
        self.assertEqual(AQI_LABELS[0], 'Good')
        self.assertEqual(AQI_LABELS[5], 'Hazardous')

    def test_validate_reading_accepts_plausible_values(self):
        valid, errors = validate_reading(25.0, 55.0, 150.0)
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_validate_reading_rejects_impossible_values(self):
        valid, errors = validate_reading(999.0, 55.0, 150.0)
        self.assertFalse(valid)
        self.assertTrue(errors)

    def test_preprocess_repairs_missing_and_out_of_range(self):
        frame = pd.DataFrame({
            'temp': [25.0, None, 25.0, 999.0],
            'hum': [55.0, 55.0, None, 55.0],
            'gas': [150.0, 150.0, 150.0, 150.0],
        })

        clean, report = preprocess_dataframe(frame, smooth_window=1)

        self.assertEqual(report['out_of_range_values'], 1)
        self.assertGreaterEqual(report['missing_values_filled'], 2)
        self.assertEqual(clean[FEATURE_COLS].isna().sum().sum(), 0)
        self.assertEqual(len(clean), 4)

    def test_features_from_reading_rejects_invalid(self):
        model_frame, errors = features_from_reading(999.0, 55.0, 150.0)
        self.assertIsNone(model_frame)
        self.assertTrue(errors)

    def test_features_from_reading_builds_expected_columns(self):
        model_frame, errors = features_from_reading(25.0, 55.0, 150.0)
        self.assertEqual(errors, [])
        self.assertEqual(list(model_frame.columns), FEATURE_COLS)

    def test_trained_metadata_matches_feature_schema(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'model_metadata.json')
        self.assertTrue(os.path.exists(path), "models must be trained before this test")

        with open(path) as handle:
            metadata = json.load(handle)

        self.assertEqual(metadata['feature_cols'], FEATURE_COLS)
        self.assertEqual(sorted(int(k) for k in metadata['labels']), [0, 1, 2, 3, 4, 5])
        self.assertNotIn('pm25', metadata['feature_cols'])


if __name__ == '__main__':
    unittest.main()

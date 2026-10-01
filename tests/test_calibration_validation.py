"""Tests for the calibration and validation modules."""

import os
import unittest

import bootstrap_tf  # noqa: F401
import numpy as np

from calibration import GasCalibrator, mq135_ppm, mq135_r0, mq135_rs_from_adc
from validation import (
    category_agreement,
    regression_metrics,
    validate_against_reference,
)


class CalibrationTests(unittest.TestCase):
    def test_calibrator_recovers_a_known_power_law(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(300, 2200, 500)
        temp = rng.uniform(0, 35, 500)
        hum = rng.uniform(20, 90, 500)

        # log(target) = b0 + b1*log(raw) + b2*temp + b3*hum
        target = np.exp(-15.0 + 2.7 * np.log(raw) + 0.002 * temp + 0.0003 * hum)

        calibrator = GasCalibrator().fit(raw, temp, hum, target)
        self.assertGreater(calibrator.metrics['r2_log'], 0.99)

        predicted = calibrator.transform(raw, temp, hum)
        self.assertTrue(np.allclose(predicted, target, rtol=0.01))

    def test_calibrator_requires_enough_rows(self):
        with self.assertRaises(ValueError):
            GasCalibrator().fit([1, 2], [20, 20], [50, 50], [1, 2])

    def test_mq135_resistance_is_positive_and_monotonic(self):
        rs_low = mq135_rs_from_adc(500)
        rs_high = mq135_rs_from_adc(2000)
        self.assertGreater(rs_low, 0)
        self.assertGreater(rs_high, 0)
        # Higher ADC -> lower sensor resistance.
        self.assertLess(rs_high, rs_low)

    def test_mq135_r0_and_ppm(self):
        r0 = mq135_r0([10.0, 10.0, 12.0, 9.0])
        self.assertAlmostEqual(r0, 10.0)

        # Lower Rs/R0 (more gas) -> higher ppm.
        ppm_more_gas = mq135_ppm(5.0, 10.0)
        ppm_clean = mq135_ppm(10.0, 10.0)
        self.assertGreater(ppm_more_gas, ppm_clean)


class ValidationTests(unittest.TestCase):
    def test_perfect_regression(self):
        metrics = regression_metrics([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertAlmostEqual(metrics['mae'], 0.0)
        self.assertAlmostEqual(metrics['rmse'], 0.0)
        self.assertAlmostEqual(metrics['r2'], 1.0)
        self.assertAlmostEqual(metrics['pearson'], 1.0)

    def test_regression_ignores_nan_pairs(self):
        metrics = regression_metrics([1, np.nan, 3], [1, 2, 3])
        self.assertEqual(metrics['n'], 2)
        self.assertAlmostEqual(metrics['mae'], 0.0)

    def test_category_agreement_exact_and_within_one(self):
        exact = category_agreement([10, 60, 150], [10, 60, 150])
        self.assertAlmostEqual(exact['exact_match'], 1.0)

        within = category_agreement([10, 60, 150], [10, 60, 250])
        self.assertAlmostEqual(within['exact_match'], 2 / 3)
        self.assertAlmostEqual(within['within_one'], 1.0)

    def test_validate_report_shape(self):
        report = validate_against_reference([10, 60, 150], [12, 62, 155], 'unit')
        self.assertEqual(report['name'], 'unit')
        self.assertIn('regression', report)
        self.assertIn('categories', report)


class DatasetContractTests(unittest.TestCase):
    def test_uci_loader_returns_canonical_columns(self):
        path = os.path.join('data', 'AirQualityUCI.csv')
        if not os.path.exists(path):
            self.skipTest("UCI dataset not downloaded")

        from dataset import load_uci_air_quality

        frame = load_uci_air_quality(path)
        for column in ['timestamp', 'temp', 'hum', 'gas_raw', 'gas_target', 'aqi', 'status']:
            self.assertIn(column, frame.columns)

        # The primary target is the VOC sub-index, not the composite AQI.
        self.assertIn('aqi_composite', frame.columns)
        self.assertTrue(frame['gas_target'].notna().any())


if __name__ == '__main__':
    unittest.main()

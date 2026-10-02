"""Tests for MQ-135 hardware calibration."""

import os
import tempfile
import unittest

import bootstrap_tf  # noqa: F401
import numpy as np

from calibration import GasCalibrator
from mq135 import (
    DATASHEET_COEFFICIENTS,
    MQ135Profile,
    r0_from_adc_samples,
)


class MQ135ProfileTests(unittest.TestCase):
    def test_resistance_falls_as_adc_rises(self):
        profile = MQ135Profile(r0=10.0)
        self.assertLess(profile.rs(2500), profile.rs(1000))

    def test_concentration_rises_as_gas_rises(self):
        # Higher gas -> lower Rs/R0 -> higher ppm.
        profile = MQ135Profile(r0=10.0, gas='co2')
        self.assertGreater(profile.ppm(2500), profile.ppm(1000))

    def test_ratio_requires_r0(self):
        with self.assertRaises(RuntimeError):
            MQ135Profile().ratio(1000)

    def test_r0_is_median_of_samples(self):
        r0, count = r0_from_adc_samples([2000, 2010, 1990, 2005])
        self.assertEqual(count, 4)
        self.assertGreater(r0, 0)

    def test_profile_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profile.json')
            MQ135Profile(r0=9.5, gas='co2', source='clean_air').save(path)
            loaded = MQ135Profile.load(path)

        self.assertAlmostEqual(loaded.r0, 9.5)
        self.assertEqual(loaded.gas, 'co2')
        self.assertTrue(loaded.calibrated)

    def test_all_datasheet_slopes_are_negative(self):
        for gas, coefficients in DATASHEET_COEFFICIENTS.items():
            self.assertLess(coefficients['b'], 0, gas)


class MQ135CalibrationFitTests(unittest.TestCase):
    def test_fit_recovers_negative_slope(self):
        rng = np.random.default_rng(0)
        ratio = rng.uniform(0.15, 2.0, 400)
        temp = rng.uniform(5, 40, 400)
        hum = rng.uniform(20, 90, 400)
        reference = np.exp(1.5 - 2.35 * np.log(ratio) + 0.003 * temp + 0.0004 * hum)

        calibrator = GasCalibrator().fit(
            ratio, temp, hum, reference, trained_on='mq135:test'
        )

        self.assertLess(calibrator.metrics['coefficients']['log_raw'], 0)
        self.assertGreater(calibrator.metrics['r2_log'], 0.99)

        # Direction: more gas (lower ratio) -> higher concentration.
        more_gas = calibrator.transform([0.5], [25.0], [55.0])[0]
        less_gas = calibrator.transform([1.5], [25.0], [55.0])[0]
        self.assertGreater(more_gas, less_gas)


if __name__ == '__main__':
    unittest.main()

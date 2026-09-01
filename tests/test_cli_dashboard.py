"""Regression tests for CLI dashboard data-source handling."""

from collections import deque
from unittest.mock import patch
import unittest

import dashboard_v2
from aqi_utils import OutlierDetector


class FakeDatabase:
    def __init__(self):
        self.readings = []

    def add_reading(self, *args, **kwargs):
        self.readings.append((args, kwargs))

    def add_prediction(self, *args, **kwargs):
        pass

    def get_statistics(self):
        return {"readings": len(self.readings), "alerts": 0}


class FakePredictor:
    def add_reading(self, *args):
        pass

    def predict_all(self, *args):
        return {
            "current": {"label": "Moderate", "confidence": 90},
            "future": {"label": "Moderate", "confidence": 80},
            "trend": {"trend": "stable", "aqi": 75},
        }


class CliDashboardRegressionTests(unittest.TestCase):
    def test_cli_dashboard_defaults_to_real_sensor_mode_when_available(self):
        self.assertFalse(dashboard_v2.FORCE_MOCK)

    def test_mock_readings_are_saved_with_mock_source(self):
        database = FakeDatabase()
        readings = iter([[25.0, 55.0, 150.0]] * 10)

        with (
            patch.object(dashboard_v2, "mock_data_generator", return_value=readings),
            patch.object(dashboard_v2.time, "sleep", side_effect=[None] * 9 + [KeyboardInterrupt]),
        ):
            dashboard_v2.run_mock_mode(
                database,
                FakePredictor(),
                OutlierDetector(window_size=10),
                deque(maxlen=10),
            )

        self.assertEqual(database.readings[0][1]["source"], "mock")


if __name__ == "__main__":
    unittest.main()

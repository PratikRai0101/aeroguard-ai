"""Tests for Phase 3: reliability, airborne risk assessment and SHAP."""

import os
import sqlite3
import tempfile
import unittest

import bootstrap_tf  # noqa: F401
import joblib

from database import SensorDatabase
from explain import SHAP_AVAILABLE, ModelExplainer
import reliability
import risk


class ReliabilityTests(unittest.TestCase):
    def test_prediction_interval_is_symmetric_and_widens_with_level(self):
        narrow = reliability.prediction_interval(100, 10, level=0.80)
        wide = reliability.prediction_interval(100, 10, level=0.95)
        self.assertAlmostEqual(narrow['low'], 100 - narrow['half_width'], places=1)
        self.assertAlmostEqual(narrow['high'], 100 + narrow['half_width'], places=1)
        self.assertGreater(wide['half_width'], narrow['half_width'])

    def test_category_probability_is_higher_away_from_a_boundary(self):
        mid_band = reliability.category_probability(25, 10)   # middle of Good
        on_boundary = reliability.category_probability(50, 10)  # Good/Moderate edge
        self.assertGreater(mid_band, on_boundary)
        self.assertAlmostEqual(on_boundary, 0.5, places=1)

    def test_assess_returns_expected_shape(self):
        report = reliability.assess(188.8, 11.38)
        self.assertIn(report['label'], ['High', 'Moderate', 'Low'])
        self.assertIn('interval', report)
        self.assertGreater(report['category_probability'], 0.0)


class RiskTests(unittest.TestCase):
    def test_good_air_normal_conditions_is_low(self):
        report = risk.assess_risk(20, temp=24, hum=50, trend='stable')
        self.assertEqual(report['level'], 'Low')

    def test_hazardous_air_is_high(self):
        report = risk.assess_risk(420, temp=24, hum=50, trend='stable')
        self.assertEqual(report['level'], 'High')

    def test_dry_humidity_and_rising_trend_raise_risk(self):
        calm = risk.assess_risk(120, temp=24, hum=50, trend='falling')
        worse = risk.assess_risk(120, temp=10, hum=25, trend='rising')
        self.assertGreater(worse['score'], calm['score'])

    def test_risk_is_labelled_as_environmental_not_diagnosis(self):
        report = risk.assess_risk(150, temp=24, hum=50)
        self.assertIn('not a medical diagnosis', report['disclaimer'])
        self.assertTrue(report['factors'])


class ExplainTests(unittest.TestCase):
    def setUp(self):
        if not SHAP_AVAILABLE:
            self.skipTest("shap not installed")
        if not os.path.exists('rf_air_model.pkl'):
            self.skipTest("model not trained")

    def test_explanation_ranks_features_and_has_three_factors(self):
        model = joblib.load('rf_air_model.pkl')
        explainer = ModelExplainer(model, ['temp', 'hum', 'gas'])
        result = explainer.explain([[18.0, 49.0, 10.0]], predicted_class=2)

        self.assertEqual(result['method'], 'TreeSHAP')
        self.assertEqual(len(result['all_factors']), 3)
        self.assertEqual(len(result['top_factors']), 3)

        magnitudes = [abs(f['contribution']) for f in result['all_factors']]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))


class PredictorIntegrationTests(unittest.TestCase):
    def test_predict_all_includes_phase3_outputs(self):
        from predictors import AQIPredictor

        predictor = AQIPredictor('.')
        result = None
        for index in range(11):
            result = predictor.predict_all(18.0, 30.0, 900 + index * 40)

        self.assertIn('reliability', result)
        self.assertIn('risk', result)
        self.assertIn('explanation', result)
        self.assertIn(result['risk']['level'], ['Low', 'Moderate', 'High'])


class DatabasePhase3Tests(unittest.TestCase):
    def test_prediction_stores_risk_and_reliability(self):
        with tempfile.TemporaryDirectory() as directory:
            db = SensorDatabase(os.path.join(directory, 'readings.db'))
            db.add_prediction(
                'Poor', 90.0, 'Poor', 80.0, 'stable', 150.0,
                risk_level='Moderate', reliability_label='High',
                reliability_probability=0.84, explanation={'top_factors': []},
            )

            conn = sqlite3.connect(db.db_file)
            row = conn.execute(
                'SELECT risk_level, reliability_label, reliability_probability, explanation '
                'FROM predictions ORDER BY id DESC LIMIT 1'
            ).fetchone()
            conn.close()

        self.assertEqual(row[0], 'Moderate')
        self.assertEqual(row[1], 'High')
        self.assertAlmostEqual(row[2], 0.84)
        self.assertIn('top_factors', row[3])


if __name__ == '__main__':
    unittest.main()

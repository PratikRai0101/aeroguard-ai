# reliability.py
"""
Prediction reliability.

The review panel warned against displaying an arbitrary "86% confidence".
This module derives reliability from validated model error instead:

1. A **prediction interval** for the AQI value, from the residual standard
   deviation measured on the held-out test set.
2. The **probability that the true AQI falls in the same CPCB category** as
   the prediction, by integrating a Gaussian ``N(predicted, residual_std)``
   over the category band.

Both are consequences of the validation in ``validation.py``; neither is
invented. If the model is retrained and its RMSE changes, the reliability
changes with it.
"""

import math

from aqi_utils import get_aqi_category

# Upper AQI bound of each CPCB category (index == category).
CATEGORY_UPPER_BOUNDS = [50, 100, 200, 300, 400, 500]

# Two-sided z-scores for common interval levels.
_Z_SCORES = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}


def _normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def prediction_interval(predicted_aqi, residual_std, level=0.80):
    """
    Prediction interval around a predicted AQI value.

    Returns a dict with the lower/upper bounds and the half-width.
    """
    z = _Z_SCORES.get(level, _Z_SCORES[0.80])
    half_width = z * float(residual_std)
    return {
        'level': level,
        'low': round(max(0.0, predicted_aqi - half_width), 1),
        'high': round(predicted_aqi + half_width, 1),
        'half_width': round(half_width, 1),
    }


def category_probability(predicted_aqi, residual_std):
    """
    Probability that the true AQI shares the predicted CPCB category.

    This is the mass of ``N(predicted_aqi, residual_std)`` inside the
    category band. A prediction far from a category boundary is reliable;
    one sitting on a boundary is not.
    """
    residual_std = float(residual_std)
    if residual_std <= 0:
        return 1.0

    category = get_aqi_category(predicted_aqi)
    lower = 0.0 if category == 0 else CATEGORY_UPPER_BOUNDS[category - 1]
    upper = CATEGORY_UPPER_BOUNDS[category]

    probability = (
        _normal_cdf((upper - predicted_aqi) / residual_std)
        - _normal_cdf((lower - predicted_aqi) / residual_std)
    )
    return float(min(max(probability, 0.0), 1.0))


def reliability_label(probability):
    """Bucket a category probability into High / Moderate / Low."""
    if probability >= 0.80:
        return 'High'
    if probability >= 0.60:
        return 'Moderate'
    return 'Low'


def assess(predicted_aqi, residual_std, level=0.80):
    """
    Full reliability report for an AQI prediction.

    Returns
    -------
    dict
        ``category_probability``, ``label``, ``interval`` and the
        ``residual_std`` the estimate is based on.
    """
    probability = category_probability(predicted_aqi, residual_std)
    return {
        'category_probability': round(probability, 3),
        'label': reliability_label(probability),
        'interval': prediction_interval(predicted_aqi, residual_std, level),
        'residual_std': round(float(residual_std), 2),
    }

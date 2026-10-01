# risk.py
"""
Airborne disease risk assessment.

**This is an environmental risk indicator, not a medical diagnosis.**

It combines the predicted air quality with environmental conditions and the
recent trend to estimate how favourable current conditions are for airborne
pathogen transmission:

* Air quality      — poor air quality raises respiratory risk.
* Humidity         — dry air (< 40%) favours airborne transmission and
                     irritates airways; very humid air (> 70%) favours mould.
* Temperature      — cold and very hot conditions add environmental stress.
* Recent trend     — a rising AQI means conditions are deteriorating.
* Prediction       — if reliability is Low, the assessment is flagged as
  reliability         less certain.

Outputs one of ``Low`` / ``Moderate`` / ``High`` with the contributing
factors, so the result can be explained rather than asserted.
"""

from aqi_utils import get_aqi_category

RISK_LEVELS = ['Low', 'Moderate', 'High']

# AQI category -> base points. Good/Moderate are low, Poor/Very Poor moderate,
# Severe/Hazardous high.
_AQI_POINTS = {0: 0, 1: 1, 2: 4, 3: 6, 4: 8, 5: 10}

_ADVISORY = {
    'Low': 'Conditions are favourable. Normal activities are fine; keep rooms ventilated.',
    'Moderate': (
        'Some risk factors are present. Ventilate rooms, maintain comfortable '
        'humidity, and take extra care if you are in a sensitive group.'
    ),
    'High': (
        'Conditions favour airborne transmission. Improve ventilation, avoid '
        'crowded indoor spaces, and take preventive measures.'
    ),
}

DISCLAIMER = (
    'Environmental risk indicator only. It is not a medical diagnosis and '
    'does not replace advice from a health professional.'
)


def _humidity_points(hum):
    if hum is None:
        return 0, None
    if hum < 30:
        return 2, f'Very low humidity ({hum:.0f}%) favours airborne transmission'
    if hum < 40:
        return 1, f'Low humidity ({hum:.0f}%) mildly favours airborne transmission'
    if hum <= 60:
        return 0, None
    if hum <= 70:
        return 1, f'Elevated humidity ({hum:.0f}%) may support mould growth'
    return 2, f'Very high humidity ({hum:.0f}%) favours mould and allergens'


def _temperature_points(temp):
    if temp is None:
        return 0, None
    if temp < 15:
        return 1, f'Cool temperature ({temp:.1f}°C) can increase respiratory stress'
    if temp > 35:
        return 1, f'High temperature ({temp:.1f}°C) adds heat stress'
    return 0, None


def _trend_points(trend):
    if trend == 'rising':
        return 1, 'Air quality is deteriorating (rising trend)'
    if trend == 'falling':
        return -1, 'Air quality is improving (falling trend)'
    return 0, None


def assess_risk(aqi, temp=None, hum=None, trend='stable', reliability=None):
    """
    Assess airborne-disease environmental risk.

    Parameters
    ----------
    aqi : float
        Predicted AQI value.
    temp, hum : float, optional
        Temperature (°C) and relative humidity (%).
    trend : {'rising', 'stable', 'falling'}
    reliability : dict, optional
        Output of :func:`reliability.assess`; used to flag uncertainty.

    Returns
    -------
    dict
        ``level``, ``score``, ``factors``, ``advisory`` and ``disclaimer``.
    """
    category = get_aqi_category(aqi)
    factors = [f'Air quality: AQI {aqi:.0f} (category {category})']
    score = _AQI_POINTS.get(category, 0)

    for points, factor in (
        _humidity_points(hum),
        _temperature_points(temp),
        _trend_points(trend),
    ):
        score += points
        if factor:
            factors.append(factor)

    score = max(0, score)

    if score <= 3:
        level = 'Low'
    elif score <= 7:
        level = 'Moderate'
    else:
        level = 'High'

    result = {
        'level': level,
        'score': score,
        'factors': factors,
        'advisory': _ADVISORY[level],
        'disclaimer': DISCLAIMER,
    }

    if reliability and reliability.get('label') == 'Low':
        result['factors'].append(
            'Air-quality prediction reliability is low; treat this assessment '
            'as indicative'
        )
        result['reliability_caveat'] = True

    return result

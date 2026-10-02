# mq135.py
"""
MQ-135 hardware calibration.

Stages
------
1. **R0** — the clean-air baseline resistance. Measured by sampling the sensor
   in clean air and taking the median Rs.
2. **Concentration** — either from the datasheet power law, or from a
   data-driven fit against a reference (co-location or reference gas).

The datasheet curves follow ``ppm = A * (Rs/R0) ** B``. Coefficients below are
the published MQ-135 datasheet values (as used by the MQSensorsLib project).

A fitted calibration is stored in ``gas_calibrator.pkl`` with
``trained_on='mq135:<gas>'``, so ``predictors.py`` knows to convert the raw
ADC reading to ``Rs/R0`` before applying it.
"""

import json
from datetime import datetime

import numpy as np

from calibration import mq135_rs_from_adc

PROFILE_FILE = 'mq135_profile.json'

# Published MQ-135 datasheet coefficients. B is negative: resistance falls as
# concentration rises.
DATASHEET_COEFFICIENTS = {
    'co':      {'a': 605.18, 'b': -3.937},
    'alcohol': {'a': 77.255, 'b': -3.180},
    'co2':     {'a': 110.47, 'b': -2.862},
    'toluene': {'a': 44.947, 'b': -3.445},
    'nh4':     {'a': 102.20, 'b': -2.473},
    'acetone': {'a': 34.668, 'b': -3.369},
}


class MQ135Profile:
    """Hardware configuration and clean-air baseline for one MQ-135 board."""

    def __init__(self, r0=None, rl_kohm=10.0, vcc=3.3, adc_max=4095.0,
                 gas='co2', source='datasheet', r0_samples=None):
        self.r0 = float(r0) if r0 is not None else None
        self.rl_kohm = float(rl_kohm)
        self.vcc = float(vcc)
        self.adc_max = float(adc_max)
        self.gas = gas if gas in DATASHEET_COEFFICIENTS else 'co2'
        self.source = source
        self.r0_samples = int(r0_samples) if r0_samples is not None else None

    @property
    def calibrated(self):
        """True once a clean-air R0 has been measured."""
        return self.r0 is not None and self.r0 > 0

    def rs(self, adc):
        """Sensor resistance (kOhm) for a raw ADC reading."""
        return mq135_rs_from_adc(adc, self.rl_kohm, self.vcc, self.adc_max)

    def ratio(self, adc):
        """Rs/R0 for a raw ADC reading."""
        if not self.calibrated:
            raise RuntimeError("R0 has not been measured; run the `r0` step first.")
        return self.rs(adc) / self.r0

    def ppm(self, adc):
        """Datasheet concentration estimate for the configured gas."""
        coefficients = DATASHEET_COEFFICIENTS[self.gas]
        ratio = np.clip(self.ratio(adc), 1e-6, None)
        return coefficients['a'] * np.power(ratio, coefficients['b'])

    def to_dict(self):
        return {
            'r0': self.r0,
            'rl_kohm': self.rl_kohm,
            'vcc': self.vcc,
            'adc_max': self.adc_max,
            'gas': self.gas,
            'source': self.source,
            'calibrated': self.calibrated,
            'r0_samples': self.r0_samples,
            'datasheet_coefficients': DATASHEET_COEFFICIENTS[self.gas],
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }

    def save(self, path=PROFILE_FILE):
        with open(path, 'w') as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path=PROFILE_FILE):
        with open(path) as handle:
            data = json.load(handle)
        return cls(
            r0=data.get('r0'),
            rl_kohm=data.get('rl_kohm', 10.0),
            vcc=data.get('vcc', 3.3),
            adc_max=data.get('adc_max', 4095.0),
            gas=data.get('gas', 'co2'),
            source=data.get('source', 'datasheet'),
            r0_samples=data.get('r0_samples'),
        )


def r0_from_adc_samples(adc_samples, rl_kohm=10.0, vcc=3.3, adc_max=4095.0):
    """Median clean-air R0 (kOhm) from raw ADC samples."""
    adc = np.asarray(adc_samples, dtype=float)
    adc = adc[np.isfinite(adc)]
    if adc.size == 0:
        raise ValueError("No valid ADC samples supplied.")
    rs = mq135_rs_from_adc(adc, rl_kohm, vcc, adc_max)
    rs = rs[np.isfinite(rs) & (rs > 0)]
    if rs.size == 0:
        raise ValueError("ADC samples produced no valid resistance values.")
    return float(np.median(rs)), int(adc.size)

# calibration.py
"""
Gas sensor calibration.

Two related pieces
------------------
1. :class:`GasCalibrator` — a data-driven, multivariate log-log calibration
   of a metal-oxide gas sensor against a reference concentration, with
   temperature and humidity compensation. This is the method used in the
   on-field calibration literature (De Vito et al., Sens. Actuators B, 2008).

       log(target) = b0 + b1*log(raw) + b2*temp + b3*hum

   For the UCI dataset the raw channel is ``PT08.S2(NMHC)`` and the target is
   reference benzene (C6H6), so the calibrated feature is a benzene-equivalent
   VOC concentration in ug/m^3.

2. MQ-135 helpers — convert a raw ADC reading from the live ESP32 node into a
   sensor resistance ``Rs``, a clean-air baseline ``R0``, and a ppm estimate
   from the datasheet power-law curve.

Important limitation
--------------------
The UCI calibration coefficients describe *that* sensor array, not the MQ-135.
The MQ-135 helpers are provisional until field data is available: they need a
clean-air ``R0`` measured on the actual board. Both paths output a
concentration in ug/m^3 so the model receives a physically meaningful,
comparable feature rather than an arbitrary ADC number.
"""

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class GasCalibrator:
    """Multivariate log-log calibration for a metal-oxide gas sensor."""

    RAW_FLOOR = 1e-6

    def __init__(self):
        self.model = None
        self.metrics = {}
        self.trained_on = None

    def _design(self, raw, temp, hum):
        raw = np.asarray(raw, dtype=float)
        temp = np.asarray(temp, dtype=float)
        hum = np.asarray(hum, dtype=float)
        return np.column_stack([
            np.log(np.clip(raw, self.RAW_FLOOR, None)),
            temp,
            hum,
        ])

    def fit(self, raw, temp, hum, target, trained_on=None):
        """Fit the calibration on rows with a valid reference target."""
        raw = np.asarray(raw, dtype=float)
        temp = np.asarray(temp, dtype=float)
        hum = np.asarray(hum, dtype=float)
        target = np.asarray(target, dtype=float)

        mask = (
            np.isfinite(raw) & np.isfinite(temp) & np.isfinite(hum)
            & np.isfinite(target) & (raw > 0) & (target > 0)
        )
        if mask.sum() < 10:
            raise ValueError(
                f"Not enough valid calibration rows ({int(mask.sum())}); need >= 10."
            )

        design = self._design(raw[mask], temp[mask], hum[mask])
        log_target = np.log(target[mask])

        self.model = LinearRegression().fit(design, log_target)
        predicted = np.exp(self.model.predict(design))

        self.metrics = {
            'n': int(mask.sum()),
            'r2_log': float(r2_score(log_target, self.model.predict(design))),
            'rmse': float(np.sqrt(mean_squared_error(target[mask], predicted))),
            'mae': float(mean_absolute_error(target[mask], predicted)),
            'coefficients': {
                'intercept': float(self.model.intercept_),
                'log_raw': float(self.model.coef_[0]),
                'temp': float(self.model.coef_[1]),
                'hum': float(self.model.coef_[2]),
            },
        }
        self.trained_on = trained_on
        return self

    def transform(self, raw, temp, hum):
        """Apply the calibration, returning concentration in ug/m^3."""
        if self.model is None:
            raise RuntimeError("GasCalibrator must be fitted before transform().")
        raw = np.atleast_1d(np.asarray(raw, dtype=float))
        temp = np.atleast_1d(np.asarray(temp, dtype=float))
        hum = np.atleast_1d(np.asarray(hum, dtype=float))
        return np.exp(self.model.predict(self._design(raw, temp, hum)))

    def is_fitted(self):
        return self.model is not None

    def save(self, path='gas_calibrator.pkl'):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path='gas_calibrator.pkl'):
        return joblib.load(path)


# ---------------------------------------------------------------------------
# MQ-135 live-node helpers
# ---------------------------------------------------------------------------

# Datasheet power-law defaults: ppm = A * (Rs/R0) ** B.
# B is negative because sensor resistance falls as gas concentration rises.
# These are provisional and must be re-fitted with a known reference gas.
MQ135_A = 116.6020682
MQ135_B = -2.769034857


def mq135_rs_from_adc(adc, rl_kohm=10.0, vcc=3.3, adc_max=4095.0):
    """
    Sensor resistance ``Rs`` (kOhm) from a raw ADC reading.

        Rs = RL * (Vcc - Vout) / Vout,   Vout = adc / adc_max * Vcc
    """
    adc = np.clip(np.asarray(adc, dtype=float), 1.0, adc_max - 1.0)
    vout = adc / adc_max * vcc
    return rl_kohm * (vcc - vout) / vout


def mq135_r0(rs_clean_air):
    """Clean-air baseline resistance ``R0`` (kOhm) from calibration samples."""
    rs = np.asarray(rs_clean_air, dtype=float)
    rs = rs[np.isfinite(rs) & (rs > 0)]
    if rs.size == 0:
        raise ValueError("No valid clean-air resistance samples.")
    return float(np.median(rs))


def mq135_ppm(rs, r0, a=MQ135_A, b=MQ135_B):
    """Concentration estimate (ppm) from the datasheet power-law curve."""
    ratio = np.asarray(rs, dtype=float) / float(r0)
    ratio = np.clip(ratio, 1e-6, None)
    return a * np.power(ratio, b)

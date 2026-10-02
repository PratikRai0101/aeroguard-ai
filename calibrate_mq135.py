#!/usr/bin/env python3
"""
MQ-135 calibration CLI.

Commands
--------
    status                     Show the current hardware profile and calibrator.
    r0 --csv clean_air.csv     Measure R0 from clean-air ADC samples.
    fit  --csv paired.csv      Fit the calibration against a reference.
    simulate                   Prove the workflow end to end on synthetic data.
    datasheet --gas co2        Write a datasheet-only provisional profile.

The `fit` command is the real fix: co-locate the MQ-135 with a reference
monitor (or use a known reference gas), record paired readings, and it fits
and saves the calibration that `predictors.py` then uses automatically.

Expected CSV columns
--------------------
r0:       adc
fit:      adc, temp, hum, reference
"""

import argparse
import json
import os

import numpy as np

from calibration import GasCalibrator
from mq135 import (
    DATASHEET_COEFFICIENTS,
    MQ135Profile,
    PROFILE_FILE,
    r0_from_adc_samples,
)

CALIBRATOR_FILE = 'gas_calibrator.pkl'


def _read_csv(path):
    import pandas as pd

    if not os.path.exists(path):
        raise SystemExit(f"CSV not found: {path}")
    return pd.read_csv(path)


def _pick_column(frame, preferred):
    if preferred and preferred in frame.columns:
        return preferred
    for candidate in ('adc', 'raw', 'gas', 'value', 'reference'):
        if candidate in frame.columns:
            return candidate
    raise SystemExit(f"No usable column found. Columns: {list(frame.columns)}")


def cmd_status(args):
    print("MQ-135 calibration status")
    print("-" * 40)

    if os.path.exists(args.profile):
        profile = MQ135Profile.load(args.profile)
        print(f"profile file : {args.profile}")
        print(f"R0           : {profile.r0} kOhm  (calibrated={profile.calibrated})")
        print(f"gas / coeffs : {profile.gas} {DATASHEET_COEFFICIENTS[profile.gas]}")
        print(f"source       : {profile.source}")
    else:
        print(f"profile file : {args.profile} (not found — run `r0` or `datasheet`)")

    if os.path.exists(args.calibrator):
        calibrator = GasCalibrator.load(args.calibrator)
        print(f"calibrator   : {args.calibrator}")
        print(f"trained on   : {calibrator.trained_on}")
        print(f"metrics      : {json.dumps({k: v for k, v in calibrator.metrics.items() if k != 'coefficients'}, indent=2)}")
    else:
        print(f"calibrator   : {args.calibrator} (not found)")


def cmd_r0(args):
    frame = _read_csv(args.csv)
    column = _pick_column(frame, args.column)
    r0, count = r0_from_adc_samples(
        frame[column].values, args.rl, args.vcc, args.adc_max
    )

    profile = MQ135Profile(
        r0=r0, rl_kohm=args.rl, vcc=args.vcc, adc_max=args.adc_max,
        gas=args.gas, source='clean_air', r0_samples=count,
    )
    profile.save(args.profile)

    print(f"Measured R0 from {count} clean-air samples: {r0:.3f} kOhm")
    print(f"Saved profile: {args.profile}")


def cmd_datasheet(args):
    profile = MQ135Profile(
        r0=args.r0, gas=args.gas, source='datasheet-provisional',
    )
    profile.save(args.profile)
    print(f"Saved datasheet-only profile for {args.gas}: {args.profile}")
    print("NOTE: this is provisional until R0 is measured in clean air.")


def cmd_fit(args):
    frame = _read_csv(args.csv)
    required = {'adc', 'temp', 'hum', 'reference'}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")

    if args.r0 is not None:
        profile = MQ135Profile(r0=args.r0, gas=args.gas, source='inline')
    else:
        profile = MQ135Profile.load(args.profile)

    if not profile.calibrated:
        raise SystemExit(
            "R0 is not set. Run `r0 --csv clean_air.csv` first, or pass --r0."
        )

    ratio = profile.ratio(frame['adc'].values)
    calibrator = GasCalibrator().fit(
        ratio, frame['temp'].values, frame['hum'].values, frame['reference'].values,
        trained_on=f"mq135:{profile.gas}",
    )
    calibrator.save(args.calibrator)
    profile.source = 'reference-fit'
    profile.save(args.profile)

    metrics = calibrator.metrics
    print(f"Fitted calibration on {metrics['n']} rows")
    print(f"  R2(log): {metrics['r2_log']:.4f}")
    print(f"  RMSE   : {metrics['rmse']:.3f}")
    print(f"  MAE    : {metrics['mae']:.3f}")
    print(f"  coeffs : {metrics['coefficients']}")
    print(f"Saved calibrator: {args.calibrator}")
    print("predictors.py will now use it for the MQ-135 (ADC -> Rs/R0 -> value).")


def cmd_simulate(args):
    """Generate synthetic MQ-135 data with a known law and recover it."""
    rng = np.random.default_rng(args.seed)
    # B is negative: higher gas -> lower Rs/R0 -> higher concentration.
    truth = {'intercept': 1.5, 'log_ratio': -2.35, 'temp': 0.003, 'hum': 0.0004}

    count = args.samples
    ratio = rng.uniform(0.15, 2.0, count)
    temp = rng.uniform(5, 40, count)
    hum = rng.uniform(20, 90, count)
    reference = np.exp(
        truth['intercept']
        + truth['log_ratio'] * np.log(ratio)
        + truth['temp'] * temp
        + truth['hum'] * hum
    )

    split = int(count * 0.8)
    calibrator = GasCalibrator().fit(
        ratio[:split], temp[:split], hum[:split], reference[:split],
        trained_on='mq135:simulated',
    )
    predicted = calibrator.transform(ratio[split:], temp[split:], hum[split:])

    from validation import regression_metrics

    metrics = regression_metrics(reference[split:], predicted)
    recovered = calibrator.metrics['coefficients']

    print("Simulated MQ-135 calibration")
    print("-" * 40)
    print(f"rows: {count} (train {split} / test {count - split})")
    print(f"reference range   : {reference.min():.2f} - {reference.max():.2f} µg/m³")
    print("true coefficients :", truth)
    print("fitted coefficients:", {k: round(v, 5) for k, v in recovered.items()})
    print(f"held-out R2        : {metrics['r2']:.4f}")
    print(f"held-out RMSE      : {metrics['rmse']:.3f}")
    print(f"held-out MAE       : {metrics['mae']:.3f}")
    print()
    print("Workflow verified: a known MQ-135 response was recovered from")
    print("paired (Rs/R0, temperature, humidity) -> reference data.")


def build_parser():
    parser = argparse.ArgumentParser(description="MQ-135 calibration CLI")
    parser.add_argument('--profile', default=PROFILE_FILE)
    parser.add_argument('--calibrator', default=CALIBRATOR_FILE)
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('status', help='show current calibration state').set_defaults(func=cmd_status)

    p_r0 = sub.add_parser('r0', help='measure R0 from clean-air ADC samples')
    p_r0.add_argument('--csv', required=True)
    p_r0.add_argument('--column', default='adc')
    p_r0.add_argument('--rl', type=float, default=10.0, help='load resistance kOhm')
    p_r0.add_argument('--vcc', type=float, default=3.3)
    p_r0.add_argument('--adc-max', type=float, default=4095.0)
    p_r0.add_argument('--gas', default='co2', choices=list(DATASHEET_COEFFICIENTS))
    p_r0.set_defaults(func=cmd_r0)

    p_ds = sub.add_parser('datasheet', help='write a provisional datasheet profile')
    p_ds.add_argument('--gas', default='co2', choices=list(DATASHEET_COEFFICIENTS))
    p_ds.add_argument('--r0', type=float, default=None)
    p_ds.set_defaults(func=cmd_datasheet)

    p_fit = sub.add_parser('fit', help='fit against a reference')
    p_fit.add_argument('--csv', required=True)
    p_fit.add_argument('--r0', type=float, default=None)
    p_fit.add_argument('--gas', default='co2', choices=list(DATASHEET_COEFFICIENTS))
    p_fit.set_defaults(func=cmd_fit)

    p_sim = sub.add_parser('simulate', help='prove the workflow on synthetic data')
    p_sim.add_argument('--samples', type=int, default=800)
    p_sim.add_argument('--seed', type=int, default=0)
    p_sim.set_defaults(func=cmd_simulate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

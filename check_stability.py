#!/usr/bin/env python3
"""
Report MQ-135 warm-up stability from recent sensor readings.

The MQ-135 drifts for 24-48 hours after power-on. This tool reads the recent
sensor history from the database and reports whether the gas channel has
settled, so you know when it is safe to measure R0.

    python check_stability.py
    python check_stability.py --hours 6 --threshold 3
"""

import argparse
import sqlite3

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Check MQ-135 warm-up stability")
    parser.add_argument('--db', default='aeroguard.db')
    parser.add_argument('--hours', type=float, default=6, help='look-back window')
    parser.add_argument('--threshold', type=float, default=3.0,
                        help='max %% drift over the last hour to call it stable')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    frame = pd.read_sql_query(
        "SELECT timestamp, gas FROM readings WHERE source='sensor' ORDER BY timestamp",
        conn,
    )
    conn.close()

    if frame.empty:
        raise SystemExit("No sensor readings in the database yet.")

    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    end = frame['timestamp'].max()
    frame = frame[frame['timestamp'] >= end - pd.Timedelta(hours=args.hours)]

    if len(frame) < 20:
        raise SystemExit("Not enough recent sensor readings to assess stability.")

    frame['bucket'] = frame['timestamp'].dt.floor('10min')
    trend = frame.groupby('bucket')['gas'].mean()

    print(f"MQ-135 warm-up stability (last {args.hours:g}h)")
    print("-" * 48)
    for ts, value in trend.items():
        print(f"  {ts.strftime('%m-%d %H:%M')}   gas={value:7.1f}")

    last_hour = trend[trend.index >= end - pd.Timedelta(hours=1)]
    if len(last_hour) < 2:
        raise SystemExit("\nNot enough data in the last hour to judge stability.")

    drift = last_hour.max() - last_hour.min()
    drift_pct = drift / last_hour.mean() * 100
    elapsed = frame['timestamp'].max() - frame['timestamp'].min()

    print("-" * 48)
    print(f"session span : {elapsed}")
    print(f"last-hour mean: {last_hour.mean():.1f}")
    print(f"last-hour drift: {drift:.1f} ({drift_pct:.1f}%)")

    if drift_pct <= args.threshold:
        print("\nVERDICT: STABLE — gas has settled over the last hour.")
        print("Next: take the sensor to CLEAN AIR and measure R0:")
        print("  python log_serial.py --seconds 120 --out clean_air.csv")
        print("  python calibrate_mq135.py r0 --csv clean_air.csv --rl <your_RL>")
    else:
        print("\nVERDICT: STILL DRIFTING — wait longer (target 24-48h total).")

    if elapsed < pd.Timedelta(hours=24):
        print(f"Note: the sensor has only run {elapsed}. The datasheet asks for 24-48h.")


if __name__ == '__main__':
    main()

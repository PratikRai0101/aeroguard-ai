#!/usr/bin/env python3
"""
Log raw ESP32 serial readings to a CSV file.

Used to capture the clean-air samples needed for MQ-135 R0 calibration:

    python log_serial.py --seconds 120 --out clean_air.csv
    python calibrate_mq135.py r0 --csv clean_air.csv

The CSV has columns: timestamp, temp, hum, gas. The `r0` command picks the
`gas` column automatically.

If the Streamlit dashboard is running it already holds the port; stop it
first (or it will steal some lines), then restart it afterwards.
"""

import argparse
import csv
import time

from serial_utils import describe_ports, open_serial


def main():
    parser = argparse.ArgumentParser(description="Log ESP32 serial data to CSV")
    parser.add_argument('--port', default=None, help='serial port (auto-detected if omitted)')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--seconds', type=int, default=60, help='how long to log')
    parser.add_argument('--out', default='sensor_log.csv')
    args = parser.parse_args()

    ser, port = open_serial(args.port, args.baud, timeout=2)
    if ser is None:
        print("No serial port could be opened. Detected ports:")
        for entry in describe_ports():
            print("  ", entry)
        raise SystemExit(1)

    print(f"Logging {port} for {args.seconds}s -> {args.out}")
    count = 0
    start = time.time()

    with open(args.out, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'temp', 'hum', 'gas'])

        while time.time() - start < args.seconds:
            line = ser.readline().decode('utf-8', 'ignore').strip()
            if not line or line.count(',') != 2:
                continue
            parts = [p.strip() for p in line.split(',')]
            try:
                temp, hum, gas = (float(p) for p in parts)
            except ValueError:
                continue

            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), temp, hum, gas])
            handle.flush()
            count += 1
            if count % 10 == 0:
                print(f"  {count} rows | latest: {temp}°C {hum}% gas={gas}")

    ser.close()
    print(f"Done: {count} rows written to {args.out}")


if __name__ == '__main__':
    main()

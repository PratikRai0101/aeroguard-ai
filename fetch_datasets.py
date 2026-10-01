#!/usr/bin/env python3
"""
Download the datasets AeroGuard AI uses for calibration and validation.

UCI Air Quality (De Vito et al., 2008):
    9,357 hourly records from a metal-oxide sensor array co-located with a
    certified reference analyzer. Used to calibrate the gas channel and to
    validate predictions against reference data.

Usage:
    python fetch_datasets.py
"""

import os
import urllib.request
import zipfile

UCI_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
DATA_DIR = 'data'
ZIP_PATH = os.path.join(DATA_DIR, 'air_quality.zip')
CSV_PATH = os.path.join(DATA_DIR, 'AirQualityUCI.csv')


def fetch_uci_air_quality(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH) and not force:
        print(f"✓ UCI dataset already present: {CSV_PATH}")
        return CSV_PATH

    print(f"Downloading {UCI_URL} ...")
    urllib.request.urlretrieve(UCI_URL, ZIP_PATH)

    print(f"Extracting to {DATA_DIR}/ ...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)

    if not os.path.exists(CSV_PATH):
        raise RuntimeError("Extraction finished but AirQualityUCI.csv was not found.")

    print(f"✓ Saved: {CSV_PATH}")
    return CSV_PATH


if __name__ == '__main__':
    fetch_uci_air_quality()

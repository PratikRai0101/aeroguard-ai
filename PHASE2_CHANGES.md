# Phase 2 — Calibration, Real Dataset & Reference Validation (Complete)

Implements Steps 3, 4 and 10 of the review plan. No 30-day data collection
was needed: the system now trains on a real, reference-validated public
dataset.

## The dataset decision

Instead of collecting data for a month, we use the **UCI Air Quality
dataset** (De Vito et al., 2008):

| Property | Value |
|---|---|
| Records | 9,357 hourly (8,991 usable with reference AQI) |
| Sensors | Metal-oxide array (PT08.S1–S5), temperature, humidity |
| Reference | Certified co-located analyzer: CO, NO₂, C₆H₆ (benzene) |
| Span | March 2004 → April 2005 (one full year) |
| License | Free for research use |

This is the canonical dataset for metal-oxide sensor calibration and
reference validation. It is fetched with `python fetch_datasets.py`.

## A finding that changed the target

The composite AQI in this data is driven by **NO₂ 81% of the time**, and the
VOC-class sensor correlates only **0.09** with NO₂. The apparent 0.85
correlation between gas and composite AQI is a spurious co-occurrence, not a
measurable relationship.

The sensor correlates **0.987 with the benzene (C₆H₆) sub-index**. So the
system now predicts a **VOC air-quality sub-index** — the thing an MQ-135 can
actually measure — and keeps the composite AQI as a reference column only.
This is stated explicitly rather than papering over it.

## Gas calibration

`calibration.py` fits a multivariate log-log model with temperature and
humidity compensation (the on-field calibration method from De Vito et al.):

```
log(target) = b0 + b1*log(raw) + b2*temp + b3*hum
```

| Calibration metric | Value |
|---|---|
| R² (log scale) | **0.985** |
| RMSE | **1.67 µg/m³** |
| MAE | **0.73 µg/m³** |
| Rows fitted | 7,192 |

The calibrated feature is a **benzene-equivalent VOC concentration (µg/m³)**,
not a raw ADC number. The MQ-135 helpers (`mq135_rs_from_adc`, `mq135_r0`,
`mq135_ppm`) provide the live-node path; they need a clean-air `R0` and a
reference gas before deployment.

## Reference validation (Step 10)

Chronological 80/20 split; the test set is the unseen final 1,799 hours.

| Model | Metric | Value |
|---|---|---|
| Random Forest | accuracy | **98.4%** |
| Linear Regression | MAE / RMSE | **9.65 / 11.46** |
| Linear Regression | R² / Pearson r | **0.969 / 0.991** |
| Linear Regression | category exact / within-one | **89.9% / 100%** |
| LSTM | validation accuracy | 73.7% |

For comparison, a constant-prediction baseline has MAE ≈ 160 and category
exact ≈ 3.6%.

## New modules

| File | Purpose |
|---|---|
| `dataset.py` | Loads UCI and legacy data into a canonical frame; computes the reference AQI |
| `calibration.py` | `GasCalibrator` (log-log + T/RH compensation) and MQ-135 ADC→Rs→ppm helpers |
| `validation.py` | MAE, RMSE, R², Pearson r, category agreement, confusion matrix |
| `training.py` | Shared dataset → split → calibration → metadata pipeline |
| `fetch_datasets.py` | Downloads the UCI dataset |

## Pipeline now

```
UCI sensor array ─┐
                  ├─> chronological split
temperature ──────┤
humidity ─────────┘
        │
        ▼
   gas calibration (R² 0.985)  ->  VOC concentration (µg/m³)
        │
        ▼
   preprocessing (validation + missing handling)
        │
        ▼
   RF / LSTM / Linear Regression  ->  VOC AQI + category
        │
        ▼
   validation vs certified reference (MAE, RMSE, r, category agreement)
```

## Known limitation (stated honestly)

The UCI calibration coefficients describe the UCI sensor array, not the
MQ-135. The live node needs its own clean-air `R0` and reference gas fit.
The model, pipeline and validation methodology are complete; only the
per-device calibration coefficients are outstanding. This is recorded in
`model_metadata.json` under `calibration.note`.

## Reproduce

```bash
python fetch_datasets.py          # download UCI dataset
python train_model.py             # RF + Linear Regression
python train_lstm.py              # LSTM
python -m unittest discover -s tests -v   # 26 tests
```

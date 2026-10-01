# AeroGuard AI — Panel Summary (one page)

**Project:** IoT Air Quality Prediction & Airborne Disease Risk Alert
**Status:** All 12 review steps complete. One field item remains.

## What the system does

An ESP32 node with a DHT22 (temperature, humidity) and an MQ-135 (gas/VOC)
feeds a pipeline that predicts air quality, assesses airborne-disease
environmental risk, explains the prediction, and answers questions in natural
language. A React Native app shows the result; a local Qwen model explains it.

```
Measure → Clean → Calibrate → Predict → Reliability → Risk → Explain → Alert → Validate → Chat
```

## Results (validated against a certified reference analyzer, unseen future data)

| Metric | Value |
|---|---|
| Gas calibration R² (log) / MAE | **0.985 / 0.73 µg/m³** |
| Random Forest accuracy | **98.4%** |
| Linear Regression MAE / RMSE | **9.65 / 11.46** |
| Linear Regression R² / correlation | **0.969 / 0.991** |
| AQI-category exact / within-one | **89.9% / 100%** |
| LSTM validation accuracy | **73.7%** |

Baseline (constant prediction): MAE ≈ 160, category exact ≈ 3.6%.

## How each review point was answered

1. **Hardware** — unchanged; already sufficient.
2. **Preprocessing** — one module used by training and inference.
3. **MQ-135 calibration** — log-log calibration with temperature/humidity
   compensation; R² = 0.985.
4. **Dataset** — 9,357 real hourly records (8,991 usable) replacing 768
   synthetic rows.
5. **Storage** — predictions now store risk, reliability and explanation.
6. **Models** — two inference defects fixed, chronological splits, retrained.
7. **Reliability** — prediction interval and category probability from
   validated error, not an invented "confidence %".
8. **Risk assessment** — Low/Moderate/High from AQI, humidity, temperature and
   trend; explicitly not a diagnosis.
9. **SHAP** — exact TreeSHAP factors behind each prediction.
10. **Validation** — MAE, RMSE, correlation and category agreement against a
    co-located certified analyzer.
11. **Dashboard** — prediction, reliability, risk, "why this prediction?",
    history and a validation screen.
12. **Chatbot** — now explains the ML result instead of producing it.

## Two decisions worth highlighting

* **The target is a VOC air-quality sub-index, not the composite AQI.** The
  composite index is NO₂-driven, and the VOC sensor correlates only 0.09 with
  NO₂. The sensor correlates 0.987 with the benzene sub-index, so we predict
  what the hardware can actually measure. This is stated openly.
* **We kept the 2008 UCI dataset after evaluating newer ones.** The newest
  co-located reference set (2025) had a VOC channel correlating 0.226 with
  reference AQI. UCI is the only free dataset with a metal-oxide array
  co-located with a certified analyzer, and it is still the calibration
  benchmark in 2024 literature. Rationale is in `DATASET_EVALUATION.md`.

## Remaining item

The physical MQ-135 needs its own clean-air `R0` and reference-gas fit before
live readings are calibrated. The method, code and validation are complete;
only the per-device coefficients are outstanding. This is a short field
calibration, not a development task.

## Reproduce

```bash
python fetch_datasets.py && python train_model.py && python train_lstm.py
python -m unittest discover -s tests -v     # 36 tests pass
```

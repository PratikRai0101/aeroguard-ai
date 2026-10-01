# Phase 4 — Mobile Dashboard & Chatbot Integration (Complete)

Implements Steps 11 and 12 of the review plan.

## Step 11 — Dashboard

The React Native app now shows the full result in one place.

**Dashboard tab** (`mobile/app/(tabs)/index.tsx`)

| Card | Content |
|---|---|
| Indoor Air Quality | the model-predicted AQI + category + advice |
| Prediction Reliability | High/Moderate/Low badge, category probability, 80% interval, sensor estimate |
| Airborne Disease Risk | Low/Moderate/High badge, contributing factors, advisory, disclaimer |
| Why this prediction? | top SHAP factors with direction and contribution |
| Temperature / Humidity / Gas / Mode | sensor readings |
| AQI History | last 12 readings, colour-coded by category |
| Outdoor AQI / Preventive measures / Recent readings | existing |

**Validation tab** (`mobile/app/(tabs)/validation.tsx`, new)

Shows the certified-reference validation: dataset size and period, gas
calibration R²/RMSE/MAE, Random Forest accuracy, Linear Regression
MAE/RMSE/R²/r, LSTM accuracy, category agreement (exact / within-one) and the
6×6 confusion matrix.

## Step 12 — Chatbot explains, does not predict

`backend/chat.py` now injects the prediction, reliability, risk level, risk
factors and top SHAP factors into the model context. The system prompt states
explicitly:

* the ML models and risk module produce the result;
* the assistant explains it and must not invent a different prediction, risk
  level or reliability;
* the risk level is an environmental indicator, not a diagnosis.

So "why is my risk high?" is answered from the SHAP factors and risk factors,
not guessed.

## Backend changes

| Endpoint | Change |
|---|---|
| `GET /api/stats` | now includes `prediction`, `future_prediction`, `predicted_aqi`, `reliability`, `risk`, `explanation`, and `sensor_aqi` |
| `GET /api/validation` | **new** — returns the trained validation report from `model_metadata.json` |

`predictors.assess()` was added: a **stateless** assessment for API callers.
It does not mutate the LSTM buffer, so polling `/api/stats` every few seconds
is safe. The stateful `predict_all()` remains for streaming dashboards.

## Coherence fix

The stored `readings.aqi` was a legacy estimate (`gas * 0.15`), which did not
match the calibrated model. The backend now displays the **model prediction**
as the air-quality value and keeps the raw estimate separately as
`sensor_aqi`. The mock generators were moved into the calibrated sensor domain
(metal-oxide response ≈ 400–2000) so the demo chain is consistent:

```
gas 900 → calibrated VOC 7.6 µg/m³ → model AQI 79 (Moderate), reliability High
```

## Verification

```
venv/bin/python -m unittest discover -s tests   # Ran 36 tests ... OK
cd mobile && npx tsc --noEmit                   # clean
```

End-to-end check: `/api/stats` returns prediction, reliability, risk and
explanation; `/api/validation` returns the dataset, calibration and metrics.

## Remaining

* **Live MQ-135 calibration** — the model is validated on the UCI reference
  sensor array. The physical MQ-135 still needs its own clean-air `R0` and a
  reference-gas fit before live readings are calibrated. Until then, live
  predictions are indicative.
* **LSTM SHAP** — optional; the Random Forest explanation covers the headline
  category.

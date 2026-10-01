# Phase 3 — Reliability, Airborne Risk & SHAP (Complete)

Implements Steps 7, 8 and 9 of the review plan.

## Step 7 — Prediction reliability

`reliability.py`. The panel warned against an invented "86% confidence", so
the reliability is **derived from validated error**, not asserted.

Two measures, both consequences of the held-out validation:

1. **Prediction interval** — `predicted ± z · residual_std`, where
   `residual_std` is the Linear Regression residual standard deviation on the
   test set (11.38 AQI points).
2. **Category probability** — the mass of `N(predicted, residual_std)` inside
   the predicted CPCB category band. A prediction near a category boundary is
   unreliable; one in the middle of a band is reliable.

Example (predicted AQI 188.8, category Poor):

```json
{
  "category_probability": 0.838,
  "label": "High",
  "interval": { "level": 0.8, "low": 174.2, "high": 203.4, "half_width": 14.6 },
  "residual_std": 11.38
}
```

If the model is retrained and its RMSE changes, this number changes with it.

## Step 8 — Airborne disease risk assessment

`risk.py`. A separate module that combines air quality with environmental
conditions and trend, and outputs **Low / Moderate / High** with the
contributing factors.

| Input | Effect |
|---|---|
| AQI category | base risk: Good/Moderate low, Poor/Very Poor moderate, Severe/Hazardous high |
| Humidity | < 40% favours airborne transmission; > 70% favours mould |
| Temperature | < 15 °C or > 35 °C adds environmental stress |
| Trend | rising AQI raises risk; falling lowers it |
| Prediction reliability | low reliability flags the assessment as indicative |

Example output:

```
level:    Moderate (score 5)
factors:  Air quality: AQI 189 (category 2)
          Low humidity (30%) mildly favours airborne transmission
advisory: Some risk factors are present. Ventilate rooms, ...
```

Every result carries an explicit disclaimer: **environmental risk indicator
only, not a medical diagnosis.**

## Step 9 — SHAP explainability

`explain.py`. The Random Forest is explained exactly with **TreeSHAP**.

Example for a "Poor" prediction:

| Factor | Value | Contribution |
|---|---|---|
| Gas/VOC response | 20.99 µg/m³ benzene-equiv. | +0.406 (increases) |
| Humidity | 30.0 % | +0.060 (increases) |
| Temperature | 18.0 °C | −0.044 (decreases) |

The factors and their signs come from the trained model, not from a
hand-written rule.

The LSTM is not explained: SHAP for recurrent models is approximate and
heavy, and the Random Forest produces the headline category. A KernelSHAP
surrogate can be added if required for the viva.

## Integration

`predictors.predict_all()` now returns:

```python
{
  'current':     { label, confidence, class_probabilities },
  'future':      { label, confidence },
  'trend':       { aqi, trend },
  'reliability': { category_probability, label, interval },
  'risk':        { level, score, factors, advisory, disclaimer },
  'explanation': { top_factors, method: 'TreeSHAP' },
}
```

`database.py` stores `risk_level`, `reliability_label`,
`reliability_probability` and `explanation` on every prediction. The columns
are added with a migration-safe `_ensure_column`, so existing databases keep
working.

## New modules

| File | Purpose |
|---|---|
| `reliability.py` | Prediction interval + category probability |
| `risk.py` | Airborne disease environmental risk |
| `explain.py` | TreeSHAP explanations |
| `tests/test_phase3.py` | Reliability, risk, SHAP and DB tests |

## Tests

```
venv/bin/python -m unittest discover -s tests -v
# Ran 36 tests ... OK
```

## Remaining for the dashboard phase

The engine produces reliability, risk and SHAP. Step 11 is to surface them in
the React Native app (prediction card, risk badge, "why this prediction?"
sheet, history graph, validation screen) and to extend the chatbot context
(Step 12) with the risk level, SHAP factors and reliability.

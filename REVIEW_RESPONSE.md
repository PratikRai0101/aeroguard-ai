# AeroGuard AI — Response to Review Panel Feedback

**Project:** AeroGuard AI — IoT Air Quality Prediction & Airborne Disease Risk Alert
**Document type:** Response to the 12-step improvement plan
**Status of codebase reviewed:** commit `e3e54e8` ("Fix mobile app…"), `backend/` + `mobile/` + core Python modules

---

## 0. Our position in one paragraph

We accept almost all of the panel's direction and will implement it. The 12 steps correctly identify the gap between a working demo and a defensible research system. Before committing to a schedule, we audited the existing code and found **three issues that the panel could not have seen from the outside**, and they affect the order and definition of Steps 3, 6, 7, and 10. We are raising them now rather than quietly patching around them, because they are exactly the kind of thing that collapses under viva questioning. The rest of this document maps every step to current status, states what we will change, and lists the decisions we need from the panel.

---

## 1. Step-by-step response

Legend: ✅ done · ⚠️ partially done · ❌ not done

| # | Panel request | Current status | Our response |
|---|---|---|---|
| 1 | Hardware (ESP32 + DHT22 + MQ-135) | ✅ Confirmed | No change. Existing hardware is sufficient. |
| 2 | Data preprocessing | ⚠️ Partial | `validate_reading()` + `OutlierDetector` exist in `aqi_utils.py` and are wired into the dashboards. Missing: a real module, smoothing, normalization, missing-value handling, and — critically — the same preprocessing is **not applied to the training dataset**. We will consolidate into `preprocessing.py` and run both training and inference through it. |
| 3 | MQ-135 calibration | ❌ Not done | Agreed, and it is more urgent than the panel realises — see §2.1. We will add Rs/R0 calibration with temperature/humidity compensation and retrain. |
| 4 | Increase dataset | ❌ 768 rows | `real_air_data.csv` has 768 hourly rows (the 769 figure counts the header). Agreed: insufficient. We will extend collection and add real sensor data. See §2.3 and §3. |
| 5 | Storage (SQLite) | ⚠️ Partial | `aeroguard.db` already stores `readings`, `predictions`, `alerts`. We will add columns/tables for calibrated gas, risk level, reliability, SHAP factors, and validation results. |
| 6 | ML models (RF + LSTM + LR) | ⚠️ Present, flawed | We agree no new model is needed. We found two real defects in the current inference path — see §2.2. Fix training, splits, and evaluation of the existing three. |
| 7 | Prediction confidence/error | ❌ Not done | Agreed, and we share the panel's warning: the current "confidence %" is only a class probability, not a validated reliability measure. We will relabel and replace it. See §2.2. |
| 8 | Airborne disease risk assessment ⭐ | ❌ Not done | Accepted in full. New separate module, framed as **environmental risk assessment — not diagnosis**. Hybrid rules + trend, outputs Low/Moderate/High. |
| 9 | SHAP explainability ⭐ | ❌ Not done | Accepted. `shap` is not currently installed. Add `TreeExplainer` for Random Forest, surrogate/KernelSHAP for LSTM, surface top contributing factors. |
| 10 | Proper validation ⭐⭐⭐ | ❌ Not done | Accepted, with a methodological caveat we must agree on before building — see §2.4. Compute MAE, RMSE, correlation, AQI-category agreement. |
| 11 | Improved mobile dashboard | ⚠️ Partial | Current app has Dashboard + Chat tabs only. Add prediction, risk, reliability, "why this prediction?", history graph, validation screen. |
| 12 | Keep Qwen chatbot | ✅ Already correct | The chatbot in `backend/chat.py` is already an explainer, not the prediction engine. We will extend its injected context with risk, SHAP factors, and reliability. |

---

## 2. Critical findings from our own audit

These are things the panel could not see from outside the code. We disclose them because they change the definition of "done."

### 2.1 The training data's "gas" column is synthetic — so Step 3 is not optional polish

`collect_real_data.py` does **not** contain any MQ-135 data. It invents the gas feature from other pollutants:

```python
# collect_real_data.py — estimate_gas_from_pollutants()
gas = (pm25*2.5) + (pm10*1.5) + (co2*0.3) + (no2*2.0) + (ozone*0.8)
gas = gas / 5
return max(50, min(1200, gas + np.random.normal(0, 20)))   # random noise
```

The AQI stored in the database is then derived back from gas by an arbitrary linear guess plus more random noise (`aqi_utils.py — AQIAnalyzer._estimate_pm25`):

```python
return gas * 0.15 + np.random.normal(0, 5)
```

Consequences we must state plainly:

1. The three ML models are trained on a **fabricated gas feature** that does not correspond to any physical MQ-135 output. Real sensor output will not match this distribution.
2. The AQI in `readings` is partly random for a given gas value, which makes exact reproducibility and validation impossible.
3. Therefore calibration (Step 3) is not a refinement — it is required to make the models physically meaningful at all.

**What we will do:** calibrate the MQ-135 (Rs/R0 in clean air, log-scale response, temperature/humidity compensation), recompute the AQI from a physically defensible input, and retrain all three models on the corrected pipeline.

### 2.2 Two live defects in the inference code

**Defect A — a constant is fed to the model at inference.**

```python
# predictors.py:77 and predictors.py:123
X = pd.DataFrame([[temp, hum, gas, 25.0]], columns=['temp', 'hum', 'gas', 'pm25'])
```

Every prediction passes `pm25 = 25.0` regardless of reality, while the models were trained on real varying `pm25`. The most important feature is effectively frozen at inference. This must be fixed before any confidence/error claim is credible.

**Defect B — class labels do not match the training label space.**

`train_model.py` trains the Random Forest on up to 6 classes (0–5: Good → Hazardous), and `train_lstm.py` trains the LSTM on 3. But the inference maps in `predictors.py:83` and `predictors.py:108` only decode 0–3 and 0–2 respectively. A "Severe" or "Hazardous" prediction would be silently mislabelled. We will unify the label space end-to-end.

We also note the current data split is random (`train_test_split(..., random_state=42)`), which **leaks future information** for a time-series problem. We will switch to time-ordered splits and report cross-validated results.

### 2.3 The database row count is misleading as evidence of dataset size

`aeroguard.db` contains 3,156 readings, which sounds healthy. But they were captured at roughly **3-second intervals over ~14 days** (2026-04-29 → 2026-05-13), and the values are smooth and synthetic-looking. 3,156 near-duplicate 3-second samples are not 3,156 units of information for hourly air-quality learning. The panel's "768 hourly records" figure is the honest denominator.

**What we will do:** collect genuinely varied real data across times of day and conditions, and report the dataset size in **distinct hours / distinct conditions**, not raw row count.

### 2.4 Step 10 needs a methodological decision before we build it

Comparing a **local indoor sensor** against an **outdoor government/Open-Meteo station** is not an apples-to-apples comparison, and a naive MAE/RMSE would be indefensible. The three honest options are:

- **(a) Co-location validation (strongest):** place our node near a reference monitor for a short campaign and compare like-for-like. Requires access to a reference site.
- **(b) Matched-period trend/category validation (practical):** compare category agreement and correlation over matched windows, clearly labelled as indicative, not absolute accuracy.
- **(c) Reference instrument purchase (costly):** a calibrated PM sensor co-located with the node.

**We recommend (b) as the default deliverable and (a) if a reference site can be arranged.** We need the panel's guidance here — see §5.

---

## 3. Revised pipeline (panel's process, corrected)

```
STEP 1   ESP32 + DHT22 + MQ-135                     [exists]
STEP 2   Ingest + Outlier/Noise Preprocessing       [consolidate + apply to training]
STEP 3   MQ-135 Calibration (Rs/R0 + T/H comp.)     [new — required]
STEP 4   Validated AQI from calibrated inputs       [replace gas*0.15+noise]
STEP 5   Large, real, time-stamped dataset          [collect + store]
STEP 6   ML Models: RF + LSTM + LR                  [fix defects, retrain]
STEP 7   Air Quality Prediction                     [exists]
STEP 8   Prediction Reliability                     [new — interval + validation RMSE]
STEP 9   Airborne Disease Risk Assessment           [new — Low/Moderate/High]
STEP 10  Health / Risk Alert                        [exists, extend to risk]
STEP 11  SHAP Explanation ("Why this prediction?")  [new]
STEP 12  React Native Dashboard                     [extend]
STEP 13  Qwen + Ollama natural-language explanation [exists, extend context]

Separately:  Sensor result  ↔  Reference data  →  MAE / RMSE / r / category agreement
```

Note the two corrections to the panel's ordering: AQI is now computed **from calibrated inputs** (Step 4) before it reaches the models, and reliability (Step 8) is backed by the validation work rather than asserted.

---

## 4. Delivery plan

| Phase | Work | Delivers |
|---|---|---|
| **P1 — Correctness** | Fix hardcoded `pm25`, unify label space, time-ordered splits, `preprocessing.py`, apply preprocessing to training data | Models are trustworthy; defects closed |
| **P2 — Calibration & data** | MQ-135 calibration, temperature/humidity compensation, extended real collection, DB schema extension | Step 3 + Step 4 + Step 5 |
| **P3 — Risk & reliability** | Airborne disease risk module, prediction interval / reliability indicator, retrain on corrected data | Steps 7–9 |
| **P4 — Explainability** | SHAP (RF tree explainer + LSTM surrogate), top-factor output | Step 11 |
| **P5 — Validation** | Reference-data validation module + report (MAE, RMSE, correlation, category agreement) | Step 10 |
| **P6 — Presentation** | Mobile dashboard cards, history, explanation, validation screen; chatbot context extension | Steps 11–13 |

We will confirm dates after §5 is answered, since data-collection duration and the validation method both set hard lower bounds on the schedule.

---

## 5. Decisions we need from the panel

1. **Validation reference:** can a co-located reference monitor be arranged (option a), or do we proceed with matched-period category/trend validation (option b)?
2. **Collection duration:** what minimum real-data collection period is acceptable? We propose **≥ 30 days** of real hourly data to give the models seasonal/diurnal variation.
3. **Reliability definition:** do you accept a **residual-based prediction interval** (e.g. predicted AQI ± validated RMSE band) as the reliability measure, instead of any invented "confidence %"?
4. **Risk framing:** do you accept a rules-plus-trend environmental risk model, explicitly labelled as **not a medical diagnosis**?
5. **Explainability scope:** is SHAP on the Random Forest sufficient for viva, or is LSTM explanation also required? LSTM SHAP is approximate and heavier.

---

## 6. Summary

- We accept Steps 2–12 as the plan.
- Step 3 (calibration) moves from "nice to have" to **blocking**, because the current training gas feature is synthetic.
- Step 6 is re-scoped from "train more" to "**fix two inference defects and retrain**."
- Step 7 will use a statistically justified reliability measure, not a class-probability relabelled as confidence.
- Step 10's method depends on a decision from the panel before we build it.
- Steps 1, 5 (base), and 12 already exist; Steps 8, 9, 10 are genuinely new.

We are ready to start Phase 1 immediately. Phase 2 onward depends on §5.

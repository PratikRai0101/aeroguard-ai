# Dataset Evaluation & Selection

**Question:** is the UCI Air Quality dataset (published 2008, data from
2004–2005) too old? Should we use something newer?

**Answer:** newer datasets exist, but none matches UCI for this task. We
downloaded and measured the main candidates. This document records the
evidence and the decision.

---

## 1. Requirements

The system needs a dataset that has **all** of:

1. A metal-oxide (MQ-135-class) gas sensor response.
2. Temperature and relative humidity.
3. A **co-located certified reference analyzer** — without this, Step 10
   (validation) cannot be done.
4. A free, research-permitted licence.

## 2. Candidates evaluated

| Dataset | Period | Gas sensor | Temp/Hum | Certified reference | Licence |
|---|---|---|---|---|---|
| **UCI Air Quality** (De Vito et al.) | 2004–05 | MOX array (PT08.S1–S5) | ✅ | ✅ ARPA analyzer | free, research |
| Valencia Molí del Sol | 2024–25 | ZPHS01B VOC | ✅ | ✅ regulatory AQMS | open (Zenodo) |
| Kaggle indoor office | 2023–24 | literal `MQ135_*` | ✅ | ❌ SGP40 `AirQ` only | Kaggle terms |
| Bangladesh IoT | 2017–22 | — (derived pollutants) | ❌ | ❌ | open (Mendeley) |
| QUANT (UK) | 2019–22 | commercial packages | ✅ | ✅ | open (Zenodo) |
| SensEURCity | 2020–21 | electrochemical | ✅ | ✅ | open (Zenodo) |

## 3. Measured results

We did not choose on age; we chose on measured signal.

| Dataset | Key correlation / calibration result |
|---|---|
| **UCI** | PT08.S2 ↔ reference benzene **r = 0.987**; calibration R²(log) **0.985** |
| Valencia 2024–25 | LCS VOC ↔ reference AQI **r = 0.226**; VOC+T+RH → NO₂ calibration **R² = 0.006**, r = 0.40 |
| Bangladesh | no gas sensor or temperature/humidity columns — unusable for the sensor pipeline |
| Kaggle indoor | ~500k MQ-135 records/year, but no certified reference — cannot validate |

The Valencia result is the decisive one: it is the newest co-located
reference dataset (published Sept 2025, data from 2024–2025), and its
low-cost VOC channel is a **weak** predictor of reference air quality. Using
it would make the project look current and perform worse.

## 4. Why UCI remains the right choice

1. It is the only free dataset with a **MOX sensor array co-located with a
   certified analyzer** — the exact setup Step 10 requires.
2. Its gas channel carries a strong, physically real VOC signal
   (r = 0.987 with reference benzene).
3. Age does not invalidate the calibration method. It is still the benchmark
   in current work: a 2024 *MDPI Sensors* calibration study uses it as
   "Dataset 1" alongside 2018–2020 datasets.
4. It is small (768 KB) and reproducible via `python fetch_datasets.py`.

## 5. Decision

* **Keep UCI** as the calibration and reference-validation dataset.
* The 768-row legacy `real_air_data.csv` is superseded (9,357 vs 768 rows;
  real temperature/humidity instead of constant defaults; real reference
  targets instead of synthetic gas).
* Revisit newer datasets only if the live MQ-135 node can be co-located with
  a reference monitor for a field calibration campaign.

## 6. How to state this at the viva

> "We evaluated current datasets including a 2024–25 co-located reference set.
> The newer low-cost VOC sensor correlated only 0.23 with reference air
> quality, whereas the UCI metal-oxide array correlates 0.99 with reference
> benzene. We chose UCI because it is the only free dataset that supports
> certified-reference validation, and it remains the calibration benchmark in
> 2024 literature."

That is a stronger answer than using a newer dataset that cannot be validated.

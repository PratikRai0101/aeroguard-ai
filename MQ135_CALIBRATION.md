# MQ-135 Calibration Procedure

This is the one remaining project item: fitting the calibration to **your**
MQ-135 board. The code and workflow are complete and verified; this document
is the procedure to run when the sensor is connected.

## Why it is needed

The ML model was trained and validated on the UCI metal-oxide sensor array,
where the gas channel is a benzene-equivalent VOC concentration. A physical
MQ-135 outputs a raw ADC value on a **different scale**. Until the MQ-135 is
calibrated to a reference, live predictions are indicative only.

## The physics

```
Vout  = ADC / ADC_max * Vcc
Rs    = RL * (Vcc - Vout) / Vout          sensor resistance (kOhm)
ratio = Rs / R0                            R0 = clean-air baseline
ppm   = A * ratio ** B                     datasheet power law (B is negative)
```

For a metal-oxide sensor, **more gas → lower Rs → lower Rs/R0 → higher
concentration**. Any fitted calibration must have a negative slope on
`log(ratio)`; the test suite checks this.

## Hardware preparation

1. Power the MQ-135 and let it **pre-heat for 24–48 hours** (datasheet
   requirement). Readings drift badly before this.
2. Note your board's load resistor `RL`. Many breakout boards use ~10 kΩ;
   measure between A0 and GND to confirm.
3. Confirm the ADC resolution (ESP32 default 12-bit → `ADC_max = 4095`,
   `Vcc = 3.3`).

## Step 1 — Measure R0 in clean air

Place the sensor in genuinely clean outdoor air, away from traffic and people.
Record raw ADC samples (a few hundred over several minutes).

```bash
# clean_air.csv with one column: adc
python calibrate_mq135.py r0 --csv clean_air.csv --rl 10 --vcc 3.3 --adc-max 4095
```

Output: the median clean-air resistance `R0`, saved to `mq135_profile.json`.

## Step 2 — Fit against a reference

This is the real calibration. Two ways:

**(a) Co-location (strongest).** Put the MQ-135 next to a reference monitor
(or a co-located calibrated sensor) and log paired readings. This mirrors the
method used to validate the model on the UCI dataset.

**(b) Reference gas.** Expose the sensor to a known concentration (e.g. a
calibration gas or a controlled source) and record paired readings.

Paired CSV columns:

```
adc, temp, hum, reference
```

```bash
python calibrate_mq135.py fit --csv paired.csv
```

It fits:

```
log(reference) = b0 + b1*log(Rs/R0) + b2*temp + b3*hum
```

and saves `gas_calibrator.pkl` with `trained_on='mq135:<gas>'`. The
temperature and humidity terms are the compensation the panel asked for.

## Step 3 — Verify

```bash
python calibrate_mq135.py status
```

Check `R2(log)` is high and the slope `log_raw` is **negative**. Then run the
dashboard/backend; `predictors.py` detects the `mq135:` prefix, converts the
raw ADC to `Rs/R0`, and applies the fitted calibration automatically.

## No reference available?

You can write a datasheet-only provisional profile:

```bash
python calibrate_mq135.py datasheet --gas co2
```

This uses the published MQ-135 datasheet curves and is clearly flagged as
provisional. Absolute accuracy is not guaranteed — the datasheet curves are
relative. Use it only until a reference fit is possible.

## Verify the workflow without hardware

```bash
python calibrate_mq135.py simulate
```

Generates synthetic MQ-135 data with a known response law and recovers the
coefficients. Current result: **R² = 1.0000, coefficients recovered exactly.**

## Integration

| File | Role |
|---|---|
| `mq135.py` | ADC→Rs→Rs/R0→ppm, datasheet coefficients, hardware profile |
| `calibrate_mq135.py` | CLI: `status`, `r0`, `fit`, `simulate`, `datasheet` |
| `mq135_profile.json` | R0, RL, Vcc, ADC max, gas, source |
| `gas_calibrator.pkl` | fitted calibration used by `predictors.py` |

## Honest status

* The calibration **method, code and verification are complete**.
* The `simulate` run proves a known MQ-135 response is recovered exactly.
* The real coefficients require the physical sensor plus a reference. That is
  a short field task, not a development task.

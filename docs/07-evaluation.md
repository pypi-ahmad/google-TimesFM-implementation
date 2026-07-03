# 07 - Evaluation

> Previous: [06 - Using TimesFM on Real Data](06-using-timesfm.md). Next: [08 - Troubleshooting](08-troubleshooting.md).

A forecast that "looks right" on one chart is not the same as a forecast
you can trust. This page covers how to actually measure forecast quality.

## Run it

```bash
uv run python examples/04_evaluation_backtest_example.py
```

This runs a **rolling backtest**: instead of one train/test split, it picks
several anchor points sliding back through history, forecasts from each,
and reports the average error across all of them.

## Why one split isn't enough

[06 - Using TimesFM on Real Data](06-using-timesfm.md) used a single
train/test split: months 1-108 predict months 109-144. That number could be
an unusually easy or unusually hard 36-month window. **Backtesting**
answers "how would this forecaster have performed if I'd used it
repeatedly over time?" by repeating the split at multiple points and
averaging.

```
Anchor 1: |----context----|--horizon--|
Anchor 2:      |----context----|--horizon--|
Anchor 3:           |----context----|--horizon--|
                                              ^ always predicting forward,
                                                never backward
```

The rule that makes this valid: **every anchor's context must end strictly
before its horizon begins, and the model must never see data from after the
anchor when forecasting from it.** This is the same no-leakage principle
from [05 - Data Format and Preprocessing](05-data-format-and-preprocessing.md),
applied across time instead of within one split.

## Metrics, defined plainly

All of these compare a forecast (`y_pred`) to what actually happened
(`y_true`) over a horizon:

| Metric | Formula (intuition) | Use when |
|---|---|---|
| **MAE** (Mean Absolute Error) | average of `|actual - predicted|` | You want errors in the original units (e.g. "off by 40 passengers on average"). Sensitive to scale -- don't compare MAE across series with very different magnitudes. |
| **RMSE** (Root Mean Squared Error) | square root of the average squared error | Like MAE but penalizes large misses more heavily. Use when big misses are disproportionately costly. |
| **MAPE** (Mean Absolute Percentage Error) | average of `|actual - predicted| / actual`, as a % | Scale-free, easy to explain to non-technical stakeholders. **Breaks down (blows up or divides by zero) when actual values are near zero.** |
| **WMAPE** (Weighted MAPE) | `sum(|actual - predicted|) / sum(|actual|)` | Scale-free like MAPE but far more robust to near-zero actuals, because the denominator is a sum over the whole horizon, not per-point. Prefer this over plain MAPE for anything intermittent or near zero. |

This repo's evaluation scripts use MAE and WMAPE for exactly this reason --
see the `wmape()` function in
[`examples/04_evaluation_backtest_example.py`](../examples/04_evaluation_backtest_example.py).

## Always compare against a baseline

A number in isolation ("WMAPE = 0.09") tells you almost nothing. Is that
good? You can't know without a reference point. Always report at least one
naive baseline alongside any model result:

- `naive_last` -- repeat the last observed value. The absolute floor of
  "did the model learn anything at all."
- `seasonal_naive` -- repeat the value from one full seasonal cycle ago
  (e.g. same month last year). A much stronger, fairer baseline for
  seasonal data -- if your model can't beat this, seasonality alone
  explains what it "learned."

[`examples/02_beginner_airline_passengers_forecast.py`](../examples/02_beginner_airline_passengers_forecast.py)
and [`examples/04_evaluation_backtest_example.py`](../examples/04_evaluation_backtest_example.py)
both report these alongside TimesFM for exactly this reason.

## Reading the backtest summary correctly

The script reports **mean and standard deviation** across anchors, not just
the mean. Observed on the reference environment for this repo (your exact
numbers may vary slightly by hardware/torch version):

```
                     mean     std  count
model
seasonal_naive_12  0.0835  0.0320      6
timesfm            0.0548  0.0255      6
```

- A lower mean is better.
- A high standard deviation means performance is inconsistent across time
  periods -- worth investigating *why* (a regime change? a holiday
  season?) before trusting the average blindly.

## Checking quantile calibration (not just the point forecast)

If you're using the quantile forecast (see
[04 - Core Concepts, section 5](04-core-concepts.md#5-quantile-forecasts-and-uncertainty)),
a well-calibrated p10-p90 band should contain the actual value roughly 80%
of the time across many backtest anchors. If the actual value falls outside
the band far more than 20% of the time, the model's uncertainty estimates
are miscalibrated for your data and should be treated with extra caution
for anything risk-sensitive (e.g. safety-stock or staffing decisions).

## Common beginner mistakes

- **Reporting a metric from a single lucky/unlucky anchor** as if it were
  the model's general performance. Always backtest across multiple
  anchors before concluding anything.
- **Comparing MAE across series of very different scale** (e.g. "40" for a
  small store, "4,000" for a large one) and concluding the small store's
  forecast is "better." Use a scale-free metric (WMAPE) for cross-series
  comparison.
- **Tuning `ForecastConfig` flags against the test/backtest window itself**
  until the numbers look good. That's leakage through the back door --
  decide your configuration on a separate validation period, then confirm
  once on a held-out test period you haven't touched.
- **Silently dropping anchors that don't fit the window** without
  reporting how many were dropped. If your data can only support 3 valid
  anchors instead of 6, say so -- a smaller, honest sample beats a padded
  one.

---

**Next:** [08 - Troubleshooting](08-troubleshooting.md).

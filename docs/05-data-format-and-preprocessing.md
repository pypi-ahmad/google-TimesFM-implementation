# 05 - Data Format and Preprocessing

> Previous: [04 - Core Concepts](04-core-concepts.md). Next: [06 - Using TimesFM on Real Data](06-using-timesfm.md).

TimesFM is not picky about *where* your data comes from (CSV, database,
API), but it is specific about the *shape* it expects once you're ready to
call it. This page covers that contract and the preprocessing you're
responsible for.

## The core input contract

```python
model.forecast(horizon=H, inputs=[series_1, series_2, ...])
```

- `inputs` is a **Python list**, one element per time series.
- Each element is a **1-D sequence of numbers** (NumPy array, or plain list
  of floats/ints) -- values only, in chronological order, no timestamps.
- Series in the same call **can have different lengths**. TimesFM does not
  require you to pad or truncate to a common length.
- Values should be **raw** (not pre-normalized) -- see
  [04 - Core Concepts, section 4](04-core-concepts.md#4-internal-normalization-dont-normalize-your-data-yourself).

Timestamps never enter the model directly. *You* are responsible for
knowing what interval each value represents (hourly, daily, monthly) --
TimesFM 2.5 has no `freq` parameter to declare it explicitly (see
[04 - Core Concepts, section 8](04-core-concepts.md#8-why-the-frequency-parameter-disappeared)).
This means the burden of getting the sampling interval *regular and
consistent* falls entirely on your preprocessing.

## Going from a raw table to model input

Real data rarely arrives as a clean array. The typical path, shown in full
in [`examples/02_beginner_airline_passengers_forecast.py`](../examples/02_beginner_airline_passengers_forecast.py)
and at larger scale in this repo's [`notebooks/`](../notebooks/):

1. **Load** the raw table (CSV, SQL query result, API response).
2. **Parse timestamps** into an actual datetime type
   (`pd.to_datetime(...)`), never left as strings.
3. **Aggregate to one row per time step** if your raw data is at a finer
   grain than you want to forecast (e.g. individual transactions -> daily
   totals).
4. **Reindex onto a complete, regular calendar** so there are no missing
   dates -- see below.
5. **Extract the value column as a `float32` NumPy array.** This is what
   you pass as one element of `inputs`.

## Handling missing values and irregular sampling

TimesFM has no built-in gap detection or imputation -- if your dates skip a
day, the model has no way to know a day was skipped; it just sees the
values you gave it back-to-back as if they were consecutive. **This is a
preprocessing responsibility, not something the model does for you.**

Standard approach:

```python
full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
series = series.reindex(full_index)
```

Then choose a fill strategy deliberately:

| Situation | Reasonable fill | Why |
|---|---|---|
| A handful of missing days in an otherwise dense series | Forward-fill (`.ffill()`) or linear interpolation | Preserves local trend without inventing structure |
| A genuine "zero activity" day (e.g. a closed store) | Fill with `0`, not forward-fill | Forward-filling would fabricate nonzero activity that didn't happen |
| Long stretches of missing data | Consider excluding that series/period rather than filling | Interpolating over long gaps manufactures a trend that isn't real |

> **Common beginner mistake:** silently forward-filling large gaps and then
> being surprised the model's forecast looks unnaturally smooth right after
> the gap -- the smoothness is an artifact of your fill, not a model
> failure.

## Minimum series length

There's no hard-coded minimum enforced by the API, but practically: a
series shorter than one full seasonal cycle (e.g. fewer than 12 points for
monthly data with yearly seasonality) gives the model no way to detect that
seasonality. Match your context length to the pattern you actually need
captured -- see [04 - Core Concepts, section 3](04-core-concepts.md#3-context-and-horizon).

## Batching multiple series

Because `inputs` accepts a list, forecasting many series is one call, not a
loop:

```python
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[store_a_sales, store_b_sales, store_c_sales],  # different lengths OK
)
# point_forecast[0] -> store_a's forecast, point_forecast[1] -> store_b's, ...
```

This is both simpler and faster than calling `.forecast()` once per series
-- see it in action across many anchors in
[`examples/04_evaluation_backtest_example.py`](../examples/04_evaluation_backtest_example.py).

## Covariate format (XReg)

If you're using `forecast_with_covariates()` (see
[04 - Core Concepts, section 6](04-core-concepts.md#6-covariates--xreg)),
each covariate array must span **context length + horizon length** --
because a covariate is only useful if you also know (or can plan/estimate)
its value during the forecast window, not just historically:

```python
dynamic_numerical_covariates = {
    "price": [price_series_covering_context_and_horizon, ...],
}
```

See [`examples/03_covariates_xreg_example.py`](../examples/03_covariates_xreg_example.py)
for a complete worked example, including why the covariate arrays are
longer than the `inputs` arrays.

## Data leakage: the mistake that invalidates everything else

**Data leakage** means letting information from the future influence a
forecast for the past -- and it silently makes results look better than
they'll ever be in production. Two leakage traps specific to this workflow:

1. **Fitting any preprocessing statistic (mean, min/max, fill values) on
   the *entire* series** including the period you're trying to forecast,
   then using that statistic to prepare the context window. Always derive
   preprocessing decisions only from data strictly before your forecast
   anchor point.
2. **Using a covariate that wouldn't actually be known at forecast time.**
   Price *for the day you're pricing* is fine (you set it); tomorrow's
   actual foot traffic is not (you don't know it yet) -- only use
   covariates you could genuinely have in hand when the forecast is made.

[07 - Evaluation](07-evaluation.md) covers how backtesting design prevents
leakage across time.

## Check your understanding

- [ ] Why does TimesFM 2.5 not need a `freq` argument, and what does that
      shift onto you as the data preparer?
- [ ] What's wrong with forward-filling a week-long gap caused by a store
      being closed?
- [ ] Why must a covariate array be longer than the corresponding `inputs`
      array?

---

**Next:** [06 - Using TimesFM on Real Data](06-using-timesfm.md).

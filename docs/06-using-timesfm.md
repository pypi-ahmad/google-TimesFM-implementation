# 06 - Using TimesFM on Real Data

> Previous: [05 - Data Format and Preprocessing](05-data-format-and-preprocessing.md). Next: [07 - Evaluation](07-evaluation.md).

Time to put chapters 03-05 together on a real dataset, end to end.

## Run it

```bash
uv run python examples/02_beginner_airline_passengers_forecast.py
```

**Dataset:** the classic monthly airline passenger totals, January 1949 to
December 1960 (144 monthly values), from Box, G. E. P., & Jenkins, G. M.
(1976), *Time Series Analysis: Forecasting and Control* -- one of the most
widely used teaching datasets in time series analysis (it's the built-in
`AirPassengers` dataset in R). It's bundled at
[`examples/data/airline_passengers.csv`](../examples/data/airline_passengers.csv)
so this example needs no downloads beyond the model weights.

**Expected output:** a metrics table, then a saved plot
(`examples/02_forecast.png`). Observed on the reference environment for
this repo (Python 3.14, torch 2.12, CUDA GPU):

```
Context: 108 months, forecasting next 36 months.

Loading TimesFM 2.5 (200M, PyTorch)...
model                      MAE    MAPE %
naive_last               94.94     19.89
seasonal_naive_12        60.08     13.19
timesfm_zero_shot        51.68     11.14

Saved plot: examples/02_forecast.png

Success criteria: timesfm_zero_shot MAE should be lower than naive_last.
```

The two naive baselines are pure arithmetic on fixed data, so their numbers
are deterministic and will match exactly on any machine. TimesFM's exact
numbers can vary slightly by hardware/torch version/CPU-vs-GPU, but it
should beat both baselines, comfortably, every time on this dataset.

## Walking through the workflow

Open [`examples/02_beginner_airline_passengers_forecast.py`](../examples/02_beginner_airline_passengers_forecast.py).
This is the same five-step shape described in
[05 - Data Format and Preprocessing](05-data-format-and-preprocessing.md),
applied concretely:

1. **Load and validate.** `load_series()` reads the CSV and asserts it got
   exactly 144 rows -- a cheap sanity check that catches a corrupted or
   truncated download immediately, rather than letting a silently-short
   series produce a confusing downstream error.
2. **Split context / future.** The first 108 months (9 years) become the
   context the model sees; the last 36 months (3 years) are held out as
   ground truth to score against. This is a single train/test split -- the
   simplest possible evaluation, good for learning, but see
   [07 - Evaluation](07-evaluation.md) for why production evaluation needs
   more than one split.
3. **Load and compile TimesFM**, exactly as in
   [03 - First Forecast](03-first-forecast.md), with `infer_is_positive=True`
   added because passenger counts can never be negative -- a small but
   meaningful use of domain knowledge.
4. **Forecast**, passing the raw (unnormalized) context values.
5. **Compare against baselines.** `naive_last` (repeat the last value) and
   `seasonal_naive_12` (repeat the same month from a year ago) are
   deliberately simple. **Why bother with baselines this weak?** Because
   "beats a coin flip" is a real bar a forecaster must clear before it's
   useful at all -- if TimesFM couldn't beat `seasonal_naive_12` on a
   clearly seasonal series like this one, that would be a serious red
   flag, not a subtle one. Always compare against the simplest reasonable
   baseline before trusting a fancier model's numbers.

## Why this matters: quantiles in the plot

The saved plot shades the region between the 10th and 90th percentile
forecasts (`q10`/`q90`), not just the point forecast line. This band is
what [04 - Core Concepts, section 5](04-core-concepts.md#5-quantile-forecasts-and-uncertainty)
described in the abstract -- here you can see concretely that TimesFM's
uncertainty band widens further into the forecast horizon, which is the
expected, honest behavior: confidence should degrade the further ahead you
predict.

## Common beginner mistakes on this page

- **Judging the model from the printed MAE alone**, without looking at the
  plot. A model can have a "good" average error while being systematically
  wrong at seasonal peaks -- always eyeball the plot too.
- **Assuming a good result on this one series generalizes.** This is one
  series, one split. [07 - Evaluation](07-evaluation.md) is precisely about
  not overtrusting a single result like this one.
- **Re-running with a different `CONTEXT_LEN`/`HORIZON`  without
  re-checking `max_context`/`max_horizon` in `ForecastConfig`.** They must
  stay consistent, or you'll hit the compiled limits described in
  [03 - First Forecast](03-first-forecast.md).

## Going further: the applied case studies

Once this workflow feels routine, [`notebooks/`](../notebooks/) has ten
full, real-world case studies built on the same core APIs -- retail
demand, electricity load, hospital patient volume, ATM cash demand, cloud
capacity, airline route demand, warehouse orders, website traffic, and
financial transaction volume -- each going further into covariates,
rolling backtests, and turning forecasts into operational decisions. See
[`notebooks/README.md`](../notebooks/README.md) for how they relate to
this beginner path before diving in.

---

**Next:** [07 - Evaluation](07-evaluation.md) -- how to know if a forecast
is actually good, not just plausible-looking.

# Examples

Four standalone, runnable scripts, ordered by difficulty. Each is a single
file with no hidden setup beyond this project's environment (see
[`../docs/02-installation.md`](../docs/02-installation.md)). Run them from
the repo root with `uv run python examples/<name>.py`.

| # | Script | What it teaches | Needs |
|---|--------|------------------|-------|
| 1 | [`01_minimal_synthetic_forecast.py`](01_minimal_synthetic_forecast.py) | The smallest possible TimesFM call: load model, compile, forecast. | Nothing but the model download. |
| 2 | [`02_beginner_airline_passengers_forecast.py`](02_beginner_airline_passengers_forecast.py) | A full but small real workflow: load real data, forecast, compare to baselines, plot. | Bundled dataset (`data/airline_passengers.csv`). |
| 3 | [`03_covariates_xreg_example.py`](03_covariates_xreg_example.py) | When and how to add external signals (price, promotions) with XReg. | `timesfm[xreg]` extra (already in this project). |
| 4 | [`04_evaluation_backtest_example.py`](04_evaluation_backtest_example.py) | Rolling backtesting -- how to evaluate a forecaster honestly instead of trusting one split. | Bundled dataset. |

Each script has a companion doc in [`../docs/`](../docs/) that explains the
*why* behind the code; the scripts themselves focus on the *how*.

## After these four

The [`../notebooks/`](../notebooks/) directory contains ten full, real-dataset
case studies (retail, electricity, healthcare, aviation, etc.) built on the
same APIs shown here, at production-script scale rather than teaching-script
scale. See [`../notebooks/README.md`](../notebooks/README.md) before diving
in -- they assume everything in this `examples/` directory already makes
sense.

## `data/`

- `airline_passengers.csv` -- the classic monthly airline passenger totals,
  1949-1960 (144 rows). Public-domain dataset from Box, G. E. P., & Jenkins,
  G. M. (1976), *Time Series Analysis: Forecasting and Control*; widely
  redistributed for teaching (e.g. R's built-in `AirPassengers`). Bundled
  here so Tiers 2 and 4 run with no network access beyond the one-time model
  download.

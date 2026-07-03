# Exercise 01: First CSV Forecast (Timestamps, Gaps, Plot)

Goal: take a real-looking `(timestamp, value)` CSV, turn it into the exact
`timesfm.forecast()` input shape, and produce a plot with history + forecast.

This exercise is about preprocessing correctness, not model tricks.

## Data

Use the bundled CSV:

- `exercises/data/daily_sales_with_gaps.csv`

It contains:

- a `ds` timestamp column (daily frequency)
- a `y` numeric value column
- intentionally missing dates (gaps) and a few missing values

## Tasks

1. Load the CSV with pandas.
2. Parse `ds` as a datetime and set it as the index.
3. Reindex to a complete daily calendar from min to max date.
4. Fill missing values with a deliberate strategy:
   - short gaps: forward-fill is acceptable for this toy dataset
5. Split into:
   - context: first 60 days
   - horizon: next 14 days
6. Load TimesFM and forecast:
   - compile with `max_context=60`, `max_horizon=14`
   - call `forecast(horizon=14, inputs=[context_values])`
7. Plot:
   - context values
   - actual future values (ground truth)
   - TimesFM point forecast
   - optionally shade p10–p90

## Success Criteria

- Your script runs with no traceback.
- Forecast arrays have shapes:
  - point: `(1, 14)`
  - quantiles: `(1, 14, 10)` if you enabled the quantile head
- Your plot clearly shows the split point and the forecast window.

## Common Mistakes

- Passing `(timestamp, value)` rows directly to TimesFM. TimesFM only takes
  *values* (1-D numeric arrays). Timestamps are preprocessing-only.
- Forgetting to fill gaps. If you skip dates, the model can’t know they
  existed; it treats values as evenly spaced.

## Compare With

- Solution: `exercises/solutions/01_first_csv_forecast_solution.py`
- Data-format chapter: `docs/05-data-format-and-preprocessing.md`


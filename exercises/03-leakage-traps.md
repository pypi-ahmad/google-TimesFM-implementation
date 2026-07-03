# Exercise 03: Leakage Traps (How To Accidentally Lie To Yourself)

Goal: demonstrate (on purpose) how a small evaluation mistake can make a
forecast look better than it will be in production, then fix it.

This is the most important “mastery” skill in forecasting: *valid
evaluation design*.

## Data

Use the airline passengers series:

- `examples/data/airline_passengers.csv`

## Tasks

1. Implement a rolling backtest (you can reuse your Exercise 02 structure).
2. Create two evaluation modes:
   - **correct**: the forecast is compared to the true future window
   - **leaky**: you accidentally let future information influence the
     reported error (example leakage patterns below)
3. Print both results and explain why one is invalid.

### Two leakage patterns to choose from

Pick one:

1. **Peeking at the future to choose settings**:
   - try multiple `max_context` values
   - choose the one with the best backtest score on the same backtest window
   - report that score as if it generalizes

2. **Using future values in preprocessing**:
   - add artificial missing values to the series
   - “fill” them using statistics computed from the *full* series
   - backtest as if that fill would be available at prediction time

## Success Criteria

- Your script prints two summaries:
  - one valid
  - one artificially improved by leakage
- You clearly state which is valid and why.

## Compare With

- Solution: `exercises/solutions/03_leakage_traps_solution.py`
- Data leakage discussion: `docs/05-data-format-and-preprocessing.md`


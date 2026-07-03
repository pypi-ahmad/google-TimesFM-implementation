# Exercise 02: Rolling Backtest (The Evaluation Muscle)

Goal: implement a rolling backtest loop that produces an honest summary
statistic, not a single lucky split.

Use the bundled airline passengers series because it is small, clean, and
already used in the beginner examples.

## Data

- `examples/data/airline_passengers.csv` (144 points, monthly)

## Tasks

1. Load the series as a `float32` NumPy array.
2. Choose backtest settings:
   - `context_len = 36` (months)
   - `horizon = 12` (months)
   - `n_anchors = 6`
   - `step = 6` (months between anchors)
3. For each anchor:
   - slice context and future
   - forecast with TimesFM
   - compute a baseline (seasonal naive, period=12)
   - score both with WMAPE
4. Aggregate results:
   - mean/std across anchors per model
5. Print a small table with per-anchor scores and the summary.

## Success Criteria

- No leakage: the model only sees data strictly before each anchor.
- You report mean/std, not just mean.
- Your WMAPE function is stable (no divide-by-zero).

## Compare With

- Solution: `exercises/solutions/02_rolling_backtest_solution.py`
- Reference example: `examples/04_evaluation_backtest_example.py`
- Evaluation chapter: `docs/07-evaluation.md`


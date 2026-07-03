# Exercises: From “I Read It” to “I Can Do It”

These exercises are optional, but recommended. The main docs (`docs/00` to
`docs/10`) explain TimesFM; these exercises make you *use* it enough times
that it becomes routine.

Rules:

- Do the exercise first.
- Then run the matching solution to compare outputs and structure.
- If your result differs, focus on *why*, not just “making it match”.

## Setup

The exercises are designed to run in the **beginner path environment**:

```bash
uv venv
source .venv/bin/activate
uv sync --locked
```

## How To Use

1. Read the exercise page:
   - [01 - First CSV Forecast](01-first-csv-forecast.md)
   - [02 - Rolling Backtest](02-rolling-backtest.md)
   - [03 - Leakage Traps](03-leakage-traps.md)
2. Implement it in your own scratch script (or a notebook).
3. Compare with the solution script in `exercises/solutions/`.

## Solutions (Runnable)

Run from the repo root:

```bash
uv run python exercises/solutions/01_first_csv_forecast_solution.py
uv run python exercises/solutions/02_rolling_backtest_solution.py
uv run python exercises/solutions/03_leakage_traps_solution.py
```


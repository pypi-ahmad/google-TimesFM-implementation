"""Solution: Exercise 03 (leakage traps).

This demonstrates "peeking" leakage: choosing a configuration by optimizing
on the same backtest window you then report, which produces an overly
optimistic score.

Run:
    uv run python exercises/solutions/03_leakage_traps_solution.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import timesfm


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).sum() / (np.abs(y_true).sum() + 1e-8))


def seasonal_naive(context: np.ndarray, horizon: int, period: int = 12) -> np.ndarray:
    reps = int(np.ceil(horizon / period))
    return np.tile(context[-period:], reps)[:horizon]


def backtest(series: np.ndarray, context_len: int, horizon: int, n_anchors: int, step: int) -> float:
    """Return mean WMAPE for TimesFM across anchors for a chosen context length."""
    max_start = len(series) - context_len - horizon
    starts = [max_start - i * step for i in range(n_anchors)]
    starts = [s for s in starts if s >= 0]
    if not starts:
        raise RuntimeError("Series too short for chosen backtest settings.")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    contexts = [series[s : s + context_len] for s in starts]
    futures = [series[s + context_len : s + context_len + horizon] for s in starts]
    point, _ = model.forecast(horizon=horizon, inputs=contexts)

    scores = [wmape(futures[i], point[i]) for i in range(len(starts))]
    return float(np.mean(scores))


def main() -> None:
    torch.set_float32_matmul_precision("high")

    series = pd.read_csv(Path("examples/data/airline_passengers.csv"))["Passengers"].to_numpy(
        dtype=np.float32
    )

    horizon = 12
    n_anchors = 6
    step = 6

    # Correct-ish: pick one reasonable context length *a priori* and report it.
    context_len = 36
    correct_mean = backtest(series, context_len=context_len, horizon=horizon, n_anchors=n_anchors, step=step)

    # Leaky: try several context lengths and report the best on the same window.
    candidates = [24, 30, 36, 48, 60]
    scores = {c: backtest(series, context_len=c, horizon=horizon, n_anchors=n_anchors, step=step) for c in candidates}
    best_c = min(scores, key=scores.get)
    leaky_best_mean = scores[best_c]

    print("Correct evaluation (choose config without peeking):")
    print(f"- context_len={context_len}, mean wmape={correct_mean:.4f}")
    print("\nLeaky evaluation (choose config by optimizing on the same backtest window):")
    for c in candidates:
        print(f"- context_len={c:>2}, mean wmape={scores[c]:.4f}")
    print(f"=> reported best (INVALID): context_len={best_c}, mean wmape={leaky_best_mean:.4f}")

    print(
        "\nWhy this is invalid: you used the backtest window as a tuning target, "
        "so the best number is partly luck. In real work, pick config on a "
        "separate validation period, then report once on a held-out test period."
    )

    # Baseline reminder: always compare to something simple.
    context = series[:context_len]
    naive_pred = seasonal_naive(context, horizon=horizon)
    print(f"\nBaseline check (single split, seasonal naive): wmape={wmape(series[context_len:context_len+horizon], naive_pred):.4f}")


if __name__ == "__main__":
    main()


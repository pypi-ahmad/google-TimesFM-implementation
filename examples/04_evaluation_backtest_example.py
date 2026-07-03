"""Tier 4 - Proper backtesting: don't trust a single train/test split.

One forecast on one split can be lucky or unlucky. This script runs a
*rolling* backtest -- several anchor points slid across history -- so the
reported error is an average over multiple independent tests, the standard
way to evaluate a forecaster honestly.

Companion doc: docs/07-evaluation.md

Run:
    uv run python examples/04_evaluation_backtest_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import timesfm

torch.set_float32_matmul_precision("high")

DATA_PATH = Path(__file__).parent / "data" / "airline_passengers.csv"

CONTEXT_LEN = 36     # 3 years of history per forecast
HORIZON = 12         # forecast 1 year ahead
N_ANCHORS = 6        # number of rolling backtest windows
STEP = 6             # months between anchors


def load_series() -> np.ndarray:
    df = pd.read_csv(DATA_PATH)
    return df["Passengers"].to_numpy(dtype=np.float32)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE: robust to near-zero actuals, unlike plain MAPE."""
    return float(np.abs(y_true - y_pred).sum() / (np.abs(y_true).sum() + 1e-8))


def seasonal_naive(context: np.ndarray, horizon: int, period: int = 12) -> np.ndarray:
    reps = int(np.ceil(horizon / period))
    return np.tile(context[-period:], reps)[:horizon]


def main() -> None:
    series = load_series()

    # Walk backwards from the end of the series so every anchor has a full
    # future window to score against -- this is what "no leakage" means in
    # practice: the model only ever sees data strictly before the anchor.
    max_start = len(series) - CONTEXT_LEN - HORIZON
    anchor_starts = [max_start - i * STEP for i in range(N_ANCHORS)]
    anchor_starts = [s for s in anchor_starts if s >= 0]
    if not anchor_starts:
        raise RuntimeError(
            "Series too short for the chosen CONTEXT_LEN/HORIZON/N_ANCHORS. "
            "Reduce one of them."
        )

    print("Loading TimesFM 2.5 (200M, PyTorch)...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_LEN,
            max_horizon=HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    contexts = [series[s : s + CONTEXT_LEN] for s in anchor_starts]
    futures = [series[s + CONTEXT_LEN : s + CONTEXT_LEN + HORIZON] for s in anchor_starts]

    # One batched call for all anchors, rather than N separate calls.
    point_forecasts, _ = model.forecast(horizon=HORIZON, inputs=contexts)

    rows = []
    for i, start in enumerate(anchor_starts):
        future = futures[i]
        tfm_pred = point_forecasts[i]
        naive_pred = seasonal_naive(contexts[i], HORIZON)
        rows.append({"anchor_month": start, "model": "timesfm", "wmape": wmape(future, tfm_pred)})
        rows.append({"anchor_month": start, "model": "seasonal_naive_12", "wmape": wmape(future, naive_pred)})

    results = pd.DataFrame(rows)
    summary = results.groupby("model")["wmape"].agg(["mean", "std", "count"])

    print(f"\nRolling backtest: {len(anchor_starts)} anchors, "
          f"context={CONTEXT_LEN}mo, horizon={HORIZON}mo\n")
    print(results.pivot(index="anchor_month", columns="model", values="wmape").round(4))
    print("\nSummary (lower wmape is better):")
    print(summary.round(4))

    best = summary["mean"].idxmin()
    print(f"\nBest model on average across anchors: {best}")
    print(
        "\nWhy this matters: a single lucky/unlucky split can make either "
        "model look better than it is. The mean +/- std across anchors is "
        "what you should actually report and compare."
    )


if __name__ == "__main__":
    main()

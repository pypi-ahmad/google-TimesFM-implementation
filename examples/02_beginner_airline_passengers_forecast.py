"""Tier 2 - Beginner example on a real, tiny, bundled dataset.

Forecasts the classic monthly airline passengers series (1949-1960, 144
points) with zero-shot TimesFM, compares it against two naive baselines,
and plots the result. No API keys, no downloads beyond the model weights.

Dataset: examples/data/airline_passengers.csv
  Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting
  and Control. A public-domain dataset used throughout time-series
  education (e.g. R's built-in `AirPassengers`).

Companion doc: docs/06-using-timesfm.md

Run:
    uv run python examples/02_beginner_airline_passengers_forecast.py

Expected output: a metrics table printed to the console (TimesFM should
beat both naive baselines on this series) and a PNG plot saved next to
this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import timesfm

torch.set_float32_matmul_precision("high")

DATA_PATH = Path(__file__).parent / "data" / "airline_passengers.csv"
OUTPUT_PLOT = Path(__file__).parent / "02_forecast.png"

CONTEXT_LEN = 108   # first 9 years feed the model
HORIZON = 36        # predict the last 3 years (36 months)


def load_series() -> np.ndarray:
    df = pd.read_csv(DATA_PATH)
    values = df["Passengers"].to_numpy(dtype=np.float32)
    assert len(values) == 144, f"expected 144 monthly points, got {len(values)}"
    return values


def naive_last(context: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every future step."""
    return np.repeat(context[-1], horizon)


def seasonal_naive(context: np.ndarray, horizon: int, period: int = 12) -> np.ndarray:
    """Repeat the same month from one year (`period` steps) ago."""
    reps = int(np.ceil(horizon / period))
    return np.tile(context[-period:], reps)[:horizon]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main() -> None:
    series = load_series()
    context, future = series[:CONTEXT_LEN], series[CONTEXT_LEN:]
    assert len(future) == HORIZON

    print(f"Context: {len(context)} months, forecasting next {HORIZON} months.\n")

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
            infer_is_positive=True,   # passenger counts can't go negative
            fix_quantile_crossing=True,
        )
    )

    # Feed the raw values. Do NOT pre-normalize: TimesFM applies its own
    # internal instance normalization (see docs/04-core-concepts.md).
    point_forecast, quantile_forecast = model.forecast(
        horizon=HORIZON, inputs=[context]
    )
    tfm_pred = point_forecast[0]
    q10 = quantile_forecast[0, :, 1]   # column 0 = mean, columns 1-9 = p10..p90
    q90 = quantile_forecast[0, :, 9]

    baselines = {
        "naive_last": naive_last(context, HORIZON),
        "seasonal_naive_12": seasonal_naive(context, HORIZON),
        "timesfm_zero_shot": tfm_pred,
    }

    print(f"{'model':<20}{'MAE':>10}{'MAPE %':>10}")
    for name, pred in baselines.items():
        print(f"{name:<20}{mae(future, pred):>10.2f}{mape(future, pred):>10.2f}")

    # --- plot -----------------------------------------------------------
    months = np.arange(len(series))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(months[:CONTEXT_LEN], context, color="#374151", label="History (context)")
    ax.plot(months[CONTEXT_LEN:], future, color="#111827", lw=2, label="Actual")
    ax.plot(months[CONTEXT_LEN:], tfm_pred, color="#2563eb", lw=2, ls="--", label="TimesFM forecast")
    ax.fill_between(months[CONTEXT_LEN:], q10, q90, color="#2563eb", alpha=0.15, label="TimesFM p10-p90")
    ax.plot(months[CONTEXT_LEN:], baselines["seasonal_naive_12"], color="#9ca3af", lw=1.2, ls=":", label="Seasonal-naive baseline")
    ax.axvline(CONTEXT_LEN - 0.5, color="#d1d5db", lw=1, ls="--")
    ax.set_title("Airline Passengers: TimesFM zero-shot forecast vs baselines")
    ax.set_xlabel("Month index (1949-01 = 0)")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nSaved plot: {OUTPUT_PLOT}")
    print("\nSuccess criteria: timesfm_zero_shot MAE should be lower than naive_last.")


if __name__ == "__main__":
    main()

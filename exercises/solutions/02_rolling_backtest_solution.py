"""Solution: Exercise 02 (rolling backtest).

Run:
    uv run python exercises/solutions/02_rolling_backtest_solution.py
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


def main() -> None:
    torch.set_float32_matmul_precision("high")

    data_path = Path("examples/data/airline_passengers.csv")
    series = pd.read_csv(data_path)["Passengers"].to_numpy(dtype=np.float32)

    context_len = 36
    horizon = 12
    n_anchors = 6
    step = 6

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

    rows: list[dict[str, float | str | int]] = []
    for i, s in enumerate(starts):
        tfm_pred = point[i]
        naive_pred = seasonal_naive(contexts[i], horizon=horizon)
        rows.append({"anchor_start": s, "model": "timesfm", "wmape": wmape(futures[i], tfm_pred)})
        rows.append(
            {"anchor_start": s, "model": "seasonal_naive_12", "wmape": wmape(futures[i], naive_pred)}
        )

    results = pd.DataFrame(rows)
    summary = results.groupby("model")["wmape"].agg(["mean", "std", "count"])

    print(results.pivot(index="anchor_start", columns="model", values="wmape").round(4))
    print("\nSummary:")
    print(summary.round(4))


if __name__ == "__main__":
    main()


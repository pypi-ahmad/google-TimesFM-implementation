"""Solution: Exercise 01 (timestamps, gaps, first forecast).

Run:
    uv run python exercises/solutions/01_first_csv_forecast_solution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import timesfm


def main() -> None:
    torch.set_float32_matmul_precision("high")

    data_path = Path(__file__).resolve().parents[1] / "data" / "daily_sales_with_gaps.csv"
    out_plot = Path(__file__).resolve().parents[1] / "01_first_csv_forecast_solution.png"

    df = pd.read_csv(data_path)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.set_index("ds").sort_index()

    # Reindex to a complete daily calendar.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)

    # Toy fill strategy: forward-fill short gaps, then backfill the very first if needed.
    df["y"] = df["y"].astype("float32").ffill().bfill()

    values = df["y"].to_numpy(dtype=np.float32)
    context_len = 60
    horizon = 14
    context = values[:context_len]
    future = values[context_len : context_len + horizon]
    assert len(context) == context_len
    assert len(future) == horizon

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

    point, quantiles = model.forecast(horizon=horizon, inputs=[context])
    pred = point[0]
    q10 = quantiles[0, :, 1]
    q90 = quantiles[0, :, 9]

    print("point_forecast shape:", point.shape)
    print("quantile_forecast shape:", quantiles.shape)

    # Plot in time (index-based is fine for this exercise).
    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x[:context_len], context, color="#374151", label="History (context)")
    ax.plot(x[context_len : context_len + horizon], future, color="#111827", lw=2, label="Actual")
    ax.plot(
        x[context_len : context_len + horizon],
        pred,
        color="#2563eb",
        lw=2,
        ls="--",
        label="TimesFM forecast",
    )
    ax.fill_between(
        x[context_len : context_len + horizon],
        q10,
        q90,
        color="#2563eb",
        alpha=0.15,
        label="TimesFM p10-p90",
    )
    ax.axvline(context_len - 0.5, color="#d1d5db", lw=1, ls="--")
    ax.set_title("Exercise 01 Solution: Daily Sales Forecast")
    ax.set_xlabel("Day index")
    ax.set_ylabel("Sales (toy units)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    print(f"Saved plot: {out_plot}")


if __name__ == "__main__":
    main()


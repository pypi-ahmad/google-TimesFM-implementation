"""Tier 1 - Minimal working example.

The smallest possible TimesFM forecast: two synthetic series, no dataset
download, no covariates, no evaluation. If this script runs, your
environment is correctly set up.

Companion doc: docs/03-first-forecast.md (line-by-line explanation of every
call and parameter below).

Run:
    uv run python examples/01_minimal_synthetic_forecast.py

Expected output (values will differ slightly by hardware/torch version):
    point_forecast shape: (2, 12)
    quantile_forecast shape: (2, 12, 10)
    Series 0 next 3 steps: [...]
    Series 1 next 3 steps: [...]
"""

from __future__ import annotations

import numpy as np
import torch

import timesfm

# TF32 matmul is faster and numerically fine for forecasting. This mirrors
# the official TimesFM usage example.
torch.set_float32_matmul_precision("high")


def main() -> None:
    print("Loading TimesFM 2.5 (200M, PyTorch) from Hugging Face...")
    print("First run downloads ~800MB of weights and caches them locally.")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    # compile() fixes the max context/horizon the model will accept and
    # turns on the forecasting behavior we want. See docs/03-first-forecast.md
    # for what each flag does.
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    print("Model compiled.\n")

    # Two dummy series of *different* lengths, on purpose: TimesFM accepts
    # a list of 1-D arrays and each can have its own length.
    series_a = np.linspace(0, 1, 100)          # a slow ramp
    series_b = np.sin(np.linspace(0, 20, 67))  # a noiseless sine wave

    point_forecast, quantile_forecast = model.forecast(
        horizon=12,
        inputs=[series_a, series_b],
    )

    print("point_forecast shape:", point_forecast.shape)       # (2, 12)
    print("quantile_forecast shape:", quantile_forecast.shape)  # (2, 12, 10)
    print()
    print("Series 0 (ramp) next 3 steps:   ", np.round(point_forecast[0, :3], 3))
    print("Series 1 (sine wave) next 3 steps:", np.round(point_forecast[1, :3], 3))
    print()
    print("Success: your environment can load and run TimesFM.")
    print("Next: docs/03-first-forecast.md, then examples/02_beginner_airline_passengers_forecast.py")


if __name__ == "__main__":
    main()

"""Tier 3 - Forecasting with covariates (XReg).

Shows *when and why* you'd hand TimesFM extra signals (price, promotions,
day-of-week) instead of just the raw history, using small synthetic retail
data so the effect of each covariate is easy to see.

This mirrors the official covariates example distributed with TimesFM:
https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/covariates-forecasting

Requires the `xreg` extra (already in this project's pyproject.toml):
    uv sync --locked --group xreg   # (installs timesfm[xreg])

Companion doc: docs/04-core-concepts.md ("Covariates / XReg") and
docs/05-data-format-and-preprocessing.md ("Covariate format").

Run:
    uv run python examples/03_covariates_xreg_example.py
"""

from __future__ import annotations

import numpy as np
import torch

import timesfm

torch.set_float32_matmul_precision("high")

CONTEXT_LEN = 24   # weeks of history
HORIZON = 12       # weeks to forecast


def make_synthetic_store(seed: int, base_sales: float, price_sensitivity: float) -> dict:
    """One store's weekly sales plus the covariates that drive them.

    `sales = trend + seasonality + price_effect + promo_effect + noise`, so
    we know the ground-truth relationship a real dataset would hide.
    """
    rng = np.random.default_rng(seed)
    weeks = np.arange(CONTEXT_LEN + HORIZON)

    trend = base_sales * (1 + 0.004 * weeks)
    seasonality = 40 * np.sin(2 * np.pi * weeks / 52)
    noise = rng.normal(0, 15, len(weeks))

    price = 10.0 + rng.uniform(-0.4, 0.4, len(weeks))
    price_effect = -price_sensitivity * (price - 10.0)

    promotion = rng.choice([0.0, 1.0], len(weeks), p=[0.8, 0.2])
    promo_effect = 60 * promotion

    sales = np.maximum(trend + seasonality + price_effect + promo_effect + noise, 20.0)

    return {
        "sales": sales.astype(np.float32),
        "price": price.astype(np.float32),
        "promotion": promotion.astype(np.float32),
    }


def main() -> None:
    stores = {
        "store_A": make_synthetic_store(seed=1, base_sales=500, price_sensitivity=25),
        "store_B": make_synthetic_store(seed=2, base_sales=300, price_sensitivity=10),
    }

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
            return_backcast=True,  # required by forecast_with_covariates()
        )
    )

    inputs = [data["sales"][:CONTEXT_LEN] for data in stores.values()]

    # Baseline: TimesFM with no covariates at all.
    baseline_point, _ = model.forecast(horizon=HORIZON, inputs=inputs)

    # With covariates: price and promotion are known/planned for the
    # forecast horizon too (that's the point of using them), so each
    # covariate array spans CONTEXT_LEN + HORIZON, same as `inputs`
    # spans only the context portion.
    dynamic_numerical_covariates = {
        "price": [data["price"] for data in stores.values()],
        "promotion": [data["promotion"] for data in stores.values()],
    }

    # Default xreg_mode: TimesFM forecasts first, then a regression explains
    # the *residual* using the covariates. This is the mode this repo's own
    # airline-demand notebook uses in production. See docs/04-core-concepts.md
    # for what the alternative mode ("timesfm + xreg") does differently --
    # at the time of writing it errors on single-series inputs with the
    # installed timesfm==2.0.2, so this example sticks to the verified path.
    xreg_point, _ = model.forecast_with_covariates(
        inputs=inputs,
        dynamic_numerical_covariates=dynamic_numerical_covariates,
        xreg_mode="xreg + timesfm",
        ridge=1e-3,
    )

    print(f"\n{'store':<10}{'model':<24}{'MAE vs actual future':>22}")
    for i, (name, data) in enumerate(stores.items()):
        future = data["sales"][CONTEXT_LEN:]
        mae_baseline = float(np.mean(np.abs(future - baseline_point[i])))
        mae_xreg = float(np.mean(np.abs(future - xreg_point[i])))
        print(f"{name:<10}{'timesfm (no covariates)':<24}{mae_baseline:>22.2f}")
        print(f"{name:<10}{'timesfm + xreg':<24}{mae_xreg:>22.2f}")

    print(
        "\nInterpretation: price and promotion swings are, by construction, "
        "part of the ground truth here but invisible in the raw sales history "
        "alone -- yet covariates do NOT automatically win (run this script "
        "and compare the two rows per store above). Whether XReg helps "
        "depends on how much *residual* variance your covariates explain "
        "versus how much signal TimesFM already captures zero-shot. Treat "
        "this like any other modeling choice: back-test it on your own data "
        "(see docs/07-evaluation.md) before trusting it, rather than "
        "assuming covariates are a free win."
    )


if __name__ == "__main__":
    main()

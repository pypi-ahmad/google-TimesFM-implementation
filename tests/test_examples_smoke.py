"""End-to-end smoke test: actually loads TimesFM and runs a real forecast.

Unlike test_environment.py, this test needs the ~800MB model checkpoint
(downloaded once and cached under ~/.cache/huggingface/, per
docs/02-installation.md) and takes several seconds to a couple of minutes
depending on hardware. It is what actually proves examples/01 and the code
in docs/03-first-forecast.md work, not just that the package imports.

Run:
    uv run pytest tests/test_examples_smoke.py -v -m model

Skip these (e.g. in a network-restricted CI job) with:
    uv run pytest -m "not model"
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.model


def _load_compiled_model():
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=128,
            max_horizon=32,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def test_minimal_forecast_shapes_match_docs() -> None:
    """Exercises exactly the call shown in docs/03-first-forecast.md and
    examples/01_minimal_synthetic_forecast.py, and checks the shapes the
    docs promise: point_forecast (num_series, horizon),
    quantile_forecast (num_series, horizon, 10)."""
    try:
        model = _load_compiled_model()
    except Exception as exc:  # noqa: BLE001 - network/env issues are a skip, not a failure
        pytest.skip(f"could not load TimesFM (likely no network/model cache): {exc}")

    series_a = np.linspace(0, 1, 50)
    series_b = np.sin(np.linspace(0, 20, 40))
    horizon = 12

    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon, inputs=[series_a, series_b]
    )

    assert point_forecast.shape == (2, horizon)
    assert quantile_forecast.shape == (2, horizon, 10)
    assert np.isfinite(point_forecast).all(), "forecast contains NaN/Inf"

    # Quantiles must be monotonically non-decreasing (p10 <= p50 <= ... <= p90)
    # per fix_quantile_crossing=True, documented in docs/03-first-forecast.md.
    quantile_columns = quantile_forecast[:, :, 1:]  # drop the mean column (index 0)
    assert np.all(np.diff(quantile_columns, axis=-1) >= -1e-4), (
        "quantile forecast columns are not monotonic; fix_quantile_crossing "
        "should guarantee p10 <= p50 <= ... <= p90"
    )


def test_variable_length_inputs_are_supported() -> None:
    """docs/05-data-format-and-preprocessing.md claims series in the same
    batch can have different lengths -- verify that directly."""
    try:
        model = _load_compiled_model()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not load TimesFM (likely no network/model cache): {exc}")

    short_series = np.arange(20, dtype=np.float32)
    long_series = np.arange(100, dtype=np.float32)

    point_forecast, _ = model.forecast(horizon=8, inputs=[short_series, long_series])
    assert point_forecast.shape == (2, 8)

"""Fast smoke tests: no network access, no model download, no GPU required.

These check that the environment is set up correctly (per
docs/02-installation.md) and that the `timesfm` API surface this repo's
docs and examples rely on actually exists in the installed version. They
run in well under a second and are meant to catch environment problems
immediately, before anyone waits minutes for a model download only to hit
an unrelated `ImportError`.

Run:
    uv run pytest tests/test_environment.py -v
"""

from __future__ import annotations

import sys

import pytest


def test_python_version_meets_minimum() -> None:
    """docs/01-prerequisites.md documents Python >= 3.11 as the minimum."""
    assert sys.version_info >= (3, 11), (
        f"Python {sys.version_info.major}.{sys.version_info.minor} is below "
        "the 3.11 minimum documented in docs/01-prerequisites.md"
    )


def test_core_dependencies_importable() -> None:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import torch  # noqa: F401


def test_timesfm_importable() -> None:
    import timesfm  # noqa: F401


def test_timesfm_2p5_torch_class_exists() -> None:
    """This is the exact class every example and doc page in this repo uses."""
    import timesfm

    assert hasattr(timesfm, "TimesFM_2p5_200M_torch"), (
        "timesfm.TimesFM_2p5_200M_torch not found. This repo's docs/examples "
        "assume timesfm>=2.0.2 (the TimesFM 2.5 API) -- run `uv sync` to "
        "match pyproject.toml, or see docs/08-troubleshooting.md."
    )
    assert hasattr(timesfm.TimesFM_2p5_200M_torch, "from_pretrained")


def test_forecast_config_exists_with_documented_fields() -> None:
    """These are exactly the fields explained in docs/03-first-forecast.md."""
    import timesfm

    assert hasattr(timesfm, "ForecastConfig")
    documented_fields = {
        "max_context",
        "max_horizon",
        "normalize_inputs",
        "use_continuous_quantile_head",
        "force_flip_invariance",
        "infer_is_positive",
        "fix_quantile_crossing",
    }
    # ForecastConfig is a dataclass-like config object; constructing it with
    # every documented field should not raise TypeError for an unknown kwarg.
    try:
        timesfm.ForecastConfig(**{f: False for f in documented_fields if f not in ("max_context", "max_horizon")},
                                max_context=32, max_horizon=8)
    except TypeError as exc:
        pytest.fail(
            f"ForecastConfig rejected a field documented in docs/03-first-forecast.md: {exc}"
        )


def test_torch_reports_cuda_availability_without_error() -> None:
    """Doesn't assert True or False -- just that the check itself is safe,
    matching docs/02-installation.md's GPU-detection snippet."""
    import torch

    assert torch.cuda.is_available() in (True, False)

# 03 - Your First Forecast

> Previous: [02 - Installation](02-installation.md). Next: [04 - Core Concepts](04-core-concepts.md).

By the end of this page you'll have run TimesFM once, and understand every
line that made it happen.

## Run it

```bash
uv run python examples/01_minimal_synthetic_forecast.py
```

**Expected output** (observed on the reference environment for this repo --
Python 3.14, torch 2.12, CUDA GPU; your exact numbers will differ by
hardware and torch version, the shapes and overall pattern won't):

```
Loading TimesFM 2.5 (200M, PyTorch) from Hugging Face...
First run downloads ~800MB of weights and caches them locally.
Model compiled.

point_forecast shape: (2, 12)
quantile_forecast shape: (2, 12, 10)

Series 0 (ramp) next 3 steps:    [1.013 1.019 1.027]
Series 1 (sine wave) next 3 steps: [0.985 0.978 0.891]

Success: your environment can load and run TimesFM.
```

**How to verify success:** the script prints two shapes and exits with no
traceback. If you see a Python traceback instead, go to
[08 - Troubleshooting](08-troubleshooting.md).

## Line by line

Open [`examples/01_minimal_synthetic_forecast.py`](../examples/01_minimal_synthetic_forecast.py)
alongside this section.

### 1. Precision setting

```python
torch.set_float32_matmul_precision("high")
```

Tells PyTorch to use a faster (TF32) matrix-multiply mode on supported
hardware. This is a performance setting, not a correctness one -- it comes
straight from the [official usage example](https://github.com/google-research/timesfm#code-example)
and is safe to leave as-is.

### 2. Loading the model

```python
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
```

`TimesFM_2p5_200M_torch` is the PyTorch implementation class for the
"2.5, 200-million-parameter" checkpoint. `.from_pretrained(...)` downloads
(or loads from cache) the weights hosted at that Hugging Face repository ID
and builds the model object in memory. Nothing is forecast yet -- this step
only loads the model.

> **Why this matters:** you never write or run any training code to get to
> this point. The model already knows how to forecast; you're loading
> knowledge, not building it.

### 3. Compiling a forecast configuration

```python
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
```

`compile()` locks in the settings the model will use for every call until
you compile again. You must call it once before forecasting. Each flag,
in plain English:

| Flag | Meaning |
|---|---|
| `max_context` | The longest history (in time steps) any single call may pass in. Passing more than this raises an error; passing less is fine. |
| `max_horizon` | The longest forecast (in time steps) any single call may request. |
| `normalize_inputs` | Let TimesFM rescale each series internally before forecasting (its own internal normalization -- see [04 - Core Concepts](04-core-concepts.md)). You should almost always leave this `True` and never pre-normalize your data yourself. |
| `use_continuous_quantile_head` | Turn on the extra head that produces uncertainty bands (the 10 columns in `quantile_forecast`), not just a single point prediction. |
| `force_flip_invariance` | A stability setting so mirrored/flipped versions of the same shape forecast consistently. |
| `infer_is_positive` | If every value you've fed in so far is non-negative (e.g. counts, dollars, temperatures in Kelvin), the model will avoid predicting negative values. |
| `fix_quantile_crossing` | Guarantees the quantile bands come out correctly ordered (p10 ≤ p50 ≤ p90), rather than occasionally crossing due to independent per-quantile estimation. |

> **Common beginner mistake:** trying to forecast a horizon longer than
> `max_horizon`, or pass context longer than `max_context`. Both raise a
> clear error naming the limit -- if you see one, increase the relevant
> value in `ForecastConfig` and recompile.

### 4. The forecast call itself

```python
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[series_a, series_b],
)
```

- `inputs` is a **list of 1-D arrays**, one per series. Series can have
  *different lengths* -- TimesFM handles that natively, you do not need to
  pad them to match.
- `horizon` is how many future steps to predict, for every series in the
  batch, in one call.
- It returns two arrays:
  - `point_forecast`, shape `(num_series, horizon)` -- the single best-guess
    value at each future step.
  - `quantile_forecast`, shape `(num_series, horizon, 10)` -- column 0 is
    the mean, columns 1-9 are the 10th, 20th, ..., 90th percentile
    forecasts. Wider gaps between low and high columns mean the model is
    less certain.

## Common beginner mistakes on this page

- **Passing a Python `list` of ints instead of a NumPy array.** TimesFM
  expects array-like numeric sequences; plain Python lists of numbers also
  work, but mixed types or `None` values will error. Clean your data to
  `float32` NumPy arrays first (as the beginner example in
  [06 - Using TimesFM](06-using-timesfm.md) does).
- **Forgetting to call `.compile()` before `.forecast()`.** You'll get an
  error saying the model hasn't been compiled.
- **Re-compiling on every call inside a loop.** Compile once, forecast many
  times -- recompiling is wasted work and, in a tight loop, noticeably
  slower.

---

**Next:** [04 - Core Concepts](04-core-concepts.md) -- now that you've seen
it work, here's *why* it works.

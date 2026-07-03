# 02 - Installation

> Previous: [01 - Prerequisites](01-prerequisites.md). Next: [03 - First Forecast](03-first-forecast.md).

Two supported paths: this repo's exact environment (recommended, guarantees
every example and doc command works as written), or a minimal standalone
install if you only want the `timesfm` package itself.

## Option A: this repo's environment (recommended)

```bash
git clone https://github.com/pypi-ahmad/google-TimesFM-implementation.git
cd google-TimesFM-implementation

uv venv
source .venv/bin/activate
uv sync
```

What each command does:

- `uv venv` -- creates an isolated Python environment in `.venv/`, so
  nothing here touches your system Python.
- `source .venv/bin/activate` -- makes that environment the active one for
  your current shell. (On Windows: `.venv\Scripts\activate`.)
- `uv sync` -- reads `pyproject.toml` and `uv.lock` and installs the exact
  dependency versions this repo was built and tested against, including
  `timesfm[torch,xreg]`, PyTorch, pandas, and the plotting/notebook stack.

### Verify the install

```bash
uv run python -c "import timesfm; import torch; print('timesfm OK, torch', torch.__version__)"
```

**Expected output:** something like `timesfm OK, torch 2.x.x`. No traceback.

If this fails, jump to [08 - Troubleshooting](08-troubleshooting.md#installation-errors)
before continuing.

## Option B: minimal standalone install (just the model)

If you only want to experiment with TimesFM outside this repo:

```bash
pip install timesfm[torch]
# Add XReg support (covariates, used in examples/03 and docs/04):
pip install timesfm[xreg]
```

This is exactly what the [official TimesFM repository](https://github.com/google-research/timesfm#install)
documents. Everything after this page assumes Option A (this repo's
environment) so that file paths and bundled datasets resolve correctly, but
the `timesfm` API itself is identical either way.

## GPU vs. CPU

You don't configure this explicitly -- PyTorch detects hardware
automatically:

```bash
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

- `True` -- TimesFM will use your GPU automatically. No code changes
  needed.
- `False` -- TimesFM runs on CPU. Every example in this repo is sized to
  finish in well under a minute on CPU.

To **force** CPU even if a GPU is present (useful for reproducing this
repo's documented outputs exactly, or debugging GPU-specific errors):

```bash
CUDA_VISIBLE_DEVICES="" uv run python examples/01_minimal_synthetic_forecast.py
```

## Model weight download and Hugging Face access

The first time you run any example, `timesfm` downloads the model
checkpoint (`google/timesfm-2.5-200m-pytorch`, ~800 MB) from Hugging Face
Hub and caches it under `~/.cache/huggingface/`. This checkpoint is
**public** -- no Hugging Face account or access token is required.

Subsequent runs load from the local cache and need no network access for
the model itself.

> **Common mistake:** running the first example in an environment with no
> internet access (a locked-down CI runner, an air-gapped machine) and
> assuming the code is broken. It isn't -- it's waiting on a download. See
> [08 - Troubleshooting](08-troubleshooting.md#network--download-errors) for
> how to pre-download or vendor the checkpoint.

## Optional: Jupyter kernel for the notebooks

Only needed if you plan to run the case-study notebooks in
[`notebooks/`](../notebooks/):

```bash
uv run python -m ipykernel install --user --name timesfm-local --display-name "Python (timesfm-local)"
```

## Exact version assumptions used throughout this repo

So examples stay reproducible, every doc and script in this repo assumes:

- `timesfm[torch,xreg] >= 2.0.2` (the PyPI package version; this ships the
  "TimesFM 2.5" model API described throughout these docs -- see
  [04 - Core Concepts](04-core-concepts.md) for why the PyPI version number
  and the model version name differ)
- `torch >= 2.0`
- Python `>= 3.11`

Run `uv sync` again after pulling repo updates to stay aligned with these
pins.

---

**Next:** [03 - First Forecast](03-first-forecast.md) -- run your first
forecast and understand every line.

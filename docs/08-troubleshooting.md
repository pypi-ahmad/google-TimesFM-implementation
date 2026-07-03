# 08 - Troubleshooting

> Previous: [07 - Evaluation](07-evaluation.md). Next: [09 - FAQ](09-faq.md).

Organized by when the problem shows up. Use Ctrl+F for your error message.

## Installation errors

**`uv: command not found`**
`uv` isn't installed. Install it per the
[official uv docs](https://docs.astral.sh/uv/getting-started/installation/),
or substitute plain `pip` throughout: `python3 -m venv .venv && source
.venv/bin/activate && pip install -e .`

**`ModuleNotFoundError: No module named 'timesfm'`**
You're likely running Python outside the project's virtual environment.
Fix: prefix commands with `uv run`, or explicitly activate the environment
first (`source .venv/bin/activate`) before running plain `python`.

**Dependency resolution fails / version conflicts during `uv sync`**
Delete `.venv/` and retry (`rm -rf .venv && uv venv && uv sync`) to rule out
a stale environment. If it persists, confirm your Python version matches
[01 - Prerequisites](01-prerequisites.md) (`python3 --version` >= 3.11).

**Installing the `xreg` extra pulls in JAX/CUDA packages you don't want**
The official `timesfm[xreg]` extra depends on `jax[cuda]` even if you only
use the PyTorch backend (this is an upstream packaging choice, not a bug in
this repo). On a CPU-only machine or Apple Silicon, this can be slow to
install or pull the wrong accelerator build. If you don't need covariates
(XReg) at all, skip that extra and use `timesfm[torch]` alone -- everything
except [`examples/03_covariates_xreg_example.py`](../examples/03_covariates_xreg_example.py)
will work without it.

## Network / download errors

**Hangs or fails on the first run of any example, with a connection error**
The first call to `.from_pretrained(...)` downloads the ~800MB model
checkpoint from Hugging Face Hub. If you're on a restricted network:

1. Confirm you can reach `huggingface.co` from this machine at all.
2. If you're behind a corporate proxy, set the standard `HTTPS_PROXY`
   environment variable before running.
3. To pre-download once (e.g. on a machine with internet) and reuse the
   cache elsewhere, copy the `~/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/`
   directory to the target machine's equivalent path.

**`401 Unauthorized` from Hugging Face**
This shouldn't happen for `google/timesfm-2.5-200m-pytorch` -- it's a
public checkpoint requiring no token. If you see this, check for a stale
or invalid `HF_TOKEN` environment variable forcing authentication it
doesn't need; unset it and retry.

## Runtime / inference errors

**`RuntimeError` mentioning context or horizon length**
You passed `inputs` longer than `max_context`, or requested `horizon`
longer than `max_horizon`, at compile time. Fix: increase the relevant
value in `ForecastConfig` and call `.compile()` again before forecasting.
See [03 - First Forecast](03-first-forecast.md#3-compiling-a-forecast-configuration).

**`CUDA out of memory`**
Your GPU doesn't have enough free memory for the batch size / context
length you requested. Options, in order of ease:
1. Force CPU for this run: `CUDA_VISIBLE_DEVICES="" uv run python <script>.py`
   (see [02 - Installation](02-installation.md#gpu-vs-cpu)).
2. Reduce how many series you forecast in one `inputs=[...]` batch.
3. Reduce `max_context` if you don't need very long history.

**Forecast values look wrong (e.g. negative counts, wildly unstable)**
- Confirm you didn't pre-normalize your data (see
  [04 - Core Concepts, section 4](04-core-concepts.md#4-internal-normalization-dont-normalize-your-data-yourself)).
- If negative values are impossible for your domain (counts, prices,
  physical quantities), set `infer_is_positive=True` in `ForecastConfig`.
- Check for silent data-quality issues upstream -- duplicated rows, wrong
  units, unhandled gaps (see
  [05 - Data Format and Preprocessing](05-data-format-and-preprocessing.md)).

**`ValueError: For XReg, return_backcast must be set to True in the forecast config`**
You called `model.forecast_with_covariates(...)` on a model compiled
without `return_backcast=True`. Add that flag to `ForecastConfig` and
recompile -- see [`examples/03_covariates_xreg_example.py`](../examples/03_covariates_xreg_example.py)
for the working configuration.

**Shape mismatch / `IndexError` when reading `quantile_forecast`**
Remember the shape is `(num_series, horizon, 10)`, where column `0` is the
mean and columns `1`-`9` are p10 through p90 in 10-point steps (not columns
`0`-`8`). See [03 - First Forecast](03-first-forecast.md#4-the-forecast-call-itself).

## Notebook-specific issues (the case studies in `notebooks/`)

The ten applied case-study notebooks pull real datasets from Kaggle and
other public sources and have their own setup needs. See
[`HANDBOOK.md`, section 13](../HANDBOOK.md) for the full list, including
Kaggle authentication (`403`/`401` errors), execution timeouts on
constrained hardware, and reuse-mode artifact paths. That material is
specific to the advanced/production tier and intentionally kept separate
from this beginner troubleshooting page.

## Still stuck?

1. Re-read the companion doc page for the script you're running -- most
   scripts link back to a specific chapter above their relevant code.
2. Check [09 - FAQ](09-faq.md) for conceptual (not error-message) confusion.
3. Compare your code against the
   [official TimesFM README](https://github.com/google-research/timesfm) --
   if the official example doesn't work either, it's an upstream issue, not
   this repo.
4. Search (or open) an issue on the
   [official TimesFM GitHub repository](https://github.com/google-research/timesfm/issues)
   for bugs in the model/package itself.

---

**Next:** [09 - FAQ](09-faq.md).

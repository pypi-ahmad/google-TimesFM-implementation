# 01 - Prerequisites

> Previous: [00 - Overview](00-overview.md). Next: [02 - Installation](02-installation.md).

This page tells you exactly what you need to know and have installed
*before* touching code. If you're missing something, it links to where to
learn it -- this repo won't re-teach Python or pandas from scratch.

## Knowledge prerequisites

### Required

- **Basic Python.** You should be comfortable reading a `for` loop, a
  function definition, and a dictionary. You do not need to be an expert.
- **A little pandas/NumPy.** You'll see `DataFrame`, `.to_numpy()`, and
  array shapes like `(2, 12)`. If `array.shape` is unfamiliar, spend 15
  minutes with the [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
  first.
- **What a time series is.** A sequence of numbers, each tied to a point in
  time, usually at a regular interval (hourly, daily, monthly). Example:
  daily website visits for the last 90 days.

### Helpful, not required

- Any prior exposure to forecasting concepts (trend, seasonality) will make
  [04 - Core Concepts](04-core-concepts.md) click faster, but that page
  defines every term in plain English as it goes.
- Prior deep learning experience is **not required**. You are running a
  pretrained model, not building a neural network from scratch.

### Concepts this repo defines for you as you go

You don't need to know these yet -- each is introduced with a plain-English
explanation the first time it's used:

`context window`, `forecast horizon`, `backtesting`, `quantile forecast`,
`covariate`, `zero-shot`, `fine-tuning`.

## Software prerequisites

| Requirement | Version | Why |
|---|---|---|
| Python | 3.11+ | This project's minimum. (Upstream TimesFM itself only requires 3.10+; this repo pins slightly higher for other dependencies.) |
| [`uv`](https://docs.astral.sh/uv/) | any recent | Fast, reproducible Python environment manager. `pip` works too -- see [02 - Installation](02-installation.md) for the equivalent commands. |
| Git | any | To clone this repository. |
| Disk space | ~2 GB free | The TimesFM 2.5 checkpoint is roughly 800 MB; PyTorch and its CUDA libraries (if applicable) add the rest. |

## Hardware

- **CPU-only works.** Every example in this repo runs on CPU. It's slower
  (seconds instead of milliseconds per forecast call) but produces
  identical results.
- **GPU is optional, not required.** This repo's default environment
  intentionally installs **CPU-only PyTorch** for maximum portability and
  fewer install failures. If you want GPU acceleration, follow the
  instructions in [02 - Installation](02-installation.md#gpu-vs-cpu) to
  switch the PyTorch index and re-lock.
- **No internet after first run** for the core inference workflow, other
  than the one-time model download (cached locally by Hugging Face's
  `huggingface_hub` library, typically under `~/.cache/huggingface/`).

## Self-check before moving on

Run this in a terminal. If both succeed, you're ready for
[02 - Installation](02-installation.md):

```bash
python3 --version   # should print 3.11 or higher
git --version        # any version
```

If `python3 --version` shows something older than 3.11, install a newer
Python first (e.g. via [pyenv](https://github.com/pyenv/pyenv) or your OS
package manager) -- `uv` can also manage this for you, shown next.

---

**Next:** [02 - Installation](02-installation.md).

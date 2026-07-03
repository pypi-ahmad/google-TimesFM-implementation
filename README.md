# Google TimesFM: Zero to Mastery

A hands-on learning path for **[Google TimesFM](https://github.com/google-research/timesfm)**,
Google Research's pretrained, zero-shot time-series forecasting model --
plus ten real-world applied case studies once you've mastered the basics.

This repository teaches TimesFM the way a good handbook should: intuition
first, then code, then real data, then honest evaluation. It's built for
**complete beginners** to time-series forecasting as much as for engineers
who already know the field and just want a fast, accurate path to using
this specific model well.

## Who this is for

- You've never used a time-series forecasting model and want to learn by
  doing, not by reading a paper cold.
- You know some Python and pandas but nothing about forecasting,
  transformers, or foundation models.
- You're an ML/data engineer evaluating TimesFM for a real project and want
  a fast, honest, hands-on ramp-up -- including where it *doesn't* win.

## What you'll be able to do when you're done

By the end of the [learning path](#learning-path) below, you will be able
to: explain what a time-series foundation model is and why zero-shot
forecasting matters; install and run TimesFM correctly; prepare your own
data for it without common preprocessing mistakes; use covariates (XReg)
when you have extra known signals; and evaluate a forecast honestly with
backtesting and baselines, instead of trusting one lucky chart.

## Why TimesFM matters

Classical forecasting requires a model trained (and retrained) on *your*
specific series. TimesFM is pretrained once by Google Research on a broad
corpus of time series and then used directly, with no training step, on
series it has never seen -- the same "pretrain once, use everywhere"
pattern that made large language models useful zero-shot. It's not a
research curiosity either: the same model powers
[BigQuery ML](https://cloud.google.com/bigquery/docs/timesfm-model),
[Google Sheets forecasting](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html),
and Vertex AI Model Garden (TimesFM is listed in the official repo; the Model Garden console itself requires a Google
Cloud login): [Model Garden overview](https://cloud.google.com/model-garden).
Full explanation: [`docs/00-overview.md`](docs/00-overview.md).

> **Scope and honesty note:** per the
> [official repository](https://github.com/google-research/timesfm), "this
> open version is not an officially supported Google product." This
> repository reports real, reproducible results throughout -- including
> the case study where a naive baseline currently beats TimesFM (see
> [`docs/09-faq.md`](docs/09-faq.md)) -- rather than only showing wins.

## Quickstart

```bash
git clone https://github.com/pypi-ahmad/google-TimesFM-implementation.git
cd google-TimesFM-implementation

uv venv && source .venv/bin/activate
uv sync --locked

uv run python examples/01_minimal_synthetic_forecast.py
```

Expected: the script loads the model (downloading ~800MB the first time),
prints two forecast array shapes, and exits cleanly with no traceback. If
anything goes wrong, see [`docs/08-troubleshooting.md`](docs/08-troubleshooting.md).
Full setup details, GPU/CPU notes, and version requirements:
[`docs/02-installation.md`](docs/02-installation.md).

Optional installs:
- Covariates/XReg: `uv sync --locked --group xreg`
- Applied notebooks/handbook tooling: `uv sync --locked --group applied` (and `--group xreg` for the retail case study)

## Your first real forecast, in ~20 lines

This is the complete, minimal pattern every example and doc page in this
repo builds on (line-by-line explanation:
[`docs/03-first-forecast.md`](docs/03-first-forecast.md)):

```python
import numpy as np
import torch
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(
    max_context=1024,
    max_horizon=256,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=True,
    fix_quantile_crossing=True,
))

point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[np.linspace(0, 1, 100), np.sin(np.linspace(0, 20, 67))],
)
# point_forecast.shape    -> (2, 12)
# quantile_forecast.shape -> (2, 12, 10)  (mean, then p10..p90)
```

## Learning path

Follow in order. Each page defines its own terms in plain English and
links to the next.

| # | Doc | You'll learn |
|---|---|---|
| 00 | [Overview](docs/00-overview.md) | What TimesFM is, the problem it solves, where it's used in production |
| 01 | [Prerequisites](docs/01-prerequisites.md) | What knowledge/software you need before starting |
| 02 | [Installation](docs/02-installation.md) | Environment setup, GPU/CPU, version pins |
| 03 | [First Forecast](docs/03-first-forecast.md) | The smallest working example, explained line by line |
| 04 | [Core Concepts](docs/04-core-concepts.md) | Patched-decoder architecture, context/horizon, quantiles, covariates, version differences |
| 05 | [Data Format & Preprocessing](docs/05-data-format-and-preprocessing.md) | Input shapes, missing data, leakage traps |
| 06 | [Using TimesFM on Real Data](docs/06-using-timesfm.md) | A full workflow on a real dataset |
| 07 | [Evaluation](docs/07-evaluation.md) | Backtesting, metrics, baselines, calibration |
| 08 | [Troubleshooting](docs/08-troubleshooting.md) | Fixes for common errors, organized by symptom |
| 09 | [FAQ](docs/09-faq.md) | Conceptual questions answered directly |
| 10 | [Next Steps](docs/10-next-steps.md) | Fine-tuning, the applied case studies, the wider field |

Then: [`examples/`](examples/) has four runnable, standalone scripts (one
per concept above), and [`notebooks/`](notebooks/) has ten full real-world
case studies for after you've finished the path -- see
[`notebooks/README.md`](notebooks/README.md).

Optional but recommended: [`exercises/`](exercises/) turns the docs into
hands-on practice (with runnable solutions) so the workflow becomes
muscle memory.

## Repository structure

```text
.
├── README.md                 <- you are here
├── docs/                     <- the 00-10 learning path (start at 00-overview.md)
│   └── evidence/              <- machine-generated evidence backing the case-study results
├── examples/                 <- 4 tiered, standalone, runnable scripts + bundled tiny dataset
├── exercises/                <- optional practice track + runnable solutions
├── notebooks/                <- 10 applied case studies (advanced tier; see notebooks/README.md)
├── scripts/                  <- production tooling behind the applied tier (run/validate/build-evidence)
├── tests/                    <- environment + end-to-end smoke tests (pytest)
├── HANDBOOK.md / .pdf        <- full operational manual for the applied tier
├── RELEASE_NOTES.md          <- version history and what changed
└── pyproject.toml / uv.lock  <- exact, reproducible dependency pins
```

## Prerequisites, in brief

Python 3.11+, [`uv`](https://docs.astral.sh/uv/), ~2GB free disk for the
model checkpoint, and basic Python/pandas familiarity. GPU is optional --
every example here runs on CPU. Full details:
[`docs/01-prerequisites.md`](docs/01-prerequisites.md).

## Troubleshooting

Install errors, download/network issues, CUDA errors, and shape/config
mistakes are all covered, organized by symptom, in
[`docs/08-troubleshooting.md`](docs/08-troubleshooting.md).

## Scope and limitations

- This repo teaches the **PyTorch inference API** of TimesFM 2.5 in depth.
  Fine-tuning uses a different (Transformers + PEFT) stack that this repo
  points to rather than reimplements -- see
  [`docs/10-next-steps.md`](docs/10-next-steps.md).
- The applied case studies in `notebooks/` depend on external data sources
  (Kaggle, HealthData.gov, NAB) that require their own credentials/network
  access; see [`HANDBOOK.md`](HANDBOOK.md).
- Results reported anywhere in this repo (including the applied tier) come
  from real, executed runs -- not simulated or hand-picked numbers. Where
  TimesFM loses to a simple baseline, that's stated plainly rather than
  omitted.
- This is a community learning resource, independent of Google, and is not
  officially affiliated with or supported by Google Research.

## References

**Core model, official sources (primary references for every technical
claim in this repo):**

- [TimesFM official repository](https://github.com/google-research/timesfm) -- source code, install instructions, update history. The authoritative source; where this repo and upstream disagree, upstream wins.
- Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). [*A decoder-only foundation model for time-series forecasting.*](https://arxiv.org/abs/2310.10688) ICML 2024. The original architecture and methodology paper.
- [Google Research blog: "A decoder-only foundation model for time-series forecasting"](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/) -- accessible summary of the paper above.
- [TimesFM Hugging Face collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6) -- all released checkpoints, including [`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch) used throughout this repo.
- [Official fine-tuning guide (LoRA + HF Transformers)](https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/examples/finetuning/README.md) -- source for everything this repo says about fine-tuning in [`docs/10-next-steps.md`](docs/10-next-steps.md).
- [Official covariates (XReg) example](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/covariates-forecasting) -- source for [`docs/04-core-concepts.md`](docs/04-core-concepts.md)'s covariates section and [`examples/03_covariates_xreg_example.py`](examples/03_covariates_xreg_example.py).

**TimesFM in production Google products** (context for [`docs/00-overview.md`](docs/00-overview.md)):

- [BigQuery ML: TimesFM model](https://cloud.google.com/bigquery/docs/timesfm-model)
- [Connected Sheets forecasting with BigQuery ML/TimesFM](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html)
- [Vertex AI Model Garden](https://cloud.google.com/model-garden) (TimesFM is referenced in the official TimesFM repo; the console listing is not publicly crawlable without login.)

**Frameworks and libraries used in the applied tier** (see [`HANDBOOK.md`](HANDBOOK.md)):

- [Kaggle API docs](https://www.kaggle.com/docs/api), [OR-Tools](https://developers.google.com/optimization), [Polars](https://pola.rs/), [PyArrow](https://arrow.apache.org/docs/python/), [Pydantic](https://docs.pydantic.dev/)

**Dataset sources for the applied case studies**, each cited at point of use in the relevant notebook and in `HANDBOOK.md`:
[Favorita grocery sales](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting) ·
[ATM cash demand](https://www.kaggle.com/datasets/zoya77/atm-cash-demand-forecasting-and-management) ·
[Airline bookings](https://www.kaggle.com/datasets/saadharoon27/airlines-dataset) ·
[Hourly energy consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) ·
[Manufacturing sensors](https://www.kaggle.com/datasets/anshtanwar/metro-train-dataset) ·
[Financial transactions](https://www.kaggle.com/datasets/sergionefedov/fraud-detection-1m-transactions-7-fraud-types) ·
[HHS hospital capacity](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh) ·
[NAB (Numenta Anomaly Benchmark)](https://github.com/numenta/NAB)

**Beginner-tier bundled dataset:** the classic monthly airline passengers
series (1949-1960) used in `examples/02` and `examples/04`, from Box,
G. E. P., & Jenkins, G. M. (1976), *Time Series Analysis: Forecasting and
Control* -- a small teaching dataset widely redistributed for time-series
education. If you need strict dataset provenance/licensing for your use
case, replace it with your own series; this repo’s beginner path is
designed so the dataset choice doesn’t affect the core API concepts.

## License

This repository's original tutorial code and documentation are licensed
under the [MIT License](LICENSE). TimesFM itself (the model and the
official `timesfm` package) is licensed separately by Google under
[Apache 2.0](https://github.com/google-research/timesfm/blob/master/LICENSE)
-- read that license yourself before any commercial use; this repo isn't
legal advice.

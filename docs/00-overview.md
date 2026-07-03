# 00 - What Is TimesFM, and Why Should You Care?

> Part of the [zero-to-mastery learning path](../README.md#learning-path). Previous: none (start here). Next: [01 - Prerequisites](01-prerequisites.md).

## The one-paragraph version

**TimesFM** ("Time Series Foundation Model") is a neural network built by
Google Research that forecasts numbers over time -- sales, server load,
patient counts, temperature, anything recorded as a sequence of values at
regular intervals. The thing that makes it different from a typical
forecasting model: you don't train it on your data first. You show it the
recent history of your series and it predicts what comes next, **without
ever having seen that series before**. This is called *zero-shot
forecasting*.

## The problem it solves

Classical forecasting (ARIMA, exponential smoothing, gradient-boosted trees,
LSTMs) all share one requirement: you need enough historical data *for that
specific series* to fit a model, and you generally need to refit as new
data arrives. That's expensive in two ways:

- **Engineering cost** -- someone has to build and maintain a training
  pipeline for every new series or group of series.
- **Cold-start cost** -- a brand-new product, a newly opened store, a
  freshly deployed server has no history yet, so there's nothing to train
  on.

TimesFM is pretrained once, by Google, on a large and varied collection of
time series, then shipped as a checkpoint you download and run directly.
The pattern is the same one that made large language models useful before
anyone fine-tuned them: pretrain broadly, then get useful zero-shot
behavior on tasks the model never explicitly saw.

> **Why this matters in practice:** if you've ever needed "just a rough
> forecast" for a dataset too small or too new to justify building a custom
> model, TimesFM gives you a strong baseline in minutes instead of days.

## How it fits into the bigger picture

TimesFM belongs to a young category of models sometimes called **time
series foundation models** -- pretrained, general-purpose forecasters meant
to be used zero-shot or lightly fine-tuned, the same way BERT or GPT are
general-purpose text models. Google Research is not the only group working
in this space; you'll also encounter names like Chronos (Amazon) and Moirai
(Salesforce) if you read further. This repository only covers TimesFM in
depth -- that's a deliberate scope choice, not a claim that it's the only
option worth knowing.

## What TimesFM actually is, technically (a first pass)

Under the hood, TimesFM is a **decoder-only, patched-decoder attention
model**: it borrows the "predict the next chunk from everything before it"
architecture that powers modern language models, but the "tokens" are
patches (short contiguous windows) of numeric time series values instead of
words. You do not need to understand transformer internals to use TimesFM
-- Chapter [04 - Core Concepts](04-core-concepts.md) builds that intuition
gradually once you've run something real. For the full technical
description, the primary source is the paper:

> Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). *A decoder-only foundation
> model for time-series forecasting.* Proceedings of ICML 2024.
> [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)

## Where you'll find TimesFM in the real world

TimesFM isn't just a research artifact -- Google ships it inside several
products, which is a useful signal of how it's meant to be used:

- **[BigQuery ML](https://cloud.google.com/bigquery/docs/timesfm-model)** --
  call it directly from SQL for large-scale forecasting.
- **[Google Sheets (Connected Sheets)](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html)**
  -- forecast a column of numbers from a spreadsheet.
- **Vertex AI Model Garden** -- the official TimesFM repository lists a
  "Vertex Model Garden" option for a hosted, dockerized endpoint for
  programmatic/agentic use. The console listing itself requires a Google
  Cloud login; public entry point:
  <https://cloud.google.com/model-garden>.

This repository uses the **open-source Python package** directly, which is
the same model powering those products, but run locally on your own
machine or infrastructure.

## What this repo will and won't teach you

This is a learning repository, not TimesFM's own documentation. It exists
to take you from "I've never used a forecasting model" to "I can run,
evaluate, and reason about TimesFM on my own data." It does **not**
replace the official repository, and where the two disagree, the official
source wins:

- Official repo: <https://github.com/google-research/timesfm>
- Official blog post: <https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/>
- Hugging Face model collection: <https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6>

> **Disclaimer, straight from the official repository:** "This open version
> is not an officially supported Google product." TimesFM is Apache-2.0
> licensed research software, actively maintained, but without the support
> guarantees of a commercial Google Cloud product. Keep that in mind before
> depending on it for anything business-critical without your own
> validation -- see [07 - Evaluation](07-evaluation.md).

## Model versions you'll hear about

| Version | Status | Notes |
|---|---|---|
| 1.0 | Archived | Original release. Code preserved under `v1/` in the official repo; install with `pip install timesfm==1.3.0` if you need it. |
| 2.0 | Archived | 500M parameters, 2048-token context. Superseded by 2.5. |
| **2.5** | **Current, used in this repo** | 200M parameters, up to 16k context, optional continuous quantile head, simplified API (no more manual frequency flag). Released September 2025. |

Full comparison in [04 - Core Concepts](04-core-concepts.md).

## Check your understanding

- [ ] Can you explain "zero-shot forecasting" to someone who's never heard
      the term, in one sentence?
- [ ] Name one real Google product that uses TimesFM.
- [ ] True or false: TimesFM must be retrained on your data before you can
      use it. *(False -- that's the whole point. Fine-tuning is optional
      and covered later, in [10 - Next Steps](10-next-steps.md).)*

---

**Next:** [01 - Prerequisites](01-prerequisites.md) -- what you need to
know and have installed before writing any code.

# 10 - Next Steps: Beyond Zero-Shot

> Previous: [09 - FAQ](09-faq.md). This is the last page of the core learning path.

You've now covered installation, your first forecast, the core concepts,
data preparation, a real workflow, and honest evaluation. Everything so far
has been **zero-shot**. This page maps out where to go from here, roughly
in order of how much additional effort each path costs.

## 1. Get more out of zero-shot first (lowest effort)

Before reaching for fine-tuning, make sure you've exhausted the
zero-shot toolkit covered in this repo:

- **Covariates (XReg)** -- if you have known external signals (price,
  promotions, holidays, weather), see
  [04 - Core Concepts, section 6](04-core-concepts.md#6-covariates--xreg)
  and [`examples/03_covariates_xreg_example.py`](../examples/03_covariates_xreg_example.py).
  This is usually cheaper and more interpretable than fine-tuning, and
  often closes most of the gap a custom-trained model would have offered.
- **Proper backtesting** -- re-read [07 - Evaluation](07-evaluation.md) and
  make sure any "TimesFM isn't good enough" conclusion is based on a
  rolling backtest against a real baseline, not one split.
- **Domain-aware config** -- `infer_is_positive`, appropriate
  `max_context`, and clean preprocessing (
  [05 - Data Format and Preprocessing](05-data-format-and-preprocessing.md))
  often matter more than they get credit for.

## 2. The ten applied case studies in this repository

[`notebooks/`](../notebooks/) contains full, real-dataset workflows for ten
industry forecasting problems (retail, electricity, manufacturing,
hospital patient volume, ATM cash demand, cloud capacity, airline demand,
warehouse orders, website traffic, financial transactions), each with:

- real data pulled from public sources (Kaggle, HealthData.gov, NAB),
- rolling backtests against naive baselines,
- forecasts converted into operational decisions (staffing plans,
  autoscaling policies, inventory reorder points, etc.).

Start with [`notebooks/README.md`](../notebooks/README.md), and see
[`HANDBOOK.md`](../HANDBOOK.md) for the full technical/operational manual
behind that tier, including a real, machine-validated run
(`docs/evidence/real_20260703_final_v6/`) with results reported honestly,
including the one domain (retail) where a naive baseline currently wins.

## 3. Fine-tuning (highest effort, only if zero-shot + covariates aren't enough)

Fine-tuning means adjusting TimesFM's own weights on your data. It's a
**separate workflow from everything else in this repo** -- it uses a
different Hugging Face checkpoint format and a different library stack.
This repository does not implement or validate a fine-tuning pipeline;
what follows points you to the **official, maintained example** rather
than an unverified reimplementation, per this repo's policy of not
inventing capabilities it hasn't tested.

Key facts, sourced directly from the
[official fine-tuning guide](https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/examples/finetuning/README.md):

- Uses a **different checkpoint**: `google/timesfm-2.5-200m-transformers`
  (a standard Hugging Face Transformers model, `TimesFm2_5ModelForPrediction`)
  rather than `google/timesfm-2.5-200m-pytorch` used throughout this repo.
- Fine-tunes with **[PEFT](https://github.com/huggingface/peft) / LoRA**
  (parameter-efficient fine-tuning) -- with LoRA rank 4 on all linear
  layers, only about 0.6% of parameters (~1.4M of ~232M) are trained.
- Training data is sampled as **random `(context, horizon)` windows** per
  series (an approach adapted from
  [Chronos-2](https://github.com/amazon-science/chronos-forecasting)),
  which is more data-efficient than always slicing the same fixed window.
- The model's forward pass computes a training loss directly when
  `future_values` are supplied, so the training loop is standard PyTorch --
  no custom loss function needed.
- Requires additional dependencies not in this repo's `pyproject.toml`:
  `transformers accelerate peft pandas pyarrow scikit-learn`.

To try it: clone the [official repository](https://github.com/google-research/timesfm),
follow `timesfm-forecasting/examples/finetuning/README.md` directly, and
treat it as its own project with its own environment.

## 4. Other official examples worth knowing about

The official repository ships additional examples this repo doesn't
duplicate -- listed here so you know they exist, without this repo making
claims about content it hasn't verified in depth:

- `timesfm-forecasting/examples/anomaly-detection/`
- `timesfm-forecasting/examples/global-temperature/`
- `tests/` -- the official unit test suite, a good reference for how the
  maintainers themselves test core layers, configs, and utilities.

Browse them at <https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples>.

## 5. Broaden your context: the wider field

TimesFM is one entry in a growing category of pretrained time-series
foundation models. Worth knowing they exist, even though this repo doesn't
teach them:

- **Chronos** (Amazon) -- a time series foundation model built on a
  language-model-style tokenization of numeric values.
- **Moirai** (Salesforce) -- a universal forecasting transformer trained
  across many domains and frequencies.

Reading the original TimesFM paper's related-work section is the most
reliable way to see how the authors themselves position it against these
and other approaches:

> Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). *A decoder-only foundation
> model for time-series forecasting.* ICML 2024.
> [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)

## Where to go from here, concretely

- **Want to apply this to your own data right now?** Copy the structure of
  [`examples/02_beginner_airline_passengers_forecast.py`](../examples/02_beginner_airline_passengers_forecast.py),
  swap in your CSV, and re-read [05](05-data-format-and-preprocessing.md)
  and [07](07-evaluation.md) as you go.
- **Want to see forecasting turned into business decisions?** Go to
  [`notebooks/`](../notebooks/).
- **Want to specialize the model itself?** Follow the official fine-tuning
  guide linked above, as its own separate project.
- **Want the full reference list this repo drew from?** See
  [`../README.md#references`](../README.md#references).

You now have a working, evaluated, honestly-benchmarked understanding of
TimesFM. That's mastery of the fundamentals -- everything past this point
is applying them to harder problems.

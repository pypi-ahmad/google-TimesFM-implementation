# 04 - Core Concepts

> Previous: [03 - First Forecast](03-first-forecast.md). Next: [05 - Data Format and Preprocessing](05-data-format-and-preprocessing.md).

You ran TimesFM in the last chapter. This chapter explains *why* it works,
building up from intuition to the technical picture. Every term is defined
in plain English before it's used technically.

## 1. Foundation model, zero-shot forecasting

**Plain English:** a "foundation model" is trained once on a huge, broad
dataset, then reused directly on new problems it never specifically trained
for. "Zero-shot" means using it on a new problem *without* any additional
training on that problem's data.

**Why this is unusual for forecasting:** traditionally, "forecasting model"
meant "a model fit to *this* series." A model trained on Store A's sales
couldn't forecast Store B without being refit. TimesFM breaks that
coupling: it was pretrained on a large, varied corpus of time series (see
[the model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
for the documented pretraining sources) and generalizes its learned
patterns -- trend, seasonality, typical noise shapes -- to series it has
never seen.

**The tradeoff:** zero-shot is fast to deploy but is not guaranteed to beat
a model purpose-built and tuned for your exact data. Section
[07 - Evaluation](07-evaluation.md) is about proving, on your own data,
whether it does.

## 2. Patched-decoder architecture

**Plain English first:** imagine reading a sentence one word at a time and
predicting the next word -- that's roughly how GPT-style language models
work (decoder-only, autoregressive). TimesFM does the same thing, except
instead of words it reads **patches**: short, fixed-length chunks of
consecutive numeric values (e.g. 32 time steps at once, rather than one at
a time). It predicts the next patch of values from all the patches before
it.

**Why patches instead of single values:** time series are much longer than
sentences (a year of hourly data is 8,760 points) and neighboring values
are highly redundant. Grouping values into patches shortens the sequence
the model has to attend over, the same way BPE tokenization shortens text
sequences compared to reading one character at a time.

**Technical detail, for the curious:** the full architecture -- patch
embedding, transformer decoder stack, output projection -- is described in
the paper:

> Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). *A decoder-only foundation
> model for time-series forecasting.* ICML 2024.
> [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)

## 3. Context and horizon

Two terms you already used in [03 - First Forecast](03-first-forecast.md):

- **Context** -- the historical values you feed in. More context generally
  gives the model more evidence about trend and seasonality, up to the
  `max_context` you compiled with (up to 16,384 steps for TimesFM 2.5).
- **Horizon** -- how many future steps you ask it to predict, up to
  `max_horizon`.

**Common beginner mistake:** assuming "more context is always better."
Very long context on a short, noisy series can just be more noise -- match
context length to how far back the *real* seasonal or structural pattern
you care about repeats (e.g. 2-3 years of monthly data to capture yearly
seasonality, not 10 years if the underlying process changed 3 years ago).

## 4. Internal normalization (don't normalize your data yourself)

TimesFM performs its own internal instance normalization on every series it
sees -- it rescales each input to a consistent range internally before
forecasting, and rescales its output back. This is why `ForecastConfig` has
`normalize_inputs=True` by default in the official examples, and why the
official fine-tuning guide is explicit about this:

> "TimesFM 2.5 applies its own internal instance normalisation (RevIN). Do
> not normalise your data externally -- feed raw values and let the model
> handle it."
> -- [official LoRA fine-tuning README](https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/examples/finetuning/README.md)

**Common beginner mistake:** applying your own z-score or min-max scaling
before calling `.forecast()`, out of habit from classical ML pipelines.
With TimesFM this is redundant at best and can distort results at worst --
feed raw values.

## 5. Quantile forecasts and uncertainty

**Plain English:** a **point forecast** is a single number ("next month:
450 units"). A **quantile forecast** is a range with associated
probabilities ("there's a 10% chance it's below 400, a 90% chance it's
below 520"). Real decisions -- how much inventory to hold, how much
capacity to provision -- usually depend on the *range*, not just the
average.

TimesFM 2.5's optional quantile head (`use_continuous_quantile_head=True`)
returns 10 values per forecasted step: the mean, then the 10th through 90th
percentile in 10-point increments. `fix_quantile_crossing=True` guarantees
these come back correctly ordered (p10 ≤ p50 ≤ p90).

**Why this matters:** two series can have the same point forecast but very
different uncertainty. A tight band means "trust this number"; a wide band
means "plan for a range, not a point." [`examples/02_beginner_airline_passengers_forecast.py`](../examples/02_beginner_airline_passengers_forecast.py)
plots this band directly.

## 6. Covariates / XReg

**Plain English:** a covariate is *extra information you already know*
that might explain some of the variation in your series -- price, whether
a day is a holiday, a promotional flag, day-of-week. XReg ("external
regression") is TimesFM's mechanism for using that extra information
alongside the raw history.

This is a **TimesFM 2.5-only feature** (TimesFM 1.0 does not support
`forecast_with_covariates()`), requires the `xreg` extra
(`uv sync --locked --group xreg`, which installs `timesfm[xreg]`), and works through
`model.forecast_with_covariates(...)`, which accepts:

- `dynamic_numerical_covariates` -- numeric values known for both the
  context *and* the horizon (e.g. planned price, calendar features).
- `dynamic_categorical_covariates` -- category labels known for context and
  horizon (e.g. holiday name, day-of-week).
- `static_categorical_covariates` -- one label per series that never
  changes (e.g. store type).
- `xreg_mode` -- controls *how* the covariates and TimesFM's own forecast
  are combined:

| Mode | How it works | Best when |
|---|---|---|
| `"xreg + timesfm"` (default) | TimesFM forecasts first; a regression is then fit on the *residual* (actual minus TimesFM's baseline) using the covariates | covariates mostly explain short-term deviations (e.g. a promotion bump) |
| `"timesfm + xreg"` | A regression on the covariates predicts the main signal first; TimesFM then forecasts the *leftover residual* | covariates explain most of the signal (e.g. temperature driving heating demand) |

> **Common beginner mistake:** calling `forecast_with_covariates()` on a
> model compiled the same way as for plain `.forecast()` calls. XReg needs
> one additional flag: `return_backcast=True` in `ForecastConfig`, or you'll
> hit `ValueError: For XReg, return_backcast must be set to True`. Recompile
> with that flag set before using covariates.

This documentation is grounded directly in the
[official covariates example](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/covariates-forecasting)
and in this repository's own [`examples/03_covariates_xreg_example.py`](../examples/03_covariates_xreg_example.py),
which mirrors the exact `forecast_with_covariates()` call pattern already
verified working in this repo's airline notebook. The official example also
demonstrates `static_categorical_covariates`; if you need parameters not
shown in `examples/03`, check that official example directly rather than
guessing.

> **Verified vs. unverified:** `xreg_mode="xreg + timesfm"` (the default)
> is exercised end-to-end by `examples/03` and this repo's airline
> notebook, and works as described. `xreg_mode="timesfm + xreg"` is
> documented by the official example but, at the time of writing, raised a
> shape-mismatch error in local testing against `timesfm==2.0.2` on a
> single-series call. This repo does not claim that mode works -- if you
> need it, test it directly against the version you have installed and
> check the [official issue tracker](https://github.com/google-research/timesfm/issues)
> if it fails for you too.

> **Also worth knowing:** `forecast_with_covariates()` requires
> `return_backcast=True` in `ForecastConfig`, unlike plain `.forecast()`
> calls. Forgetting it raises a clear `ValueError` -- see
> [08 - Troubleshooting](08-troubleshooting.md).

Full data-shape rules are in
[05 - Data Format and Preprocessing](05-data-format-and-preprocessing.md).

## 7. Zero-shot vs. fine-tuning

Everything above is **zero-shot**: no training step, just load and
forecast. TimesFM 2.5 also supports **fine-tuning** -- adjusting the
model's weights on your own data to specialize it -- via a *separate*
Hugging Face Transformers-compatible checkpoint
(`google/timesfm-2.5-200m-transformers`) combined with
[PEFT/LoRA](https://github.com/huggingface/peft) for parameter-efficient
training. This is covered in [10 - Next Steps](10-next-steps.md); it's an
advanced, optional path, not something you need for the rest of this
learning journey.

## 8. Why the "frequency" parameter disappeared

If you read older TimesFM tutorials (1.0 or 2.0 era), you'll see a required
`freq` argument (e.g. `0` for daily, `1` for weekly). **TimesFM 2.5 removed
this** -- the model infers the right behavior without an explicit frequency
indicator, simplifying the API you saw in
[03 - First Forecast](03-first-forecast.md). If you encounter code using
`freq=`, it's targeting an older version.

## 9. Version comparison

| | 1.0 | 2.0 | 2.5 (used in this repo) |
|---|---|---|---|
| Parameters | -- | 500M | 200M |
| Max context | shorter, fixed | 2,048 | up to 16,384 |
| Quantile head | -- | -- | optional, continuous, up to 1,000-step horizon |
| Frequency indicator | required | required | removed |
| Covariates (XReg) | not supported | -- | supported |
| Package install | `pip install timesfm==1.3.0` | archived, `v1/` in official repo | `pip install timesfm[torch]` |

Source: [official TimesFM README, "Update - Sept. 15, 2025"](https://github.com/google-research/timesfm#update---sept-15-2025)
and archived-version notes in the same document.

## A quick note on PyPI version vs. model version

The PyPI package is versioned independently from the model name: as of this
writing the package is `timesfm==2.0.2` on PyPI, but it ships the *model*
called "TimesFM 2.5." Don't confuse the two numbers -- this repo pins the
**package** version (`timesfm>=2.0.2`) and always means the **2.5 model**
when it says "TimesFM 2.5" in prose.

## Check your understanding

- [ ] What's the difference between context and horizon?
- [ ] Why shouldn't you normalize your data before calling `.forecast()`?
- [ ] When would you pick `xreg_mode="timesfm + xreg"` over the default?
- [ ] What changed between TimesFM 2.0 and 2.5 that most affects how you'd
      call the API?

---

**Next:** [05 - Data Format and Preprocessing](05-data-format-and-preprocessing.md).

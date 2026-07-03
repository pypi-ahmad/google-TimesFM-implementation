# 09 - FAQ

> Previous: [08 - Troubleshooting](08-troubleshooting.md). Next: [10 - Next Steps](10-next-steps.md).

Conceptual questions, as opposed to error messages (those are in
[08 - Troubleshooting](08-troubleshooting.md)).

**Is TimesFM a chatbot / LLM I can talk to?**
No. It takes numeric sequences in and returns numeric sequences out. It
shares an architectural lineage with language models (decoder-only,
attention-based -- see [04 - Core Concepts](04-core-concepts.md)) but has
no text interface, no conversation, no world knowledge to query.

**Do I need to train it before use?**
No -- that's the entire point of a zero-shot foundation model (see
[00 - Overview](00-overview.md)). You *can* fine-tune it (see
[10 - Next Steps](10-next-steps.md)), but it's optional and separate from
normal use.

**Will TimesFM always beat my existing forecasting model?**
Not guaranteed, and this repo will not claim otherwise. It's a strong
zero-shot baseline. Whether it beats a model purpose-built and tuned on
your specific data is an empirical question you answer with
[07 - Evaluation](07-evaluation.md), not an assumption you make going in.

**Can it do classification, anomaly detection, or clustering?**
This repo only covers point and quantile *forecasting*, which is what the
core API (`forecast()` / `forecast_with_covariates()`) is built for. The
official repository documents an anomaly-detection example separately
(`timesfm-forecasting/examples/anomaly-detection/`) -- if you need that,
go to the [official repository](https://github.com/google-research/timesfm)
directly rather than assuming this repo covers it.

**Does it need a GPU?**
No. CPU works for every example in this repo; a GPU only makes it faster.
See [01 - Prerequisites](01-prerequisites.md#hardware).

**How much history (context) do I need?**
Enough to cover at least one full cycle of whatever seasonality matters to
you (e.g. 12+ months of monthly data for yearly seasonality). See
[04 - Core Concepts, section 3](04-core-concepts.md#3-context-and-horizon).
There's no hard minimum enforced by the API, but very short context gives
the model little to work with.

**Can it forecast multiple related series at once (e.g. all my stores)?**
Yes -- pass a list of series to `inputs`, one call handles all of them,
and lengths can differ across series. See
[05 - Data Format and Preprocessing](05-data-format-and-preprocessing.md#batching-multiple-series).

**What's the difference between the PyPI package version and "TimesFM 2.5"?**
The PyPI package (`timesfm`) has its own version number (`2.0.2` as of
this writing) independent from the model's marketing name ("TimesFM 2.5").
Installing the current package gives you the 2.5 model. See
[04 - Core Concepts](04-core-concepts.md#a-quick-note-on-pypi-version-vs-model-version).

**Is this an official Google product?**
No. Straight from the [official repository](https://github.com/google-research/timesfm):
"This open version is not an officially supported Google product." It's
Apache-2.0 licensed, actively maintained research software. TimesFM the
*model* also powers supported products (BigQuery ML, Connected Sheets,
Vertex Model Garden -- see [00 - Overview](00-overview.md)), but the
open-source package you're using here does not carry those support
guarantees.

**Can I use TimesFM commercially?**
The model and official code are Apache-2.0 licensed, which permits
commercial use -- but read the
[actual license text](https://github.com/google-research/timesfm/blob/master/LICENSE)
yourself rather than relying on a summary; this repo cannot give legal
advice. This repository's own original tutorial code and docs are MIT
licensed (see [`../LICENSE`](../LICENSE)) -- a separate license from
TimesFM itself. See [00 - Overview](00-overview.md#where-youll-find-timesfm-in-the-real-world)
for the distinction between "using the open package" and "using a Google
product built on it."

**Why do some of this repo's notebooks show TimesFM losing to a naive
baseline (e.g. the retail demand case study)?**
Because that's what the actual backtest run showed, and this repo reports
real results, not cherry-picked ones -- see
[`docs/evidence/real_20260703_final_v6/summary.md`](evidence/real_20260703_final_v6/summary.md)
and [07 - Evaluation](07-evaluation.md) for why "always compare to a
baseline, and report it honestly even when the baseline wins" is the whole
point of evaluating properly.

---

**Next:** [10 - Next Steps](10-next-steps.md).

# Release Notes

## v2.0.0 - 2026-07-03

### Summary

v1.0.0 published this repository as a production-style TimesFM project
across 10 applied case studies, but had no beginner-friendly learning path:
no conceptual documentation, no minimal example that ran without Kaggle
credentials, an outdated `timesfm` pin, and no disclaimer that TimesFM is
not an officially supported Google product. v2.0.0 is a substantial rework
that turns this into an actual zero-to-mastery TimesFM learning resource
while keeping the ten applied case studies as an advanced tier.

### Audit findings addressed

- No beginner onboarding existed; the simplest runnable thing required
  Kaggle API credentials and a multi-GB dataset download.
- `pyproject.toml` pinned `timesfm[torch,xreg]>=2.0.1` against a current
  upstream PyPI release of `2.0.2`, and the package name
  (`timesfm-retail-demand`) didn't match the repo's actual 10-domain scope.
- No tests existed anywhere in the repository.
- No disclaimer that the open-source TimesFM package is not an officially
  supported Google product.
- References were a flat, uncontextualized link dump.
- `HANDBOOK.md` embedded a hardcoded local filesystem path and a local
  Kaggle username, neither useful nor appropriate for other readers.

### Research performed

Verified directly against primary sources before writing any teaching
material: the live [official TimesFM repository](https://github.com/google-research/timesfm)
(README, `pyproject.toml`, `tests/`, official examples for covariates and
fine-tuning), the [arXiv paper](https://arxiv.org/abs/2310.10688), and the
[Hugging Face model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch).
Every runnable example added in this release was executed end-to-end
against the real model as part of QA -- one real API constraint
(`return_backcast=True` required for `forecast_with_covariates()`) and one
real upstream limitation (`xreg_mode="timesfm + xreg"` erroring on
single-series input in `timesfm==2.0.2`) were discovered this way and are
now documented rather than silently avoided.

### Added

- `docs/00-overview.md` through `docs/10-next-steps.md`: an 11-page
  zero-to-mastery learning path (what TimesFM is, prerequisites,
  installation, first forecast walked through line by line, core concepts,
  data preparation, a real-data workflow, evaluation/backtesting,
  troubleshooting, FAQ, and next steps including fine-tuning).
- `examples/`: four tiered, standalone, verified-runnable scripts (minimal
  synthetic forecast, beginner real-dataset forecast, covariates/XReg, and
  rolling-backtest evaluation), plus a bundled public-domain dataset
  (`examples/data/airline_passengers.csv`) so the beginner path needs no
  external accounts or downloads beyond the model weights.
- `tests/`: environment/API-surface smoke tests (no network needed) and
  end-to-end model smoke tests (`pytest -m model`), mirroring the testing
  discipline of the official TimesFM repository's own `tests/` directory.
- `notebooks/README.md`: frames the 10 applied case studies as an advanced
  tier and states the prerequisite beginner path explicitly.
- Proper, contextualized references throughout `README.md` and `docs/`,
  each tied to the specific claim it supports.

### Changed

- `README.md`: fully rewritten as a beginner-first entry point (purpose,
  audience, quickstart, learning roadmap, repo structure, references,
  scope/limitations, license clarification).
- `pyproject.toml`: package renamed to `google-timesfm-implementation`
  (matching actual repo scope), `timesfm` bumped to `>=2.0.2`, `pytest`
  added as a dev dependency, pytest markers registered.
- `HANDBOOK.md`: repositioned as the advanced/production-tier manual with
  an explicit pointer to the new beginner path; removed the hardcoded local
  filesystem path and local Kaggle username; bumped to `v2.0.0`.
- `HANDBOOK.pdf`: regenerated from the updated `HANDBOOK.md`.

### Not changed

- The ten applied case-study notebooks and their supporting scripts are
  preserved as-is (already validated, real-run evidence intact) and
  repositioned rather than rewritten, per this release's goal of improving
  the repo without discarding working material.
- Reported results are unchanged and still reported honestly, including
  the retail domain where a naive seasonal baseline currently beats
  TimesFM in the validated run (see `docs/evidence/real_20260703_final_v6/summary.md`).

## v1.0.0 - 2026-07-03

### Highlights

- Published production-grade documentation set:
  - `README.md`
  - `HANDBOOK.md`
  - `HANDBOOK.pdf`
  - `docs/evidence/real_20260703_final_v6/summary.{md,json}`
- Added documentation evidence generator:
  - `scripts/build_docs_evidence.py`
- Documented and validated real run evidence for all 10 forecasting workflows plus retail script output.

### Validated Run Evidence

Authoritative run:
- `artifacts/final_real_runs/real_20260703_final_v6/final_run_report.json`
- `artifacts/final_real_runs/real_20260703_final_v6/final_validation_report.json`

Status:
- Final run: `PASS` (`11/11` successful targets)
- Validation: `PASS` (`98/98` checks)

### Notable Result Pattern

- TimesFM is best on mean MAE for non-retail domains in this run artifact set.
- Retail validation metrics (`nwrmsle`) are currently better for `naive_seasonal7` baseline than `tfm` in the reused retail artifacts.

### Packaging and Publication Notes

- Repository publish strategy is hybrid for large data/artifact footprint:
  - Code, notebooks, docs, and lightweight evidence in GitHub
  - Full heavy snapshot intended for dataset hosting (Kaggle)

### References

- TimesFM repo: https://github.com/google-research/timesfm
- TimesFM blog: https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/
- HF model card: https://huggingface.co/google/timesfm-2.5-200m-pytorch


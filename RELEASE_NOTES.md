# Release Notes

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


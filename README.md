# Google TimesFM Implementation: 10 Real-World Forecasting Workflows

Production-style implementation of Google TimesFM for 10 time-series forecasting problems, with runnable notebooks, a production script for retail demand, and validated real-run artifacts.

## Project Overview

This repository applies **Google TimesFM (decoder-only foundation model for time series forecasting)** across 10 industry scenarios:

1. Retail demand forecasting
2. Electricity load forecasting
3. Manufacturing machine sensor forecasting
4. Hospital patient volume forecasting
5. ATM cash demand forecasting
6. Cloud infrastructure capacity forecasting
7. Airline passenger demand forecasting
8. Warehouse order volume forecasting
9. Website traffic forecasting
10. Financial transaction volume forecasting

The project includes:
- End-to-end notebooks for each use case (`notebooks/*.ipynb`)
- Executed notebooks with outputs (`notebooks/*.executed.ipynb`)
- Production script for retail demand (`scripts/retail_demand_timesfm_favorita.py`)
- Orchestrated real-run and validation tooling (`scripts/final_real_run.py`, `scripts/validate_final_real_run.py`)
- Real artifact bundle (`artifacts/final_real_runs/real_20260703_final_v6`)

## Problem Statement

Operational teams in retail, utilities, healthcare, logistics, fintech, and cloud infrastructure need reliable short- and medium-horizon forecasts for planning and control decisions.

Poor forecasts create direct business risk:
- Stockouts or overstock
- Generator over/under commitment
- Maintenance delays
- Staffing mismatches
- Cash-out ATM events
- Under/over-provisioned compute
- Service degradation and avoidable cost

## Objectives

- Build reproducible TimesFM-based forecasting pipelines for real, web-downloaded datasets.
- Compare TimesFM to practical baselines (naive/seasonal) with backtesting.
- Translate forecasts into operational planning outputs (inventory, staffing, autoscaling, capacity, routing, logistics).
- Validate full execution with machine-checkable reports.

## Architecture and Approach

### Core implementation patterns

- **Modeling**: TimesFM inference on historical context windows, optional covariate/xreg mode in retail workflow.
- **Evaluation**: per-domain backtesting metrics from CSV artifacts.
- **Decision layer**: domain-specific planning outputs from forecast values (for example, staffing plans, autoscaling plans, capacity plans).
- **Reproducibility**:
  - Deterministic seed controls
  - Structured run orchestration and validation reports
  - Traceable artifact directories per run ID

### Main components

- `scripts/retail_demand_timesfm_favorita.py`
  - Data ingestion from Kaggle competition
  - Series panel preparation
  - TimesFM forecasting horizons (`7/14/30`)
  - Inventory policy and warehouse allocation (heuristic + OR-Tools MILP)
  - Kaggle-format submission export
- `scripts/final_real_run.py`
  - Executes the full multi-notebook + script workflow
  - Produces `final_run_report.json`
- `scripts/validate_final_real_run.py`
  - Checks executed notebooks and required artifacts
  - Produces `final_validation_report.json`
- `scripts/build_docs_evidence.py`
  - Generates documentation evidence snapshots from run outputs

## Implementation Process

1. Define and standardize each business problem as a forecasting task.
2. Download data from web sources (Kaggle, HealthData.gov, NAB GitHub raw files).
3. Build cleaned and regularized time-series panels.
4. Run TimesFM and baseline models.
5. Evaluate via backtest metrics.
6. Generate operational outputs for each domain.
7. Execute orchestrated final run and strict validation.
8. Extract evidence for documentation.

## Setup and Installation

### Requirements

- Python `>=3.11`
- `uv` package manager
- Kaggle credentials (for Kaggle-backed datasets)

### Environment setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

Optional notebook kernel setup:

```bash
python -m ipykernel install --user --name timesfm-local --display-name "Python (timesfm-local)"
```

## Usage

### A) Run retail demand script directly

```bash
uv run python scripts/retail_demand_timesfm_favorita.py \
  --data-root data/favorita_real \
  --artifacts-root artifacts/retail_demand_timesfm_real \
  --horizons 7 14 30 \
  --device auto \
  --use-xreg \
  --run-milp \
  --include-submission
```

### B) Execute full real run orchestrator

```bash
.venv/bin/python scripts/final_real_run.py \
  --run-id real_20260703_final_v6 \
  --include-notebooks \
  --include-scripts \
  --reuse-retail-artifacts-from artifacts/retail_demand_timesfm_real \
  --reuse-retail-executed-notebook notebooks/retail_demand_timesfm_favorita.executed.ipynb
```

### C) Validate run outputs

```bash
.venv/bin/python scripts/validate_final_real_run.py \
  --run-id real_20260703_final_v6 \
  --strict-real-checks
```

### D) Build evidence snapshot for docs

```bash
.venv/bin/python scripts/build_docs_evidence.py --run-id real_20260703_final_v6
```

## Experiments and Workflow

Primary validated run:
- Run ID: `real_20260703_final_v6`
- Run report: `artifacts/final_real_runs/real_20260703_final_v6/final_run_report.json`
- Validation report: `artifacts/final_real_runs/real_20260703_final_v6/final_validation_report.json`
- Evidence summary: `docs/evidence/real_20260703_final_v6/summary.md`

Validation status:
- Final run: `PASS` (`11/11` targets)
- Validation: `PASS` (`98/98` checks)

## Outputs and Results

### Key output directories

- `artifacts/final_real_runs/real_20260703_final_v6/airline_demand/`
- `artifacts/final_real_runs/real_20260703_final_v6/atm_cash_demand/`
- `artifacts/final_real_runs/real_20260703_final_v6/cloud_capacity/`
- `artifacts/final_real_runs/real_20260703_final_v6/electricity_load/`
- `artifacts/final_real_runs/real_20260703_final_v6/hospital_patient_volume/`
- `artifacts/final_real_runs/real_20260703_final_v6/manufacturing_sensor/`
- `artifacts/final_real_runs/real_20260703_final_v6/payment_transactions/`
- `artifacts/final_real_runs/real_20260703_final_v6/warehouse_orders/`
- `artifacts/final_real_runs/real_20260703_final_v6/website_traffic/`
- `artifacts/final_real_runs/real_20260703_final_v6/retail_demand_notebook/`
- `artifacts/final_real_runs/real_20260703_final_v6/retail_demand_script/`

### Observed metric pattern (from real run artifacts)

- In 9/11 domain summaries (all non-retail domains), TimesFM is the best model on mean `mae` over recorded horizons.
- In retail validation (`nwrmsle`), baseline `naive_seasonal7` outperforms `tfm` in this run configuration.
- Full details are in `docs/evidence/real_20260703_final_v6/summary.md`.

## Limitations

- Retail metrics in this validated run reflect **reuse mode** (`reuse_retail_artifacts`) for final orchestration, not full recompute during that run ID.
- Some domains use heuristic operational conversion layers (for example staffing/autoscaling conversion factors) that should be calibrated per production environment.
- Data access for several workflows depends on external services (Kaggle/API/network availability).
- No CI workflow is currently configured in this repository.

## Future Improvements

- Add CI checks for notebook execution health and artifact schema validation.
- Add probabilistic interval calibration and business KPI-oriented loss functions.
- Introduce hierarchical reconciliation for multi-entity forecasting domains.
- Add experiment tracking integration (MLflow/W&B) for run registry and drift monitoring.
- Extend deployment packaging for scheduled batch inference.

## References and Sources

### Core model and official docs

- Google Research TimesFM repository: https://github.com/google-research/timesfm
- Google Research blog (TimesFM): https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/
- Hugging Face model used in code (`google/timesfm-2.5-200m-pytorch`): https://huggingface.co/google/timesfm-2.5-200m-pytorch

### Framework/library docs used by implementation

- Kaggle API docs: https://www.kaggle.com/docs/api
- OR-Tools linear solver docs: https://developers.google.com/optimization
- Polars docs: https://pola.rs/
- PyArrow docs: https://arrow.apache.org/docs/python/
- Pydantic docs: https://docs.pydantic.dev/

### Project dataset sources (as referenced in notebooks/scripts)

- Kaggle Favorita competition: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting
- Kaggle ATM dataset: https://www.kaggle.com/datasets/zoya77/atm-cash-demand-forecasting-and-management
- Kaggle airline dataset: https://www.kaggle.com/datasets/saadharoon27/airlines-dataset
- Kaggle hourly energy dataset: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- Kaggle manufacturing dataset: https://www.kaggle.com/datasets/anshtanwar/metro-train-dataset
- Kaggle Olist e-commerce dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Kaggle NASA web logs dataset: https://www.kaggle.com/datasets/pasanbhanuguruge/nasa-http-logs-dataset-processed-for-lstm-models
- Kaggle financial transactions dataset: https://www.kaggle.com/datasets/sergionefedov/fraud-detection-1m-transactions-7-fraud-types
- HealthData.gov hospital dataset: https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh
- NAB repository (AWS CloudWatch series): https://github.com/numenta/NAB


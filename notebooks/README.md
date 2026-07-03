# Notebooks: Advanced / Applied Case Studies

**Start here first if you haven't yet:** these ten notebooks assume you've
already completed the beginner learning path in
[`../docs/00-overview.md`](../docs/00-overview.md) through
[`../docs/07-evaluation.md`](../docs/07-evaluation.md), and the runnable
scripts in [`../examples/`](../examples/). They use the same TimesFM APIs
introduced there, but at real-world scale: full datasets pulled from public
sources, rolling backtests across many series, and forecasts converted into
concrete operational decisions.

If you came here directly and terms like *context*, *horizon*, *covariate*,
or *backtest* aren't yet second nature, go to
[`../docs/00-overview.md`](../docs/00-overview.md) first -- these notebooks
do not re-explain them.

## What's here

| Notebook | Domain | Forecast target | Extra technique shown |
|---|---|---|---|
| `retail_demand_timesfm_favorita.ipynb` | Retail | Product sales | Covariates (XReg), OR-Tools MILP allocation, Kaggle submission format |
| `electricity_load_timesfm_pjm.ipynb` | Utilities | Power demand | Weekly operational scheduling |
| `manufacturing_sensor_timesfm.ipynb` | Manufacturing | Sensor temperature | Predictive-maintenance alerting |
| `hospital_patient_volume_timesfm.ipynb` | Healthcare | Patient arrivals | Staffing plan generation |
| `atm_cash_demand_timesfm.ipynb` | Banking/logistics | Cash withdrawals | Refill/logistics planning |
| `cloud_capacity_timesfm.ipynb` | Cloud infrastructure | CPU utilization | Autoscaling policy |
| `airline_passenger_forecasting_timesfm.ipynb` | Aviation | Route passenger demand | Calendar covariates, pricing/fleet/crew planning |
| `warehouse_order_volume_timesfm.ipynb` | Logistics | Daily order volume | Operations planning |
| `website_traffic_timesfm.ipynb` | Web/SRE | Hourly traffic | Autoscaling + infra KPI planning |
| `financial_transaction_volume_timesfm.ipynb` | Fintech | Transaction volume | Capacity/fraud-monitoring planning |

Each `*.ipynb` is the source notebook; the matching `*.executed.ipynb` is
the same notebook already run, with real outputs saved in place, so you can
read the results without re-running anything.

## Full technical manual

[`../HANDBOOK.md`](../HANDBOOK.md) is the complete operational reference
for this tier: environment setup for Kaggle-backed data pulls, code
walkthroughs of the supporting scripts, the full run/validation/evidence
pipeline, and a real, machine-checked validation run with honestly-reported
results (including the one domain where a naive baseline currently beats
TimesFM). Read it before modifying or extending these notebooks.

## Setup specific to this tier

These notebooks need Kaggle API credentials for the datasets that come from
Kaggle competitions/datasets. See
[`../HANDBOOK.md`, section 4](../HANDBOOK.md#4-environment-setup) for setup,
and [`../docs/08-troubleshooting.md`](../docs/08-troubleshooting.md) for
general (non-Kaggle) environment issues.

## Should I read the notebooks or the beginner examples first?

The beginner examples (`../examples/`) exist specifically because these
notebooks are too large and too domain-specific to be a good *first*
introduction to TimesFM. Read them in this order:

1. [`../docs/`](../docs/) 00 through 07
2. [`../examples/`](../examples/) 01 through 04
3. This directory

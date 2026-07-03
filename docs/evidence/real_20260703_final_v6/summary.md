# Evidence Summary: real_20260703_final_v6

## Run Health

- Run status: `PASS` (11/11)
- Validation status: `PASS` (98/98)
- Run report: `/home/ahmad/AI/Github/google-TimesFM-implementation/artifacts/final_real_runs/real_20260703_final_v6/final_run_report.json`
- Validation report: `/home/ahmad/AI/Github/google-TimesFM-implementation/artifacts/final_real_runs/real_20260703_final_v6/final_validation_report.json`

## Domain Metrics Snapshot

| Domain | Metric | TimesFM | Best Baseline | Delta (TimesFM - Baseline) | Best Model Overall |
| --- | --- | ---: | --- | ---: | --- |
| Airline Passenger Forecasting | mae | 20.228029 | naive_last: 22.867857 | -2.639828 | timesfm: 20.228029 |
| ATM Cash Demand Forecasting | mae | 3141.915321 | seasonal7: 4215.261604 | -1073.346282 | timesfm: 3141.915321 |
| Cloud Infrastructure Capacity Forecasting | mae | 2.650035 | seasonal_day: 4.266599 | -1.616564 | timesfm: 2.650035 |
| Electricity Load Forecasting | mae | 1046.888585 | seasonal24: 1413.846728 | -366.958143 | timesfm: 1046.888585 |
| Hospital Patient Volume Forecasting | mae | 534.200979 | seasonal7: 740.085083 | -205.884104 | timesfm: 534.200979 |
| Manufacturing Machine Sensor Forecasting | mae | 2.658274 | naive_last: 3.488952 | -0.830678 | timesfm: 2.658274 |
| Financial Transaction Volume Forecasting | mae | 5.212540 | seasonal168: 7.608730 | -2.396190 | timesfm: 5.212540 |
| Warehouse Order Volume Forecasting | mae | 46.714279 | seasonal7: 56.442261 | -9.727983 | timesfm: 46.714279 |
| Website Traffic Forecasting | mae | 354.303942 | seasonal168: 577.554242 | -223.250300 | timesfm: 354.303942 |
| Retail Demand Forecasting (Notebook) | nwrmsle | 1.948008 | naive_seasonal7: 1.156076 | 0.791932 | naive_seasonal7: 1.156076 |
| Retail Demand Forecasting (Script) | nwrmsle | 1.948008 | naive_seasonal7: 1.156076 | 0.791932 | naive_seasonal7: 1.156076 |

## Execution Commands

| Target | Type | Status | Duration (s) | Reuse Mode | Command |
| --- | --- | --- | ---: | --- | --- |
| airline_passenger_forecasting_timesfm.ipynb | notebook | success | 41.75 | False | `/home/ahmad/AI/Github/google-TimesFM-implementation/.venv/bin/python -m jupyter nbconvert --to notebook --execute /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/src/airline_passenger_forecasting_timesfm.ipynb --output airline_passenger_forecasting_timesfm.executed.ipynb --output-dir /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/executed --ExecutePreprocessor.timeout=-1` |
| atm_cash_demand_timesfm.ipynb | notebook | success | 24.34 | False | `/home/ahmad/AI/Github/google-TimesFM-implementation/.venv/bin/python -m jupyter nbconvert --to notebook --execute /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/src/atm_cash_demand_timesfm.ipynb --output atm_cash_demand_timesfm.executed.ipynb --output-dir /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/executed --ExecutePreprocessor.timeout=-1` |
| cloud_capacity_timesfm.ipynb | notebook | success | 17.41 | False | `/home/ahmad/AI/Github/google-TimesFM-implementation/.venv/bin/python -m jupyter nbconvert --to notebook --execute /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/src/cloud_capacity_timesfm.ipynb --output cloud_capacity_timesfm.executed.ipynb --output-dir /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/executed --ExecutePreprocessor.timeout=-1` |
| electricity_load_timesfm_pjm.ipynb | notebook | success | 16.37 | False | `/home/ahmad/AI/Github/google-TimesFM-implementation/.venv/bin/python -m jupyter nbconvert --to notebook --execute /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/src/electricity_load_timesfm_pjm.ipynb --output electricity_load_timesfm_pjm.executed.ipynb --output-dir /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/executed --ExecutePreprocessor.timeout=-1` |
| financial_transaction_volume_timesfm.ipynb | notebook | success | 20.34 | False | `/home/ahmad/AI/Github/google-TimesFM-implementation/.venv/bin/python -m jupyter nbconvert --to notebook --execute /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/src/financial_transaction_volume_timesfm.ipynb --output financial_transaction_volume_timesfm.executed.ipynb --output-dir /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/executed --ExecutePreprocessor.timeout=-1` |
| hospital_patient_volume_timesfm.ipynb | notebook | success | 13.92 | False | `/home/ahmad/AI/Github/google-TimesFM-implementation/.venv/bin/python -m jupyter nbconvert --to notebook --execute /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/src/hospital_patient_volume_timesfm.ipynb --output hospital_patient_volume_timesfm.executed.ipynb --output-dir /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/executed --ExecutePreprocessor.timeout=-1` |
| manufacturing_sensor_timesfm.ipynb | notebook | success | 19.04 | False | `/home/ahmad/AI/Github/google-TimesFM-implementation/.venv/bin/python -m jupyter nbconvert --to notebook --execute /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/src/manufacturing_sensor_timesfm.ipynb --output manufacturing_sensor_timesfm.executed.ipynb --output-dir /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/executed --ExecutePreprocessor.timeout=-1` |
| retail_demand_timesfm_favorita.ipynb | notebook | success | 0.0 | True | `reuse_retail_artifacts` |
| warehouse_order_volume_timesfm.ipynb | notebook | success | 14.43 | False | `/home/ahmad/AI/Github/google-TimesFM-implementation/.venv/bin/python -m jupyter nbconvert --to notebook --execute /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/src/warehouse_order_volume_timesfm.ipynb --output warehouse_order_volume_timesfm.executed.ipynb --output-dir /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/executed --ExecutePreprocessor.timeout=-1` |
| website_traffic_timesfm.ipynb | notebook | success | 14.45 | False | `/home/ahmad/AI/Github/google-TimesFM-implementation/.venv/bin/python -m jupyter nbconvert --to notebook --execute /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/src/website_traffic_timesfm.ipynb --output website_traffic_timesfm.executed.ipynb --output-dir /home/ahmad/AI/Github/google-TimesFM-implementation/notebooks/final_real_runs/real_20260703_final_v6/executed --ExecutePreprocessor.timeout=-1` |
| retail_demand_timesfm_favorita.py | script | success | 0.0 | True | `reuse_retail_artifacts` |

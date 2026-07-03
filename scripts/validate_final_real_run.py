from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import pyarrow.parquet as pq


NOTEBOOK_SLUGS: dict[str, str] = {
    "retail_demand_timesfm_favorita.ipynb": "retail_demand_notebook",
    "electricity_load_timesfm_pjm.ipynb": "electricity_load",
    "manufacturing_sensor_timesfm.ipynb": "manufacturing_sensor",
    "hospital_patient_volume_timesfm.ipynb": "hospital_patient_volume",
    "atm_cash_demand_timesfm.ipynb": "atm_cash_demand",
    "cloud_capacity_timesfm.ipynb": "cloud_capacity",
    "airline_passenger_forecasting_timesfm.ipynb": "airline_demand",
    "warehouse_order_volume_timesfm.ipynb": "warehouse_orders",
    "website_traffic_timesfm.ipynb": "website_traffic",
    "financial_transaction_volume_timesfm.ipynb": "payment_transactions",
}


REQUIRED_ARTIFACTS: dict[str, list[str]] = {
    "retail_demand_notebook": [
        "validation_metrics.csv",
        "inventory_policy.parquet",
        "warehouse_allocation_heuristic.parquet",
        "warehouse_allocation_milp.parquet",
        "submission_timesfm.csv",
        "horizon_7/forecast_h7.parquet",
        "horizon_14/forecast_h14.parquet",
        "horizon_16/forecast_h16.parquet",
        "horizon_30/forecast_h30.parquet",
    ],
    "electricity_load": [
        "backtest_metrics.csv",
        "weekly_operational_schedule.csv",
    ],
    "manufacturing_sensor": [
        "backtest_metrics.csv",
        "future_temperature_forecast.csv",
        "future_temperature_alerts.csv",
        "maintenance_schedule.csv",
    ],
    "hospital_patient_volume": [
        "backtest_metrics.csv",
        "holiday_backtest_eval.csv",
        "forecast_next_90_days.csv",
        "staffing_plan_next_14_days.csv",
        "holiday_staffing_plan.csv",
    ],
    "atm_cash_demand": [
        "backtest_metrics.csv",
        "forecast_next_30_days.csv",
        "tomorrow_cash_requirement.csv",
        "next_week_cash_requirement.csv",
        "holiday_demand_forecast.csv",
        "refill_plan_forecast_policy.csv",
        "logistics_summary.csv",
    ],
    "cloud_capacity": [
        "backtest_metrics.csv",
        "server_forecast_next_24h.csv",
        "cluster_capacity_plan_24h.csv",
        "autoscaling_kpis.csv",
    ],
    "airline_demand": [
        "backtest_metrics.csv",
        "backtest_detail.csv",
        "forecast_next_14_days_route.csv",
        "dynamic_pricing_signals.csv",
        "fleet_allocation_plan.csv",
        "crew_schedule_plan.csv",
        "route_priority_for_planning.csv",
    ],
    "warehouse_orders": [
        "backtest_metrics.csv",
        "backtest_detail.csv",
        "forecast_next_30_days.csv",
        "warehouse_operations_plan.csv",
        "kpi_summary.csv",
    ],
    "website_traffic": [
        "backtest_metrics.csv",
        "backtest_detail.csv",
        "forecast_next_168_hours.csv",
        "autoscaling_plan.csv",
        "infra_kpi_summary.csv",
    ],
    "payment_transactions": [
        "backtest_metrics.csv",
        "backtest_detail.csv",
        "forecast_next_168_hours.csv",
        "capacity_resource_plan.csv",
        "kpi_summary.csv",
    ],
    "retail_demand_script": [
        "validation_metrics.csv",
        "inventory_policy.parquet",
        "warehouse_allocation_heuristic.parquet",
        "warehouse_allocation_milp.parquet",
        "submission_timesfm.csv",
        "horizon_7/forecast_h7.parquet",
        "horizon_14/forecast_h14.parquet",
        "horizon_16/forecast_h16.parquet",
        "horizon_30/forecast_h30.parquet",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final real run outputs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--strict-real-checks", action="store_true", default=True)
    parser.add_argument("--output-report", type=Path, default=None)
    return parser.parse_args()


def count_notebook_errors(path: Path) -> int:
    nb = nbformat.read(path, as_version=4)
    count = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                count += 1
    return count


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def file_rows(path: Path) -> int:
    return parquet_rows(path) if path.suffix == ".parquet" else csv_rows(path)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: dict[str, Any]) -> None:
    checks.append({"name": name, "passed": passed, "details": details})


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    run_artifacts_root = project_root / "artifacts" / "final_real_runs" / args.run_id
    run_notebooks_root = project_root / "notebooks" / "final_real_runs" / args.run_id
    executed_dir = run_notebooks_root / "executed"

    report_path = args.output_report or (run_artifacts_root / "final_validation_report.json")
    run_report_path = run_artifacts_root / "final_run_report.json"

    checks: list[dict[str, Any]] = []

    add_check(
        checks,
        "run_report_exists",
        run_report_path.exists(),
        {"path": str(run_report_path)},
    )

    run_report: dict[str, Any] | None = None
    if run_report_path.exists():
        run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
        summary_status = run_report.get("summary", {}).get("status")
        add_check(
            checks,
            "run_report_summary_pass",
            summary_status == "PASS",
            {"status": summary_status},
        )

    source_notebooks = sorted(
        p
        for p in (project_root / "notebooks").glob("*.ipynb")
        if not p.name.endswith(".executed.ipynb")
    )

    for notebook in source_notebooks:
        executed_name = f"{notebook.stem}.executed.ipynb"
        executed_path = executed_dir / executed_name
        exists = executed_path.exists()
        add_check(
            checks,
            f"executed_notebook_exists::{notebook.name}",
            exists,
            {"path": str(executed_path)},
        )
        if exists:
            err_count = count_notebook_errors(executed_path)
            add_check(
                checks,
                f"executed_notebook_no_errors::{notebook.name}",
                err_count == 0,
                {"error_count": err_count, "path": str(executed_path)},
            )

    for notebook_name, slug in NOTEBOOK_SLUGS.items():
        artifact_dir = run_artifacts_root / slug
        add_check(
            checks,
            f"artifact_dir_exists::{slug}",
            artifact_dir.exists(),
            {"path": str(artifact_dir)},
        )

        for rel in REQUIRED_ARTIFACTS[slug]:
            p = artifact_dir / rel
            exists = p.exists()
            size_ok = exists and p.stat().st_size > 0
            add_check(
                checks,
                f"artifact_exists::{slug}::{rel}",
                exists and size_ok,
                {"path": str(p), "exists": exists, "size": p.stat().st_size if exists else 0},
            )

    # Script output checks
    script_slug = "retail_demand_script"
    script_dir = run_artifacts_root / script_slug
    add_check(
        checks,
        f"artifact_dir_exists::{script_slug}",
        script_dir.exists(),
        {"path": str(script_dir)},
    )
    for rel in REQUIRED_ARTIFACTS[script_slug]:
        p = script_dir / rel
        exists = p.exists()
        size_ok = exists and p.stat().st_size > 0
        add_check(
            checks,
            f"artifact_exists::{script_slug}::{rel}",
            exists and size_ok,
            {"path": str(p), "exists": exists, "size": p.stat().st_size if exists else 0},
        )

    if args.strict_real_checks:
        favorita_train = project_root / "data" / "favorita_real" / "raw" / "train.csv"
        train_ok = favorita_train.exists() and favorita_train.stat().st_size > 1_000_000_000
        add_check(
            checks,
            "strict_real::favorita_train_size_gt_1gb",
            train_ok,
            {
                "path": str(favorita_train),
                "exists": favorita_train.exists(),
                "size": favorita_train.stat().st_size if favorita_train.exists() else 0,
            },
        )

        retail_nb_h30 = run_artifacts_root / "retail_demand_notebook" / "horizon_30" / "forecast_h30.parquet"
        if retail_nb_h30.exists():
            nb_rows = parquet_rows(retail_nb_h30)
            add_check(
                checks,
                "strict_real::retail_notebook_h30_rows_ge_100000",
                nb_rows >= 100_000,
                {"path": str(retail_nb_h30), "rows": nb_rows},
            )

        retail_script_h30 = run_artifacts_root / "retail_demand_script" / "horizon_30" / "forecast_h30.parquet"
        if retail_script_h30.exists():
            script_rows = parquet_rows(retail_script_h30)
            add_check(
                checks,
                "strict_real::retail_script_h30_rows_ge_100000",
                script_rows >= 100_000,
                {"path": str(retail_script_h30), "rows": script_rows},
            )

    passed = sum(1 for c in checks if c["passed"])
    failed = sum(1 for c in checks if not c["passed"])
    status = "PASS" if failed == 0 else "FAIL"

    final_report = {
        "run_id": args.run_id,
        "project_root": str(project_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_real_checks": args.strict_real_checks,
        "summary": {
            "status": status,
            "checks_total": len(checks),
            "checks_passed": passed,
            "checks_failed": failed,
        },
        "checks": checks,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")

    print(f"[validation] report: {report_path}")
    print(f"[validation] status={status} passed={passed} failed={failed}")

    if status != "PASS":
        failed_names = [c["name"] for c in checks if not c["passed"]]
        print("[validation] failed checks:")
        for name in failed_names:
            print(f" - {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

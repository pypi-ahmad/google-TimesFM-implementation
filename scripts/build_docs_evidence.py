from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DomainSpec:
    slug: str
    title: str
    metrics_file: str
    primary_metric: str
    higher_is_better: bool = False


DOMAINS: tuple[DomainSpec, ...] = (
    DomainSpec("airline_demand", "Airline Passenger Forecasting", "backtest_metrics.csv", "mae"),
    DomainSpec("atm_cash_demand", "ATM Cash Demand Forecasting", "backtest_metrics.csv", "mae"),
    DomainSpec("cloud_capacity", "Cloud Infrastructure Capacity Forecasting", "backtest_metrics.csv", "mae"),
    DomainSpec("electricity_load", "Electricity Load Forecasting", "backtest_metrics.csv", "mae"),
    DomainSpec("hospital_patient_volume", "Hospital Patient Volume Forecasting", "backtest_metrics.csv", "mae"),
    DomainSpec("manufacturing_sensor", "Manufacturing Machine Sensor Forecasting", "backtest_metrics.csv", "mae"),
    DomainSpec("payment_transactions", "Financial Transaction Volume Forecasting", "backtest_metrics.csv", "mae"),
    DomainSpec("warehouse_orders", "Warehouse Order Volume Forecasting", "backtest_metrics.csv", "mae"),
    DomainSpec("website_traffic", "Website Traffic Forecasting", "backtest_metrics.csv", "mae"),
    DomainSpec("retail_demand_notebook", "Retail Demand Forecasting (Notebook)", "validation_metrics.csv", "nwrmsle"),
    DomainSpec("retail_demand_script", "Retail Demand Forecasting (Script)", "validation_metrics.csv", "nwrmsle"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build traceable evidence files for documentation.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to docs/evidence/<run-id>",
    )
    return parser.parse_args()


def select_best(model_metrics: pd.Series, higher_is_better: bool) -> tuple[str, float]:
    if higher_is_better:
        idx = model_metrics.idxmax()
    else:
        idx = model_metrics.idxmin()
    return str(idx), float(model_metrics.loc[idx])


def load_model_summary(path: Path, metric_col: str, higher_is_better: bool) -> dict[str, Any]:
    df = pd.read_csv(path)
    if "model" not in df.columns:
        raise ValueError(f"Expected 'model' column in {path}")
    if metric_col not in df.columns:
        raise ValueError(f"Expected '{metric_col}' column in {path}")

    model_metrics = df.groupby("model", dropna=False)[metric_col].mean(numeric_only=True)
    best_model, best_value = select_best(model_metrics, higher_is_better)

    model_label = "timesfm" if "timesfm" in model_metrics.index else "tfm" if "tfm" in model_metrics.index else None
    timesfm_value = float(model_metrics.loc[model_label]) if model_label else None
    baselines = model_metrics.drop(labels=["timesfm", "tfm"], errors="ignore")
    baseline_model, baseline_value = (None, None)
    if not baselines.empty:
        baseline_model, baseline_value = select_best(baselines, higher_is_better)

    delta_vs_best_baseline = None
    if timesfm_value is not None and baseline_value is not None:
        if higher_is_better:
            delta_vs_best_baseline = timesfm_value - baseline_value
        else:
            delta_vs_best_baseline = timesfm_value - baseline_value

    return {
        "rows": int(len(df)),
        "metric": metric_col,
        "timesfm_value": timesfm_value,
        "timesfm_model_label": model_label,
        "best_baseline_model": baseline_model,
        "best_baseline_value": baseline_value,
        "best_model": best_model,
        "best_value": best_value,
        "delta_timesfm_minus_best_baseline": delta_vs_best_baseline,
        "source_file": str(path),
    }


def build_summary(project_root: Path, run_id: str) -> dict[str, Any]:
    run_root = project_root / "artifacts" / "final_real_runs" / run_id
    run_report_path = run_root / "final_run_report.json"
    validation_report_path = run_root / "final_validation_report.json"

    if not run_report_path.exists():
        raise FileNotFoundError(f"Missing run report: {run_report_path}")
    if not validation_report_path.exists():
        raise FileNotFoundError(f"Missing validation report: {validation_report_path}")

    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))

    domain_summaries: list[dict[str, Any]] = []
    for domain in DOMAINS:
        metrics_path = run_root / domain.slug / domain.metrics_file
        if not metrics_path.exists():
            domain_summaries.append(
                {
                    "slug": domain.slug,
                    "title": domain.title,
                    "status": "missing_metrics",
                    "metrics_file": str(metrics_path),
                }
            )
            continue
        model_summary = load_model_summary(
            path=metrics_path,
            metric_col=domain.primary_metric,
            higher_is_better=domain.higher_is_better,
        )
        domain_summaries.append(
            {
                "slug": domain.slug,
                "title": domain.title,
                "status": "ok",
                **model_summary,
            }
        )

    command_runs = []
    for result in run_report.get("results", []):
        command_runs.append(
            {
                "name": result.get("name"),
                "type": result.get("type"),
                "status": result.get("status"),
                "returncode": result.get("returncode"),
                "duration_s": result.get("duration_s"),
                "reuse_mode": bool(result.get("reuse_mode", False)),
                "command": " ".join(result.get("command", [])),
                "log_path": result.get("log_path"),
            }
        )

    return {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_report_path": str(run_report_path),
        "validation_report_path": str(validation_report_path),
        "run_status": run_report.get("summary", {}).get("status"),
        "run_success": run_report.get("summary", {}).get("success"),
        "run_total": run_report.get("summary", {}).get("total"),
        "validation_status": validation_report.get("summary", {}).get("status"),
        "validation_checks_passed": validation_report.get("summary", {}).get("checks_passed"),
        "validation_checks_total": validation_report.get("summary", {}).get("checks_total"),
        "domain_summaries": domain_summaries,
        "commands": command_runs,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Evidence Summary: {summary['run_id']}")
    lines.append("")
    lines.append("## Run Health")
    lines.append("")
    lines.append(f"- Run status: `{summary['run_status']}` ({summary['run_success']}/{summary['run_total']})")
    lines.append(
        f"- Validation status: `{summary['validation_status']}` ({summary['validation_checks_passed']}/{summary['validation_checks_total']})"
    )
    lines.append(f"- Run report: `{summary['run_report_path']}`")
    lines.append(f"- Validation report: `{summary['validation_report_path']}`")
    lines.append("")
    lines.append("## Domain Metrics Snapshot")
    lines.append("")
    lines.append(
        "| Domain | Metric | TimesFM | Best Baseline | Delta (TimesFM - Baseline) | Best Model Overall |"
    )
    lines.append("| --- | --- | ---: | --- | ---: | --- |")
    for item in summary["domain_summaries"]:
        if item.get("status") != "ok":
            lines.append(
                f"| {item['title']} | n/a | n/a | n/a | n/a | missing (`{item['metrics_file']}`) |"
            )
            continue
        timesfm = item["timesfm_value"]
        baseline_name = item["best_baseline_model"] or "n/a"
        baseline_val = item["best_baseline_value"]
        baseline_text = f"{baseline_name}: {baseline_val:.6f}" if baseline_val is not None else "n/a"
        delta = item["delta_timesfm_minus_best_baseline"]
        delta_text = f"{delta:.6f}" if delta is not None else "n/a"
        best_model_text = f"{item['best_model']}: {item['best_value']:.6f}"
        timesfm_text = f"{timesfm:.6f}" if timesfm is not None else "n/a"
        lines.append(
            f"| {item['title']} | {item['metric']} | {timesfm_text} | {baseline_text} | {delta_text} | {best_model_text} |"
        )
    lines.append("")
    lines.append("## Execution Commands")
    lines.append("")
    lines.append("| Target | Type | Status | Duration (s) | Reuse Mode | Command |")
    lines.append("| --- | --- | --- | ---: | --- | --- |")
    for cmd in summary["commands"]:
        lines.append(
            f"| {cmd['name']} | {cmd['type']} | {cmd['status']} | {cmd['duration_s']} | {cmd['reuse_mode']} | `{cmd['command']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (project_root / "docs" / "evidence" / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(project_root=project_root, run_id=args.run_id)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")

    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()

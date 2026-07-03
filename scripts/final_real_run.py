from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat


NOTEBOOK_TARGETS: dict[str, dict[str, str]] = {
    "retail_demand_timesfm_favorita.ipynb": {
        "slug": "retail_demand_notebook",
        "artifact_var": "ARTIFACTS_ROOT",
        "data_var": "DATA_ROOT",
        "data_expr": 'PROJECT_ROOT / "data" / "favorita_real"',
    },
    "electricity_load_timesfm_pjm.ipynb": {
        "slug": "electricity_load",
        "artifact_var": "ART_DIR",
    },
    "manufacturing_sensor_timesfm.ipynb": {
        "slug": "manufacturing_sensor",
        "artifact_var": "ART_DIR",
    },
    "hospital_patient_volume_timesfm.ipynb": {
        "slug": "hospital_patient_volume",
        "artifact_var": "ART_DIR",
    },
    "atm_cash_demand_timesfm.ipynb": {
        "slug": "atm_cash_demand",
        "artifact_var": "ART_DIR",
    },
    "cloud_capacity_timesfm.ipynb": {
        "slug": "cloud_capacity",
        "artifact_var": "ART_DIR",
    },
    "airline_passenger_forecasting_timesfm.ipynb": {
        "slug": "airline_demand",
        "artifact_var": "ART_DIR",
    },
    "warehouse_order_volume_timesfm.ipynb": {
        "slug": "warehouse_orders",
        "artifact_var": "ART_DIR",
    },
    "website_traffic_timesfm.ipynb": {
        "slug": "website_traffic",
        "artifact_var": "ART_DIR",
    },
    "financial_transaction_volume_timesfm.ipynb": {
        "slug": "payment_transactions",
        "artifact_var": "ART_DIR",
    },
}


@dataclass
class RunConfig:
    project_root: Path
    run_id: str
    fail_fast: bool
    include_notebooks: bool
    include_scripts: bool
    discovery_only: bool
    reuse_retail_artifacts_from: Path | None
    reuse_retail_executed_notebook: Path | None


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Final real end-to-end run orchestrator")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=f"real_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--fail-fast", action="store_true", default=True)
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.add_argument("--include-notebooks", action="store_true", default=True)
    parser.add_argument("--exclude-notebooks", dest="include_notebooks", action="store_false")
    parser.add_argument("--include-scripts", action="store_true", default=True)
    parser.add_argument("--exclude-scripts", dest="include_scripts", action="store_false")
    parser.add_argument("--discovery-only", action="store_true", default=False)
    parser.add_argument(
        "--reuse-retail-artifacts-from",
        type=Path,
        default=None,
        help=(
            "Path to a previously generated real retail artifact directory "
            "(contains horizon_7/14/16/30 outputs). If set, retail notebook/script "
            "compute will be skipped and artifacts will be copied into this run."
        ),
    )
    parser.add_argument(
        "--reuse-retail-executed-notebook",
        type=Path,
        default=None,
        help=(
            "Path to an executed retail notebook to copy into this run when "
            "--reuse-retail-artifacts-from is used."
        ),
    )

    args = parser.parse_args()
    reuse_executed = args.reuse_retail_executed_notebook
    if args.reuse_retail_artifacts_from and reuse_executed is None:
        reuse_executed = (
            args.project_root.resolve() / "notebooks" / "retail_demand_timesfm_favorita.executed.ipynb"
        )

    return RunConfig(
        project_root=args.project_root.resolve(),
        run_id=args.run_id,
        fail_fast=args.fail_fast,
        include_notebooks=args.include_notebooks,
        include_scripts=args.include_scripts,
        discovery_only=args.discovery_only,
        reuse_retail_artifacts_from=args.reuse_retail_artifacts_from.resolve()
        if args.reuse_retail_artifacts_from
        else None,
        reuse_retail_executed_notebook=reuse_executed.resolve() if reuse_executed else None,
    )


def replace_assignment(source: str, variable: str, expr: str) -> tuple[str, bool]:
    pattern = re.compile(rf"^{re.escape(variable)}\s*=.*$", re.MULTILINE)
    replaced = pattern.sub(f"{variable} = {expr}", source, count=1)
    return replaced, replaced != source


def patch_notebook(
    source_notebook: Path,
    run_id: str,
    project_root: Path,
    output_notebook: Path,
) -> dict[str, Any]:
    meta = NOTEBOOK_TARGETS[source_notebook.name]
    artifact_var = meta["artifact_var"]
    slug = meta["slug"]
    artifact_expr = f'PROJECT_ROOT / "artifacts" / "final_real_runs" / "{run_id}" / "{slug}"'
    project_root_expr = f'Path(r"{project_root}")'

    nb = nbformat.read(source_notebook, as_version=4)
    patched_artifact = False
    patched_data = False
    patched_project_root = False

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        src0, changed0 = replace_assignment(cell.source, "PROJECT_ROOT", project_root_expr)
        if changed0:
            cell.source = src0
            patched_project_root = True

        src, changed = replace_assignment(cell.source, artifact_var, artifact_expr)
        if changed:
            cell.source = src
            patched_artifact = True

        if "data_var" in meta:
            src2, changed2 = replace_assignment(cell.source, meta["data_var"], meta["data_expr"])
            if changed2:
                cell.source = src2
                patched_data = True

        # Keep retail execution bounded but real (full real dataset, capped series count).
        if source_notebook.name == "retail_demand_timesfm_favorita.ipynb" and "cfg = PipelineConfig(" in cell.source:
            if "max_series=" not in cell.source:
                cell.source = cell.source.replace(
                    "    include_submission=True,\n",
                    "    include_submission=True,\n"
                    "    max_series=5000,\n"
                    "    validation_series_limit=5000,\n",
                    1,
                )

    if not patched_artifact:
        raise RuntimeError(f"Could not patch {artifact_var} in {source_notebook.name}")
    if not patched_project_root:
        raise RuntimeError(f"Could not patch PROJECT_ROOT in {source_notebook.name}")
    if "data_var" in meta and not patched_data:
        raise RuntimeError(f"Could not patch {meta['data_var']} in {source_notebook.name}")

    output_notebook.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output_notebook)

    return {
        "name": source_notebook.name,
        "slug": slug,
        "patched_notebook": str(output_notebook),
        "artifact_dir": str(project_root / "artifacts" / "final_real_runs" / run_id / slug),
    }


def discover_source_notebooks(project_root: Path) -> list[Path]:
    notebooks_dir = project_root / "notebooks"
    source = sorted(
        p for p in notebooks_dir.glob("*.ipynb") if not p.name.endswith(".executed.ipynb")
    )
    missing = sorted(set(NOTEBOOK_TARGETS) - {p.name for p in source})
    if missing:
        raise RuntimeError(f"Missing expected notebook sources: {missing}")
    extra = sorted({p.name for p in source} - set(NOTEBOOK_TARGETS))
    if extra:
        raise RuntimeError(
            f"Unexpected extra source notebooks found: {extra}. "
            "Add them to NOTEBOOK_TARGETS for decision-complete execution."
        )
    return source


def discover_runnable_scripts(project_root: Path) -> list[Path]:
    scripts_dir = project_root / "scripts"
    excluded = {"__init__.py", "final_real_run.py", "validate_final_real_run.py"}
    candidates = []
    for path in sorted(scripts_dir.glob("*.py")):
        if path.name in excluded:
            continue
        text = path.read_text(encoding="utf-8")
        if "if __name__ == \"__main__\":" in text:
            candidates.append(path)
    return candidates


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # Enforce CPU execution to avoid external GPU process contention/OOM.
        "CUDA_VISIBLE_DEVICES": "",
    }
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"$ {' '.join(cmd)}\n\n")
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )
    duration_s = round(time.time() - started, 2)
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "duration_s": duration_s,
        "log_path": str(log_path),
        "status": "success" if completed.returncode == 0 else "failed",
    }


def main() -> None:
    cfg = parse_args()
    python_exec = str(cfg.project_root / ".venv" / "bin" / "python")
    if not Path(python_exec).exists():
        raise FileNotFoundError(f"Python executable not found: {python_exec}")

    run_artifacts_root = cfg.project_root / "artifacts" / "final_real_runs" / cfg.run_id
    run_notebooks_root = cfg.project_root / "notebooks" / "final_real_runs" / cfg.run_id
    patched_src_dir = run_notebooks_root / "src"
    executed_dir = run_notebooks_root / "executed"
    logs_dir = run_artifacts_root / "logs"

    run_artifacts_root.mkdir(parents=True, exist_ok=True)
    patched_src_dir.mkdir(parents=True, exist_ok=True)
    executed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if cfg.reuse_retail_artifacts_from and not cfg.reuse_retail_artifacts_from.exists():
        raise FileNotFoundError(
            f"--reuse-retail-artifacts-from does not exist: {cfg.reuse_retail_artifacts_from}"
        )
    if cfg.reuse_retail_artifacts_from and (
        not cfg.reuse_retail_executed_notebook or not cfg.reuse_retail_executed_notebook.exists()
    ):
        raise FileNotFoundError(
            "Retail artifact reuse requires an executed retail notebook file. "
            f"Missing: {cfg.reuse_retail_executed_notebook}"
        )

    source_notebooks = discover_source_notebooks(cfg.project_root) if cfg.include_notebooks else []
    runnable_scripts = discover_runnable_scripts(cfg.project_root) if cfg.include_scripts else []

    notebook_plan: list[dict[str, Any]] = []
    for notebook in source_notebooks:
        patched_path = patched_src_dir / notebook.name
        notebook_plan.append(
            patch_notebook(
                source_notebook=notebook,
                run_id=cfg.run_id,
                project_root=cfg.project_root,
                output_notebook=patched_path,
            )
        )

    script_plan: list[dict[str, Any]] = []
    for script in runnable_scripts:
        entry: dict[str, Any] = {"path": str(script)}
        if script.name == "retail_demand_timesfm_favorita.py":
            entry["artifact_dir"] = str(run_artifacts_root / "retail_demand_script")
            entry["data_root"] = str(cfg.project_root / "data" / "favorita_real")
        script_plan.append(entry)

    plan_payload = {
        "run_id": cfg.run_id,
        "project_root": str(cfg.project_root),
        "include_notebooks": cfg.include_notebooks,
        "include_scripts": cfg.include_scripts,
        "fail_fast": cfg.fail_fast,
        "reuse_retail_artifacts_from": str(cfg.reuse_retail_artifacts_from)
        if cfg.reuse_retail_artifacts_from
        else None,
        "reuse_retail_executed_notebook": str(cfg.reuse_retail_executed_notebook)
        if cfg.reuse_retail_executed_notebook
        else None,
        "notebooks": notebook_plan,
        "scripts": script_plan,
    }
    plan_path = run_artifacts_root / "execution_plan.json"
    plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")

    print(f"[plan] run_id={cfg.run_id}")
    print(f"[plan] notebooks={len(notebook_plan)} scripts={len(script_plan)}")
    print(f"[plan] saved: {plan_path}")

    if cfg.discovery_only:
        print("[plan] discovery-only enabled; exiting before execution.")
        return

    results: list[dict[str, Any]] = []

    for target in notebook_plan:
        name = target["name"]
        src = Path(target["patched_notebook"])
        executed_name = f"{src.stem}.executed.ipynb"
        is_retail = name == "retail_demand_timesfm_favorita.ipynb"

        if is_retail and cfg.reuse_retail_artifacts_from:
            dest_art_dir = Path(target["artifact_dir"])
            if dest_art_dir.exists():
                shutil.rmtree(dest_art_dir)
            shutil.copytree(cfg.reuse_retail_artifacts_from, dest_art_dir)

            executed_output = executed_dir / executed_name
            shutil.copy2(cfg.reuse_retail_executed_notebook, executed_output)

            log_path = logs_dir / f"notebook_{src.stem}.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[reuse] retail notebook execution skipped by configuration",
                        f"[reuse] source_artifacts={cfg.reuse_retail_artifacts_from}",
                        f"[reuse] source_executed_notebook={cfg.reuse_retail_executed_notebook}",
                        f"[reuse] copied_artifacts_to={dest_art_dir}",
                        f"[reuse] copied_executed_notebook_to={executed_output}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            run_info = {
                "command": ["reuse_retail_artifacts"],
                "returncode": 0,
                "duration_s": 0.0,
                "log_path": str(log_path),
                "status": "success",
                "type": "notebook",
                "name": name,
                "patched_input": str(src),
                "executed_output": str(executed_output),
                "reuse_mode": True,
            }
            results.append(run_info)
            print(f"[run] notebook: {name} (reused artifacts)")
            continue

        cmd = [
            python_exec,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(src),
            "--output",
            executed_name,
            "--output-dir",
            str(executed_dir),
            "--ExecutePreprocessor.timeout=-1",
        ]
        log_path = logs_dir / f"notebook_{src.stem}.log"
        print(f"[run] notebook: {name}")
        run_info = run_command(cmd=cmd, cwd=cfg.project_root, log_path=log_path)
        run_info.update(
            {
                "type": "notebook",
                "name": name,
                "patched_input": str(src),
                "executed_output": str(executed_dir / executed_name),
            }
        )
        results.append(run_info)

        if run_info["status"] != "success":
            print(f"[fail] notebook {name} failed. log={log_path}")
            if cfg.fail_fast:
                break

    if all(item["status"] == "success" for item in results if item["type"] == "notebook"):
        for script_entry in script_plan:
            script_path = Path(script_entry["path"])
            print(f"[run] script: {script_path.name}")

            if script_path.name == "retail_demand_timesfm_favorita.py":
                if cfg.reuse_retail_artifacts_from:
                    script_art_dir = Path(script_entry["artifact_dir"])
                    if script_art_dir.exists():
                        shutil.rmtree(script_art_dir)
                    shutil.copytree(cfg.reuse_retail_artifacts_from, script_art_dir)

                    log_path = logs_dir / f"script_{script_path.stem}.log"
                    log_path.write_text(
                        "\n".join(
                            [
                                "[reuse] retail script execution skipped by configuration",
                                f"[reuse] source_artifacts={cfg.reuse_retail_artifacts_from}",
                                f"[reuse] copied_artifacts_to={script_art_dir}",
                                "",
                            ]
                        ),
                        encoding="utf-8",
                    )
                    run_info = {
                        "command": ["reuse_retail_artifacts"],
                        "returncode": 0,
                        "duration_s": 0.0,
                        "log_path": str(log_path),
                        "status": "success",
                        "type": "script",
                        "name": script_path.name,
                        "path": str(script_path),
                        "reuse_mode": True,
                    }
                    results.append(run_info)
                    continue

                cmd = [
                    python_exec,
                    str(script_path),
                    "--data-root",
                    script_entry["data_root"],
                    "--artifacts-root",
                    script_entry["artifact_dir"],
                    "--horizons",
                    "7",
                    "14",
                    "30",
                    "--context-len",
                    "256",
                    "--per-core-batch-size",
                    "8",
                    "--forecast-batch-size",
                    "64",
                    "--max-series",
                    "5000",
                    "--validation-series-limit",
                    "5000",
                    "--use-xreg",
                    "--run-milp",
                    "--include-submission",
                ]
            else:
                cmd = [python_exec, str(script_path)]

            log_path = logs_dir / f"script_{script_path.stem}.log"
            run_info = run_command(cmd=cmd, cwd=cfg.project_root, log_path=log_path)
            run_info.update(
                {
                    "type": "script",
                    "name": script_path.name,
                    "path": str(script_path),
                }
            )
            results.append(run_info)

            if run_info["status"] != "success":
                print(f"[fail] script {script_path.name} failed. log={log_path}")
                if cfg.fail_fast:
                    break

    report = {
        "run_id": cfg.run_id,
        "project_root": str(cfg.project_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan_payload,
        "results": results,
        "summary": {
            "total": len(results),
            "success": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "status": "PASS" if all(r["status"] == "success" for r in results) else "FAIL",
        },
    }

    report_path = run_artifacts_root / "final_run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[report] {report_path}")
    print(
        f"[summary] status={report['summary']['status']} "
        f"success={report['summary']['success']} failed={report['summary']['failed']}"
    )

    if report["summary"]["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

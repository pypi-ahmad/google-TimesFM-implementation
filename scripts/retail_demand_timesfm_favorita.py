from __future__ import annotations

import argparse
import math
import os
import random
import shlex
import subprocess
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
from loguru import logger
from ortools.linear_solver import pywraplp
from pydantic import BaseModel, Field, field_validator
from tqdm.auto import tqdm


COMPETITION_NAME = "favorita-grocery-sales-forecasting"
RAW_ARCHIVES = (
    "train.csv.7z",
    "test.csv.7z",
    "items.csv.7z",
    "stores.csv.7z",
    "oil.csv.7z",
    "holidays_events.csv.7z",
    "transactions.csv.7z",
    "sample_submission.csv.7z",
)


class PipelineConfig(BaseModel):
    """Runtime config for the Favorita + TimesFM pipeline.

    Attributes:
        data_root: Root directory for raw/interim/processed data.
        artifacts_root: Root path for model/decision outputs.
        horizons: Business forecast horizons to produce.
        context_len: TimesFM context length.
        per_core_batch_size: Per device TimesFM batch size.
        forecast_batch_size: Series batch size for pipeline iteration.
        seed: Global deterministic seed.
        device: `auto`, `cuda`, or `cpu`.
        use_xreg: Whether to run TimesFM covariate mode.
        xreg_mode: `xreg + timesfm` or `timesfm + xreg`.
        xreg_ridge: Ridge regularization for XReg.
        run_milp: Whether to run OR-Tools MILP allocation.
        n_warehouses: Number of virtual warehouses for allocation.
        warehouse_capacity_factor: Capacity multiplier over assigned demand.
        service_level: Service level used in inventory policy.
        lead_time_days: Lead time in days for reorder policy.
        max_series: Optional cap for faster debugging.
        validation_series_limit: Optional cap for backtest series.
        include_submission: Whether to build Kaggle-format submission.
    """

    data_root: Path = Path("data")
    artifacts_root: Path = Path("artifacts/retail_demand_timesfm")
    horizons: tuple[int, ...] = (7, 14, 30)
    context_len: int = 1024
    per_core_batch_size: int = 8
    forecast_batch_size: int = 64
    seed: int = 42
    device: str = "auto"
    use_xreg: bool = True
    xreg_mode: str = "xreg + timesfm"
    xreg_ridge: float = 1e-3
    run_milp: bool = True
    n_warehouses: int = 5
    warehouse_capacity_factor: float = 1.10
    service_level: float = 0.95
    lead_time_days: int = 7
    max_series: int | None = None
    validation_series_limit: int | None = 50_000
    include_submission: bool = True

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("At least one horizon is required.")
        if any(v <= 0 for v in values):
            raise ValueError("Horizons must be positive integers.")
        return tuple(sorted(set(values)))

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        allowed = {"auto", "cpu", "cuda"}
        if value not in allowed:
            raise ValueError(f"device must be one of {allowed}.")
        return value


def parse_args() -> PipelineConfig:
    """Parses CLI args into `PipelineConfig`."""
    parser = argparse.ArgumentParser(
        description="Retail demand forecasting with Google TimesFM 2.5 on Favorita"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/retail_demand_timesfm"),
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 30])
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--per-core-batch-size", type=int, default=8)
    parser.add_argument("--forecast-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--use-xreg", action="store_true")
    parser.add_argument("--xreg-mode", default="xreg + timesfm")
    parser.add_argument("--xreg-ridge", type=float, default=1e-3)
    parser.add_argument("--run-milp", action="store_true")
    parser.add_argument("--n-warehouses", type=int, default=5)
    parser.add_argument("--warehouse-capacity-factor", type=float, default=1.10)
    parser.add_argument("--service-level", type=float, default=0.95)
    parser.add_argument("--lead-time-days", type=int, default=7)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--validation-series-limit", type=int, default=50_000)
    parser.add_argument("--include-submission", action="store_true")

    args = parser.parse_args()
    return PipelineConfig(
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        horizons=tuple(args.horizons),
        context_len=args.context_len,
        per_core_batch_size=args.per_core_batch_size,
        forecast_batch_size=args.forecast_batch_size,
        seed=args.seed,
        device=args.device,
        use_xreg=args.use_xreg,
        xreg_mode=args.xreg_mode,
        xreg_ridge=args.xreg_ridge,
        run_milp=args.run_milp,
        n_warehouses=args.n_warehouses,
        warehouse_capacity_factor=args.warehouse_capacity_factor,
        service_level=args.service_level,
        lead_time_days=args.lead_time_days,
        max_series=args.max_series,
        validation_series_limit=args.validation_series_limit,
        include_submission=args.include_submission,
    )


def run_command(cmd: Sequence[str], cwd: Path | None = None) -> None:
    """Executes a shell command and fails with full stderr on non-zero exit."""
    logger.info("$ {}", shlex.join(cmd))
    completed = subprocess.run(
        list(cmd),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if "403" in stderr or "Forbidden" in stderr:
            raise RuntimeError(
                "Kaggle download failed with 403 Forbidden. "
                "Accept competition rules on Kaggle and confirm API auth."
            )
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {shlex.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{stderr}"
        )


def set_global_seeds(seed: int) -> None:
    """Sets deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)



def ensure_paths(cfg: PipelineConfig) -> dict[str, Path]:
    """Creates and returns core working directories."""
    raw_dir = cfg.data_root / "raw"
    interim_dir = cfg.data_root / "interim"
    processed_dir = cfg.data_root / "processed"
    artifacts_dir = cfg.artifacts_root

    for path in (raw_dir, interim_dir, processed_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "raw": raw_dir,
        "interim": interim_dir,
        "processed": processed_dir,
        "artifacts": artifacts_dir,
    }


def download_favorita_data(raw_dir: Path) -> None:
    """Downloads and extracts all competition files.

    Example:
        >>> download_favorita_data(Path("data/raw"))
    """
    for archive_name in RAW_ARCHIVES:
        csv_name = archive_name.replace(".7z", "")
        csv_path = raw_dir / csv_name
        archive_path = raw_dir / archive_name
        if csv_path.exists():
            logger.info("Found {}, skipping download.", csv_path.name)
            continue
        if not archive_path.exists():
            run_command(
                [
                    "kaggle",
                    "competitions",
                    "download",
                    "-c",
                    COMPETITION_NAME,
                    "-f",
                    archive_name,
                    "-p",
                    str(raw_dir),
                    "--force",
                ]
            )
        run_command(["7z", "e", "-y", str(archive_path), f"-o{raw_dir}"])


def prepare_series_panel(raw_dir: Path, processed_dir: Path, max_series: int | None) -> tuple[Path, Path]:
    """Builds train-long and panel-by-series parquet datasets.

    The panel dataset contains one row per `(store_nbr, item_nbr)` and list
    columns `date`, `unit_sales`, and `onpromotion`.
    """
    train_long_path = processed_dir / "train_long_filtered.parquet"
    panel_path = processed_dir / "series_panel.parquet"

    raw_max_date = (
        pl.scan_csv(raw_dir / "train.csv")
        .select(pl.col("date").max().alias("max_date"))
        .collect()
        .item()
    )

    train_long_is_valid = False
    if train_long_path.exists():
        existing_stats = (
            pl.scan_parquet(train_long_path)
            .select(pl.col("date").max().alias("max_date"), pl.len().alias("rows"))
            .collect(engine="streaming")
        )
        existing_max_date = str(existing_stats["max_date"][0])
        existing_rows = int(existing_stats["rows"][0])
        if existing_max_date >= str(raw_max_date) and existing_rows > 0:
            train_long_is_valid = True
            logger.info("Found {} (rows={}, max_date={})", train_long_path, existing_rows, existing_max_date)
        else:
            logger.warning(
                "Detected stale/partial {} (rows={}, max_date={}) vs raw max_date={}; rebuilding.",
                train_long_path,
                existing_rows,
                existing_max_date,
                raw_max_date,
            )
            train_long_path.unlink(missing_ok=True)

    if not train_long_is_valid:
        logger.info("Building filtered train long parquet...")
        test_keys = (
            pl.scan_csv(raw_dir / "test.csv")
            .select(["store_nbr", "item_nbr"])
            .unique()
            .collect(engine="streaming")
        )
        train_lf = (
            pl.scan_csv(raw_dir / "train.csv", try_parse_dates=True)
            .select(["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"])
            .with_columns(
                pl.col("unit_sales").cast(pl.Float32).clip(lower_bound=0.0),
                pl.when(pl.col("onpromotion").is_null())
                .then(pl.lit(False))
                .otherwise(
                    pl.col("onpromotion")
                    .cast(pl.Utf8, strict=False)
                    .str.to_lowercase()
                    .is_in(["true", "t", "1"])
                )
                .cast(pl.Boolean)
                .alias("onpromotion"),
            )
            .join(test_keys.lazy(), on=["store_nbr", "item_nbr"], how="semi")
        )
        if max_series is not None:
            keep_keys = (
                train_lf.select(["store_nbr", "item_nbr"])
                .unique()
                .limit(max_series)
                .collect(engine="streaming")
            )
            train_lf = train_lf.join(keep_keys.lazy(), on=["store_nbr", "item_nbr"], how="semi")

        tmp_train_long_path = train_long_path.with_suffix(".tmp.parquet")
        tmp_train_long_path.unlink(missing_ok=True)
        train_lf.sink_parquet(tmp_train_long_path)
        tmp_train_long_path.replace(train_long_path)

    panel_is_valid = False
    if panel_path.exists():
        series_in_panel = pl.scan_parquet(panel_path).select(pl.len().alias("n")).collect().item()
        unique_series = (
            pl.scan_parquet(train_long_path)
            .select(["store_nbr", "item_nbr"])
            .unique()
            .select(pl.len().alias("n"))
            .collect(engine="streaming")
            .item()
        )
        if int(series_in_panel) == int(unique_series) and int(series_in_panel) > 0:
            panel_is_valid = True
            logger.info("Found {} (series={})", panel_path, int(series_in_panel))
        else:
            logger.warning(
                "Detected stale/partial {} (series={}) vs expected series={}; rebuilding.",
                panel_path,
                int(series_in_panel),
                int(unique_series),
            )
            panel_path.unlink(missing_ok=True)

    if not panel_is_valid:
        logger.info("Building panel parquet...")
        tmp_panel_path = panel_path.with_suffix(".tmp.parquet")
        tmp_panel_path.unlink(missing_ok=True)
        (
            pl.scan_parquet(train_long_path)
            .sort(["store_nbr", "item_nbr", "date"])
            .group_by(["store_nbr", "item_nbr"])
            .agg(
                pl.col("date"),
                pl.col("unit_sales"),
                pl.col("onpromotion"),
            )
            .sink_parquet(tmp_panel_path)
        )
        tmp_panel_path.replace(panel_path)

    return train_long_path, panel_path


def load_metadata(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads store/item/test metadata frames."""
    items_df = pd.read_csv(raw_dir / "items.csv")
    stores_df = pd.read_csv(raw_dir / "stores.csv")
    test_df = pd.read_csv(
        raw_dir / "test.csv",
        parse_dates=["date"],
        usecols=["id", "date", "store_nbr", "item_nbr", "onpromotion"],
    )
    test_df["onpromotion"] = test_df["onpromotion"].fillna(False).astype(bool)
    return items_df, stores_df, test_df


def get_anchor_date(train_long_path: Path) -> date:
    """Returns the last available train date."""
    max_date = (
        pl.scan_parquet(train_long_path)
        .select(pl.col("date").max().alias("max_date"))
        .collect(engine="streaming")["max_date"][0]
    )
    return pd.Timestamp(max_date).date()


def get_model_and_compile(
    cfg: PipelineConfig,
    max_horizon: int,
    enable_backcast: bool,
) -> Any:
    """Loads and compiles TimesFM 2.5 model.

    Example:
        >>> model = get_model_and_compile(cfg, max_horizon=30, enable_backcast=True)
    """
    if cfg.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import timesfm

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    if cfg.device in {"auto", "cuda"} and torch.cuda.is_available():
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_mem_gb <= 8.5:
            if cfg.per_core_batch_size > 8:
                logger.warning(
                    "Auto-tuning per_core_batch_size from {} to 8 for {:.1f}GB GPU.",
                    cfg.per_core_batch_size,
                    total_mem_gb,
                )
                cfg.per_core_batch_size = 8
        elif total_mem_gb <= 12.5:
            if cfg.per_core_batch_size > 16:
                logger.warning(
                    "Auto-tuning per_core_batch_size from {} to 16 for {:.1f}GB GPU.",
                    cfg.per_core_batch_size,
                    total_mem_gb,
                )
                cfg.per_core_batch_size = 16

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

    fc = timesfm.ForecastConfig(
        max_context=cfg.context_len,
        max_horizon=max_horizon,
        normalize_inputs=True,
        per_core_batch_size=cfg.per_core_batch_size,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
        return_backcast=enable_backcast,
    )
    model.compile(fc)
    logger.info(
        "TimesFM compiled with context={} max_horizon={} xreg={}",
        fc.max_context,
        fc.max_horizon,
        enable_backcast,
    )
    return model


def calendar_features(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Creates deterministic time covariates for XReg."""
    dow = dates.dayofweek.values.astype(np.float32)
    dom = dates.day.values.astype(np.float32)
    month = dates.month.values.astype(np.float32)
    return {
        "dow_sin": np.sin(2 * np.pi * dow / 7.0).astype(np.float32),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0).astype(np.float32),
        "dom_norm": (dom / 31.0).astype(np.float32),
        "month_norm": (month / 12.0).astype(np.float32),
        "is_weekend": (dow >= 5).astype(np.float32),
    }


def build_series_maps(
    dates: Sequence[Any],
    values: Sequence[float],
    promos: Sequence[bool],
) -> tuple[dict[date, float], dict[date, float]]:
    """Builds date-indexed maps for demand and promotion."""
    ts = pd.to_datetime(np.asarray(dates))
    sales_map = {d.date(): float(v) for d, v in zip(ts, values)}
    promo_map = {d.date(): float(bool(p)) for d, p in zip(ts, promos)}
    return sales_map, promo_map


def panel_batches(panel_path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yields panel rows in memory-safe batches."""
    dataset = ds.dataset(panel_path, format="parquet")
    scanner = dataset.scanner(batch_size=batch_size)
    for record_batch in scanner.to_batches():
        yield record_batch.to_pandas()


def build_test_promo_lookup(test_df: pd.DataFrame) -> dict[tuple[int, int, date], float]:
    """Creates promotion lookup for known test horizon dates."""
    mapping: dict[tuple[int, int, date], float] = {}
    for row in test_df.itertuples(index=False):
        mapping[(int(row.store_nbr), int(row.item_nbr), row.date.date())] = float(bool(row.onpromotion))
    return mapping


def build_model_inputs_for_batch(
    batch_df: pd.DataFrame,
    anchor_date: date,
    horizon: int,
    context_len: int,
    test_promo_lookup: dict[tuple[int, int, date], float],
    family_by_item: dict[int, str],
    cluster_by_store: dict[int, int],
) -> tuple[
    list[np.ndarray],
    dict[str, list[list[float]]],
    dict[str, list[list[float]]],
    dict[str, list[Any]],
    pd.DataFrame,
]:
    """Transforms panel rows into TimesFM inputs and covariates."""
    context_start = anchor_date - timedelta(days=context_len - 1)
    context_dates = pd.date_range(context_start, anchor_date, freq="D")
    future_dates = pd.date_range(anchor_date + timedelta(days=1), periods=horizon, freq="D")
    full_dates = context_dates.append(future_dates)
    cal = calendar_features(full_dates)

    inputs: list[np.ndarray] = []
    dyn_num: dict[str, list[list[float]]] = {k: [] for k in cal}
    dyn_cat: dict[str, list[list[float]]] = {"onpromotion": []}
    static_cat: dict[str, list[Any]] = {"family": [], "store_cluster": []}
    series_meta: list[dict[str, Any]] = []

    context_date_keys = [d.date() for d in context_dates]
    full_date_keys = [d.date() for d in full_dates]

    for row in batch_df.itertuples(index=False):
        store_nbr = int(row.store_nbr)
        item_nbr = int(row.item_nbr)
        sales_map, promo_map = build_series_maps(row.date, row.unit_sales, row.onpromotion)

        context_values = np.array([sales_map.get(d, 0.0) for d in context_date_keys], dtype=np.float32)
        context_values = np.nan_to_num(context_values, nan=0.0, posinf=0.0, neginf=0.0)
        inputs.append(context_values)

        promo_full = []
        for d in full_date_keys:
            if d <= anchor_date:
                promo_full.append(promo_map.get(d, 0.0))
            else:
                promo_full.append(test_promo_lookup.get((store_nbr, item_nbr, d), 0.0))
        dyn_cat["onpromotion"].append(promo_full)

        for name, values in cal.items():
            dyn_num[name].append(values.astype(np.float32).tolist())

        static_cat["family"].append(family_by_item.get(item_nbr, "unknown"))
        static_cat["store_cluster"].append(int(cluster_by_store.get(store_nbr, -1)))

        trailing_14 = float(context_values[-14:].mean()) if len(context_values) >= 14 else float(context_values.mean())
        series_meta.append(
            {
                "store_nbr": store_nbr,
                "item_nbr": item_nbr,
                "trailing_mean_14": trailing_14,
                "context_sum_7": float(context_values[-7:].sum()),
            }
        )

    meta_df = pd.DataFrame(series_meta)
    return inputs, dyn_num, dyn_cat, static_cat, meta_df


def to_numpy_forecasts(
    point_forecast: Any,
    quantile_forecast: Any,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalizes forecast outputs from both core and XReg interfaces."""
    if isinstance(point_forecast, np.ndarray):
        point_np = point_forecast[:, :horizon].astype(np.float32)
    else:
        point_np = np.vstack([np.asarray(x, dtype=np.float32)[:horizon] for x in point_forecast])

    if isinstance(quantile_forecast, np.ndarray):
        quant_np = quantile_forecast[:, :horizon, :].astype(np.float32)
    else:
        quant_np = np.stack([np.asarray(x, dtype=np.float32)[:horizon] for x in quantile_forecast])

    return point_np, quant_np


def forecast_horizon(
    model: Any,
    panel_path: Path,
    output_dir: Path,
    anchor_date: date,
    horizon: int,
    cfg: PipelineConfig,
    family_by_item: dict[int, str],
    cluster_by_store: dict[int, int],
    test_promo_lookup: dict[tuple[int, int, date], float],
) -> tuple[Path, Path]:
    """Runs batched TimesFM inference and writes horizon parquet outputs.

    Example:
        >>> forecast_horizon(model, panel_path, out, date(2017, 8, 15), 7, cfg, fam, clu, promo)
    """
    horizon_dir = output_dir / f"horizon_{horizon}"
    parts_dir = horizon_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    stats_parts_dir = horizon_dir / "stats_parts"
    stats_parts_dir.mkdir(parents=True, exist_ok=True)

    forecast_parts: list[Path] = []
    stats_parts: list[Path] = []

    for part_idx, batch_df in enumerate(tqdm(panel_batches(panel_path, cfg.forecast_batch_size), desc=f"H{horizon}")):
        if batch_df.empty:
            continue

        inputs, dyn_num, dyn_cat, static_cat, meta_df = build_model_inputs_for_batch(
            batch_df=batch_df,
            anchor_date=anchor_date,
            horizon=horizon,
            context_len=cfg.context_len,
            test_promo_lookup=test_promo_lookup,
            family_by_item=family_by_item,
            cluster_by_store=cluster_by_store,
        )

        if cfg.use_xreg:
            try:
                point_forecast, quantile_forecast = model.forecast_with_covariates(
                    inputs=inputs,
                    dynamic_numerical_covariates=dyn_num,
                    dynamic_categorical_covariates=dyn_cat,
                    static_categorical_covariates=static_cat,
                    xreg_mode=cfg.xreg_mode,
                    ridge=cfg.xreg_ridge,
                )
            except Exception as exc:
                logger.warning("XReg failed on batch {} ({}), falling back to core forecast.", part_idx, exc)
                point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=inputs)
        else:
            point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=inputs)

        point_np, quant_np = to_numpy_forecasts(point_forecast, quantile_forecast, horizon)

        stores = meta_df["store_nbr"].to_numpy(dtype=np.int32)
        items = meta_df["item_nbr"].to_numpy(dtype=np.int32)
        n_series = len(meta_df)
        horizon_days = np.tile(np.arange(1, horizon + 1, dtype=np.int16), n_series)
        forecast_dates = np.tile(
            np.array([(anchor_date + timedelta(days=i)).isoformat() for i in range(1, horizon + 1)]),
            n_series,
        )

        forecast_df = pl.DataFrame(
            {
                "store_nbr": np.repeat(stores, horizon),
                "item_nbr": np.repeat(items, horizon),
                "horizon_day": horizon_days,
                "forecast_date": forecast_dates,
                "point_forecast": point_np.reshape(-1),
                "q10": quant_np[:, :, 1].reshape(-1),
                "q50": quant_np[:, :, 5].reshape(-1),
                "q90": quant_np[:, :, 9].reshape(-1),
            }
        )
        part_path = parts_dir / f"forecast_part_{part_idx:06d}.parquet"
        forecast_df.write_parquet(part_path)
        forecast_parts.append(part_path)

        stats_path = stats_parts_dir / f"series_stats_{part_idx:06d}.parquet"
        pl.from_pandas(meta_df).write_parquet(stats_path)
        stats_parts.append(stats_path)

    final_forecast_path = horizon_dir / f"forecast_h{horizon}.parquet"
    (
        pl.scan_parquet(str(parts_dir / "*.parquet"))
        .with_columns(
            pl.col("forecast_date").str.to_date(),
            pl.col("point_forecast").clip(lower_bound=0.0),
            pl.col("q10").clip(lower_bound=0.0),
            pl.col("q50").clip(lower_bound=0.0),
            pl.col("q90").clip(lower_bound=0.0),
        )
        .sink_parquet(final_forecast_path)
    )

    final_stats_path = horizon_dir / "series_stats.parquet"
    (
        pl.scan_parquet(str(stats_parts_dir / "*.parquet"))
        .group_by(["store_nbr", "item_nbr"])
        .agg(
            pl.col("trailing_mean_14").mean(),
            pl.col("context_sum_7").mean(),
        )
        .sink_parquet(final_stats_path)
    )

    logger.info("Wrote horizon {} forecast: {}", horizon, final_forecast_path)
    return final_forecast_path, final_stats_path


def materialize_horizon_slice(
    source_forecast_path: Path,
    output_dir: Path,
    source_horizon: int,
    target_horizon: int,
) -> Path:
    """Creates a target-horizon forecast parquet by slicing a larger horizon."""
    if target_horizon == source_horizon:
        return source_forecast_path

    if target_horizon > source_horizon:
        raise ValueError(
            f"Cannot slice horizon {target_horizon} from source horizon {source_horizon}."
        )

    target_dir = output_dir / f"horizon_{target_horizon}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"forecast_h{target_horizon}.parquet"

    if target_path.exists():
        logger.info("Found existing sliced horizon {} forecast: {}", target_horizon, target_path)
        return target_path

    (
        pl.scan_parquet(source_forecast_path)
        .filter(pl.col("horizon_day") <= pl.lit(target_horizon))
        .sink_parquet(target_path)
    )
    logger.info(
        "Materialized horizon {} forecast from horizon {}: {}",
        target_horizon,
        source_horizon,
        target_path,
    )
    return target_path


def collect_actuals_for_horizon(
    train_long_path: Path,
    anchor_date: date,
    horizon: int,
) -> pl.DataFrame:
    """Collects actual future targets for a historical anchor date."""
    start = anchor_date + timedelta(days=1)
    end = anchor_date + timedelta(days=horizon)
    actual_df = (
        pl.scan_parquet(train_long_path)
        .filter((pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end)))
        .with_columns((pl.col("date") - pl.lit(anchor_date)).dt.total_days().cast(pl.Int16).alias("horizon_day"))
        .group_by(["store_nbr", "item_nbr", "horizon_day"])
        .agg(pl.col("unit_sales").sum().alias("actual"))
        .collect(engine="streaming")
    )
    return actual_df


def nwrmsle(pred: np.ndarray, actual: np.ndarray, weight: np.ndarray) -> float:
    """Computes normalized weighted RMSLE."""
    pred = np.clip(pred, 0.0, None)
    actual = np.clip(actual, 0.0, None)
    sq = (np.log1p(pred) - np.log1p(actual)) ** 2
    return float(math.sqrt(np.sum(weight * sq) / np.sum(weight)))


def wmape(pred: np.ndarray, actual: np.ndarray) -> float:
    """Computes weighted MAPE."""
    denom = np.abs(actual).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(pred - actual).sum() / denom)


def evaluate_backtest(
    model: Any,
    panel_path: Path,
    train_long_path: Path,
    artifacts_dir: Path,
    cfg: PipelineConfig,
    family_by_item: dict[int, str],
    cluster_by_store: dict[int, int],
    perishable_weight_by_item: dict[int, float],
) -> Path:
    """Runs one historical backtest and compares TimesFM vs baselines."""
    max_h = max(cfg.horizons)
    train_anchor = get_anchor_date(train_long_path)
    val_anchor = train_anchor - timedelta(days=max_h)

    out_rows: list[dict[str, Any]] = []

    all_actuals = collect_actuals_for_horizon(train_long_path, val_anchor, max_h)
    if all_actuals.is_empty():
        raise RuntimeError("Backtest actuals are empty. Cannot evaluate.")

    all_actuals_pd = all_actuals.to_pandas()

    cap_series = cfg.validation_series_limit
    processed_series = 0

    for batch_df in tqdm(panel_batches(panel_path, cfg.forecast_batch_size), desc="Backtest"):
        if batch_df.empty:
            continue
        if cap_series is not None and processed_series >= cap_series:
            break

        if cap_series is not None:
            remaining = cap_series - processed_series
            batch_df = batch_df.head(max(0, remaining))
            if batch_df.empty:
                break

        inputs, dyn_num, dyn_cat, static_cat, _ = build_model_inputs_for_batch(
            batch_df=batch_df,
            anchor_date=val_anchor,
            horizon=max_h,
            context_len=cfg.context_len,
            test_promo_lookup={},
            family_by_item=family_by_item,
            cluster_by_store=cluster_by_store,
        )
        processed_series += len(inputs)

        tfm_point, _ = model.forecast(horizon=max_h, inputs=inputs)
        tfm_np = np.asarray(tfm_point, dtype=np.float32)

        # Baselines.
        last_value = np.array([arr[-1] for arr in inputs], dtype=np.float32)[:, None]
        naive_last = np.repeat(last_value, max_h, axis=1)

        seasonal = []
        for arr in inputs:
            if len(arr) >= 7:
                week = arr[-7:]
                reps = int(math.ceil(max_h / 7))
                seasonal.append(np.tile(week, reps)[:max_h])
            else:
                seasonal.append(np.repeat(arr[-1], max_h))
        seasonal_np = np.asarray(seasonal, dtype=np.float32)

        batch_keys = batch_df[["store_nbr", "item_nbr"]].astype(np.int32)
        batch_records = []
        for idx, row in batch_keys.iterrows():
            store_nbr = int(row.store_nbr)
            item_nbr = int(row.item_nbr)
            for h in range(1, max_h + 1):
                batch_records.append(
                    {
                        "store_nbr": store_nbr,
                        "item_nbr": item_nbr,
                        "horizon_day": h,
                        "tfm": float(tfm_np[idx, h - 1]),
                        "naive_last": float(naive_last[idx, h - 1]),
                        "naive_seasonal7": float(seasonal_np[idx, h - 1]),
                    }
                )
        pred_df = pd.DataFrame(batch_records)
        merged = pred_df.merge(all_actuals_pd, on=["store_nbr", "item_nbr", "horizon_day"], how="inner")
        if merged.empty:
            continue
        merged["weight"] = merged["item_nbr"].map(perishable_weight_by_item).fillna(1.0).astype(float)

        for horizon in cfg.horizons:
            cut = merged[merged["horizon_day"] <= horizon]
            if cut.empty:
                continue
            for model_name in ("tfm", "naive_last", "naive_seasonal7"):
                pred = cut[model_name].to_numpy(dtype=np.float32)
                actual = cut["actual"].to_numpy(dtype=np.float32)
                weight = cut["weight"].to_numpy(dtype=np.float32)
                out_rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "nwrmsle": nwrmsle(pred, actual, weight),
                        "wmape": wmape(pred, actual),
                        "rows": int(len(cut)),
                    }
                )

    metrics_df = (
        pd.DataFrame(out_rows)
        .groupby(["model", "horizon"], as_index=False)
        .agg({"nwrmsle": "mean", "wmape": "mean", "rows": "sum"})
        .sort_values(["horizon", "nwrmsle"])
    )

    metrics_path = artifacts_dir / "validation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Backtest metrics saved: {}", metrics_path)
    return metrics_path


def build_inventory_policy(
    forecast_h7_path: Path,
    series_stats_path: Path,
    stores_df: pd.DataFrame,
    items_df: pd.DataFrame,
    cfg: PipelineConfig,
    artifacts_dir: Path,
) -> Path:
    """Builds reorder policy from 7-day forecast uncertainty."""
    z_by_service = {
        0.90: 1.28,
        0.95: 1.64,
        0.975: 1.96,
        0.99: 2.33,
    }
    service_key = min(z_by_service.keys(), key=lambda x: abs(x - cfg.service_level))
    z = z_by_service[service_key]

    forecast = pl.scan_parquet(forecast_h7_path)
    lead = min(cfg.lead_time_days, 7)

    agg = (
        forecast.sort(["store_nbr", "item_nbr", "horizon_day"])
        .group_by(["store_nbr", "item_nbr"])
        .agg(
            pl.col("point_forecast").head(lead).sum().alias("lead_point_demand"),
            pl.col("q50").head(lead).sum().alias("lead_q50_demand"),
            pl.col("q90").head(lead).sum().alias("lead_q90_demand"),
            pl.col("point_forecast").sum().alias("week_point_demand"),
        )
        .collect(engine="streaming")
    )

    stats = pl.read_parquet(series_stats_path)
    inv = agg.join(stats, on=["store_nbr", "item_nbr"], how="left")
    inv = inv.with_columns(
        (pl.col("lead_q90_demand") - pl.col("lead_q50_demand")).clip(lower_bound=0.0).alias("safety_stock"),
        (pl.col("trailing_mean_14") * cfg.lead_time_days * 0.60).alias("estimated_on_hand"),
    )
    inv = inv.with_columns(
        (pl.col("lead_q50_demand") + z * pl.col("safety_stock")).alias("reorder_point"),
    )
    inv = inv.with_columns(
        (pl.col("reorder_point") - pl.col("estimated_on_hand")).clip(lower_bound=0.0).alias("recommended_order_qty"),
    )

    store_cluster = stores_df[["store_nbr", "cluster"]].copy()
    store_cluster["warehouse_id"] = store_cluster["cluster"].astype(int).mod(cfg.n_warehouses).map(lambda x: f"WH_{x:02d}")

    item_info = items_df[["item_nbr", "family", "perishable"]].copy()
    item_info["priority_weight"] = np.where(item_info["perishable"] == 1, 1.25, 1.0)

    inv_pd = inv.to_pandas()
    inv_pd = inv_pd.merge(store_cluster[["store_nbr", "warehouse_id"]], on="store_nbr", how="left")
    inv_pd = inv_pd.merge(item_info[["item_nbr", "family", "priority_weight"]], on="item_nbr", how="left")

    inventory_path = artifacts_dir / "inventory_policy.parquet"
    pl.from_pandas(inv_pd).write_parquet(inventory_path)
    logger.info("Inventory policy saved: {}", inventory_path)
    return inventory_path


def heuristic_allocation(
    inventory_path: Path,
    stores_df: pd.DataFrame,
    cfg: PipelineConfig,
    artifacts_dir: Path,
) -> Path:
    """Builds rule-based warehouse-to-store replenishment allocation."""
    inv = pl.read_parquet(inventory_path).to_pandas()

    store_demand = (
        inv.groupby(["store_nbr", "warehouse_id"], as_index=False)["recommended_order_qty"]
        .sum()
        .rename(columns={"recommended_order_qty": "store_demand"})
    )

    capacity = (
        store_demand.groupby("warehouse_id", as_index=False)["store_demand"]
        .sum()
        .rename(columns={"store_demand": "assigned_demand"})
    )
    capacity["capacity"] = capacity["assigned_demand"] * cfg.warehouse_capacity_factor

    available = dict(zip(capacity["warehouse_id"], capacity["capacity"]))
    warehouses = list(available.keys())

    allocations: list[dict[str, Any]] = []
    for row in store_demand.sort_values("store_demand", ascending=False).itertuples(index=False):
        demand = float(row.store_demand)
        store_nbr = int(row.store_nbr)
        primary = str(row.warehouse_id)

        if demand <= 0:
            continue

        take_primary = min(demand, available.get(primary, 0.0))
        if take_primary > 0:
            allocations.append(
                {
                    "store_nbr": store_nbr,
                    "warehouse_id": primary,
                    "allocated_qty": take_primary,
                    "method": "heuristic",
                    "is_primary": True,
                }
            )
            available[primary] -= take_primary
            demand -= take_primary

        if demand > 0:
            for wh in sorted(warehouses, key=lambda w: available.get(w, 0.0), reverse=True):
                if wh == primary or demand <= 0:
                    continue
                take = min(demand, available.get(wh, 0.0))
                if take <= 0:
                    continue
                allocations.append(
                    {
                        "store_nbr": store_nbr,
                        "warehouse_id": wh,
                        "allocated_qty": take,
                        "method": "heuristic",
                        "is_primary": False,
                    }
                )
                available[wh] -= take
                demand -= take

        if demand > 0:
            allocations.append(
                {
                    "store_nbr": store_nbr,
                    "warehouse_id": "UNFILLED",
                    "allocated_qty": demand,
                    "method": "heuristic",
                    "is_primary": False,
                }
            )

    out_df = pd.DataFrame(allocations)
    path = artifacts_dir / "warehouse_allocation_heuristic.parquet"
    pl.from_pandas(out_df).write_parquet(path)
    logger.info("Heuristic allocation saved: {}", path)
    return path


def milp_allocation(
    inventory_path: Path,
    stores_df: pd.DataFrame,
    cfg: PipelineConfig,
    artifacts_dir: Path,
) -> Path:
    """Runs MILP allocation on store-level replenishment demand."""
    inv = pl.read_parquet(inventory_path).to_pandas()
    store_primary = inv.groupby("store_nbr", as_index=False).agg(
        store_demand=("recommended_order_qty", "sum"),
        warehouse_id=("warehouse_id", "first"),
    )

    cap = (
        store_primary.groupby("warehouse_id", as_index=False)["store_demand"]
        .sum()
        .rename(columns={"store_demand": "assigned_demand"})
    )
    cap["capacity"] = cap["assigned_demand"] * cfg.warehouse_capacity_factor

    warehouses = cap["warehouse_id"].tolist()
    stores = store_primary["store_nbr"].tolist()

    capacity = dict(zip(cap["warehouse_id"], cap["capacity"]))
    demand = dict(zip(store_primary["store_nbr"], store_primary["store_demand"]))
    primary = dict(zip(store_primary["store_nbr"], store_primary["warehouse_id"]))

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("Could not create OR-Tools CBC solver.")

    x: dict[tuple[str, int], pywraplp.Variable] = {}
    shortage: dict[int, pywraplp.Variable] = {}

    for wh in warehouses:
        for st in stores:
            x[(wh, st)] = solver.NumVar(0.0, solver.infinity(), f"x_{wh}_{st}")
    for st in stores:
        shortage[st] = solver.NumVar(0.0, solver.infinity(), f"shortage_{st}")

    for st in stores:
        solver.Add(sum(x[(wh, st)] for wh in warehouses) + shortage[st] == demand[st])

    for wh in warehouses:
        solver.Add(sum(x[(wh, st)] for st in stores) <= capacity[wh])

    objective = solver.Objective()
    shortage_penalty = 10.0
    for wh in warehouses:
        for st in stores:
            ship_cost = 1.0 if wh == primary[st] else 1.5
            objective.SetCoefficient(x[(wh, st)], ship_cost)
    for st in stores:
        objective.SetCoefficient(shortage[st], shortage_penalty)
    objective.SetMinimization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"MILP solver returned non-feasible status: {status}")

    rows: list[dict[str, Any]] = []
    for wh in warehouses:
        for st in stores:
            qty = x[(wh, st)].solution_value()
            if qty > 1e-6:
                rows.append(
                    {
                        "warehouse_id": wh,
                        "store_nbr": st,
                        "allocated_qty": qty,
                        "method": "milp",
                        "is_primary": bool(wh == primary[st]),
                    }
                )

    for st in stores:
        q = shortage[st].solution_value()
        if q > 1e-6:
            rows.append(
                {
                    "warehouse_id": "UNFILLED",
                    "store_nbr": st,
                    "allocated_qty": q,
                    "method": "milp",
                    "is_primary": False,
                }
            )

    out_path = artifacts_dir / "warehouse_allocation_milp.parquet"
    pl.from_pandas(pd.DataFrame(rows)).write_parquet(out_path)
    logger.info("MILP allocation saved: {}", out_path)
    return out_path


def build_submission(
    forecast_h16_path: Path,
    raw_dir: Path,
    anchor_date: date,
    artifacts_dir: Path,
) -> Path:
    """Builds Kaggle submission CSV from 16-day forecast."""
    test_df = pd.read_csv(
        raw_dir / "test.csv",
        parse_dates=["date"],
        usecols=["id", "date", "store_nbr", "item_nbr"],
    )
    test_df["horizon_day"] = (test_df["date"].dt.date - anchor_date).apply(lambda x: x.days)

    forecast_df = pl.read_parquet(forecast_h16_path).to_pandas()
    merged = test_df.merge(
        forecast_df[["store_nbr", "item_nbr", "horizon_day", "point_forecast"]],
        on=["store_nbr", "item_nbr", "horizon_day"],
        how="left",
    )
    merged["unit_sales"] = merged["point_forecast"].fillna(0.0).clip(lower=0.0)
    submission = merged[["id", "unit_sales"]]

    submission_path = artifacts_dir / "submission_timesfm.csv"
    submission.to_csv(submission_path, index=False)
    logger.info("Submission saved: {}", submission_path)
    return submission_path


def main() -> None:
    """Runs the full end-to-end retail demand pipeline."""
    cfg = parse_args()
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    set_global_seeds(cfg.seed)
    paths = ensure_paths(cfg)

    logger.info("Step 1/7: Downloading Favorita data...\n")
    download_favorita_data(paths["raw"])

    logger.info("Step 2/7: Building panel datasets...\n")
    train_long_path, panel_path = prepare_series_panel(paths["raw"], paths["processed"], cfg.max_series)

    logger.info("Step 3/7: Loading metadata and model...\n")
    items_df, stores_df, test_df = load_metadata(paths["raw"])
    family_by_item = dict(zip(items_df["item_nbr"].astype(int), items_df["family"].astype(str)))
    perishable_weight_by_item = dict(
        zip(
            items_df["item_nbr"].astype(int),
            np.where(items_df["perishable"].astype(int) == 1, 1.25, 1.0),
        )
    )
    cluster_by_store = dict(zip(stores_df["store_nbr"].astype(int), stores_df["cluster"].astype(int)))
    test_promo_lookup = build_test_promo_lookup(test_df)

    submission_horizon = 16 if cfg.include_submission else 0
    max_horizon = max(max(cfg.horizons), submission_horizon)
    model = get_model_and_compile(cfg, max_horizon=max_horizon, enable_backcast=cfg.use_xreg)

    anchor = get_anchor_date(train_long_path)
    logger.info("Forecast anchor date: {}\n", anchor)

    logger.info("Step 4/7: Forecasting business horizons...\n")
    forecast_paths: dict[int, Path] = {}
    stats_paths: dict[int, Path] = {}
    required_horizons = sorted(set(list(cfg.horizons) + ([16] if cfg.include_submission else [])))
    max_required_horizon = max(required_horizons)
    logger.info(
        "Running base inference once at max horizon {} and slicing shorter horizons {}.",
        max_required_horizon,
        [h for h in required_horizons if h != max_required_horizon],
    )

    base_forecast_path, base_stats_path = forecast_horizon(
        model=model,
        panel_path=panel_path,
        output_dir=paths["artifacts"],
        anchor_date=anchor,
        horizon=max_required_horizon,
        cfg=cfg,
        family_by_item=family_by_item,
        cluster_by_store=cluster_by_store,
        test_promo_lookup=test_promo_lookup,
    )

    for horizon in cfg.horizons:
        forecast_paths[horizon] = materialize_horizon_slice(
            source_forecast_path=base_forecast_path,
            output_dir=paths["artifacts"],
            source_horizon=max_required_horizon,
            target_horizon=horizon,
        )
        stats_paths[horizon] = base_stats_path

    logger.info("Step 5/7: Running validation against baselines...\n")
    metrics_path = evaluate_backtest(
        model=model,
        panel_path=panel_path,
        train_long_path=train_long_path,
        artifacts_dir=paths["artifacts"],
        cfg=cfg,
        family_by_item=family_by_item,
        cluster_by_store=cluster_by_store,
        perishable_weight_by_item=perishable_weight_by_item,
    )

    logger.info("Step 6/7: Inventory + allocation decisions...\n")
    h7 = min(cfg.horizons)
    inventory_path = build_inventory_policy(
        forecast_h7_path=forecast_paths[h7],
        series_stats_path=stats_paths[h7],
        stores_df=stores_df,
        items_df=items_df,
        cfg=cfg,
        artifacts_dir=paths["artifacts"],
    )
    heuristic_path = heuristic_allocation(
        inventory_path=inventory_path,
        stores_df=stores_df,
        cfg=cfg,
        artifacts_dir=paths["artifacts"],
    )

    milp_path = None
    if cfg.run_milp:
        milp_path = milp_allocation(
            inventory_path=inventory_path,
            stores_df=stores_df,
            cfg=cfg,
            artifacts_dir=paths["artifacts"],
        )

    submission_path = None
    if cfg.include_submission:
        logger.info("Step 7/7: Building 16-day submission forecast...\n")
        f16_path = materialize_horizon_slice(
            source_forecast_path=base_forecast_path,
            output_dir=paths["artifacts"],
            source_horizon=max_required_horizon,
            target_horizon=16,
        )
        submission_path = build_submission(
            forecast_h16_path=f16_path,
            raw_dir=paths["raw"],
            anchor_date=anchor,
            artifacts_dir=paths["artifacts"],
        )

    summary = {
        "anchor_date": anchor.isoformat(),
        "forecast_paths": {str(k): str(v) for k, v in forecast_paths.items()},
        "metrics": str(metrics_path),
        "inventory": str(inventory_path),
        "heuristic_allocation": str(heuristic_path),
        "milp_allocation": str(milp_path) if milp_path else None,
        "submission": str(submission_path) if submission_path else None,
    }
    summary_path = paths["artifacts"] / "run_summary.json"
    pd.Series(summary).to_json(summary_path, indent=2)
    logger.info("Run summary: {}", summary_path)


if __name__ == "__main__":
    main()

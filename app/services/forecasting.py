"""Pooled TFT training and inference for all inventoried VMs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from app.config import Settings
from app.data import load_configured_inventory
from app.database import query_rows

TARGETS = {
    "cpu": "cpu_utilization_pct",
    "memory": "memory_utilization_pct",
}
QUANTILES = [0.1, 0.5, 0.9]
STATIC_CATEGORICALS = ["region", "environment", "resource_type", "plan", "team"]
STATIC_REALS = ["vcpu", "ram_gb"]
KNOWN_REALS = [
    "time_idx",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
]
DEFAULT_ENCODER_HOURS = 168


def _model_components() -> tuple[Any, ...]:
    try:
        from lightning.pytorch import Trainer, seed_everything
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError as exc:
        raise RuntimeError(
            "TFT dependencies are not installed. Install requirements.txt and "
            "the CPU-only PyTorch wheel if this host has no CUDA GPU."
        ) from exc
    return (
        Trainer,
        seed_everything,
        EarlyStopping,
        ModelCheckpoint,
        TemporalFusionTransformer,
        TimeSeriesDataSet,
        GroupNormalizer,
        QuantileLoss,
    )


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    hours = frame["timestamp"].dt.hour + frame["timestamp"].dt.minute / 60
    weekdays = frame["timestamp"].dt.dayofweek
    frame = frame.copy()
    frame["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * weekdays / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * weekdays / 7)
    frame["is_weekend"] = (weekdays >= 5).astype("float32")
    return frame


def _load_panel(
    settings: Settings,
    aggregation_hours: int,
    start_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    inventory = load_configured_inventory(settings)
    vm_inventory = inventory[inventory["type"] == "VM"].copy()
    if vm_inventory.empty:
        raise ValueError("The configured inventory contains no VMs.")
    required = {
        "resource_id",
        "plan",
        "region",
        "environment",
        "vcpu",
        "ram_gb",
    }
    if "team" not in vm_inventory:
        vm_inventory["team"] = "unknown"
    vm_inventory["type"] = "VM"
    if settings.database_url:
        rows = query_rows(
            settings,
            """
            SELECT resource_id,
                   time_bucket(%s::interval, timestamp) AS timestamp,
                   avg(cpu_utilization_pct) AS cpu_utilization_pct,
                   avg(memory_utilization_pct) AS memory_utilization_pct
            FROM telemetry
            WHERE resource_id = ANY(%s)
            GROUP BY resource_id, time_bucket(%s::interval, timestamp)
            ORDER BY resource_id, timestamp
            """,
            (
                f"{aggregation_hours} hours",
                vm_inventory["resource_id"].astype(str).tolist(),
                f"{aggregation_hours} hours",
            ),
        )
        if not rows:
            raise ValueError("No VM telemetry matched the configured inventory.")
        panel = pd.DataFrame(rows)
    else:
        telemetry_path = settings.telemetry_path
        if not telemetry_path.is_file():
            raise FileNotFoundError(f"Telemetry dataset not found: {telemetry_path}")
        if telemetry_path.suffix.lower() == ".parquet":
            lazy = pl.scan_parquet(telemetry_path)
        elif telemetry_path.suffix.lower() == ".csv":
            lazy = pl.scan_csv(telemetry_path, try_parse_dates=True)
        else:
            raise ValueError("Telemetry dataset must be .parquet or .csv.")
        columns = set(lazy.collect_schema().names())
        missing = {
            "resource_id",
            "timestamp",
            "cpu_utilization_pct",
            "memory_utilization_pct",
        } - columns
        if missing:
            raise ValueError(f"Telemetry is missing columns: {sorted(missing)}")

        vm_ids = pl.DataFrame({"resource_id": vm_inventory["resource_id"].astype(str)}).lazy()
        aggregated = (
            lazy.with_columns(pl.col("resource_id").cast(pl.String))
            .join(vm_ids, on="resource_id", how="inner")
            .sort(["resource_id", "timestamp"])
            .group_by_dynamic(
                index_column="timestamp",
                every=f"{aggregation_hours}h",
                group_by="resource_id",
            )
            .agg(
                pl.col("cpu_utilization_pct").mean(),
                pl.col("memory_utilization_pct").mean(),
            )
            .collect()
        )
        if aggregated.is_empty():
            raise ValueError("No VM telemetry matched the configured inventory.")
        panel = aggregated.to_pandas()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="raise")
    if panel["timestamp"].dt.tz is not None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(None)
    if start_time is not None and getattr(start_time, "tzinfo", None) is not None:
        start_time = start_time.tz_localize(None)
    panel = panel.merge(
        vm_inventory[
            ["resource_id", "plan", "region", "environment", "vcpu", "ram_gb", "type", "team"]
        ].rename(columns={"type": "resource_type"}),
        on="resource_id",
        how="inner",
        validate="many_to_one",
    )
    panel = panel.sort_values(["resource_id", "timestamp"]).reset_index(drop=True)
    origin = start_time if start_time is not None else panel["timestamp"].min()
    step_seconds = aggregation_hours * 3600
    panel["time_idx"] = (
        ((panel["timestamp"] - origin).dt.total_seconds() / step_seconds).round().astype("int64")
    )
    panel = _add_calendar_features(panel)
    if panel[list(required)].isna().any().any():
        raise ValueError("Inventory and telemetry contain missing model inputs.")
    return panel


def _make_dataset(
    TimeSeriesDataSet: Any,
    GroupNormalizer: Any,
    data: pd.DataFrame,
    target: str,
    encoder_length: int,
    prediction_length: int,
) -> Any:
    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target=target,
        group_ids=["resource_id"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=prediction_length,
        max_prediction_length=prediction_length,
        static_categoricals=STATIC_CATEGORICALS,
        static_reals=STATIC_REALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=[target],
        target_normalizer=GroupNormalizer(groups=["resource_id"], method="robust"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


def train_models(
    settings: Settings,
    aggregation_hours: int = 4,
    prediction_hours: int = 24,
    epochs: int = 5,
    batch_size: int = 64,
) -> dict[str, Any]:
    (
        Trainer,
        seed_everything,
        EarlyStopping,
        ModelCheckpoint,
        TemporalFusionTransformer,
        TimeSeriesDataSet,
        GroupNormalizer,
        QuantileLoss,
    ) = _model_components()
    if aggregation_hours < 1 or 168 % aggregation_hours or prediction_hours % aggregation_hours:
        raise ValueError(
            "aggregation_hours must divide 168, and prediction_hours must be "
            "a positive multiple of aggregation_hours."
        )
    seed_everything(42, workers=True)
    panel = _load_panel(settings, aggregation_hours)
    origin = pd.Timestamp(panel["timestamp"].min())
    encoder_length = DEFAULT_ENCODER_HOURS // aggregation_hours
    prediction_length = prediction_hours // aggregation_hours
    last_idx = int(panel["time_idx"].max())
    validation_start = last_idx - prediction_length + 1
    if validation_start <= encoder_length:
        raise ValueError("Not enough history for the requested train/validation split.")
    training_panel = panel[panel["time_idx"] < validation_start].copy()
    validation_panel = panel[panel["time_idx"] >= validation_start - encoder_length].copy()
    results: dict[str, Any] = {
        "resources_trained": int(panel["resource_id"].nunique()),
        "aggregation_hours": aggregation_hours,
        "prediction_hours": prediction_hours,
        "models": {},
    }

    for name, target in TARGETS.items():
        training_data = _make_dataset(
            TimeSeriesDataSet,
            GroupNormalizer,
            training_panel,
            target,
            encoder_length,
            prediction_length,
        )
        validation_data = TimeSeriesDataSet.from_dataset(
            training_data,
            validation_panel,
            predict=True,
            stop_randomization=True,
            min_prediction_idx=validation_start,
        )
        train_loader = training_data.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        validation_loader = validation_data.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )
        model_path = settings.model_dir / name
        model_path.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(
            dirpath=model_path,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
        )
        trainer = Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=0.1,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3, mode="min"),
                checkpoint,
            ],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=0.003,
            hidden_size=16,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(quantiles=QUANTILES),
            log_interval=-1,
            reduce_on_plateau_patience=2,
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
        )
        if not checkpoint.best_model_path:
            raise RuntimeError(f"Training produced no checkpoint for {name}.")
        results["models"][name] = {
            "checkpoint": str(checkpoint.best_model_path),
            "validation_loss": float(checkpoint.best_model_score),
        }

    manifest = {
        "aggregation_hours": aggregation_hours,
        "prediction_hours": prediction_hours,
        "encoder_hours": DEFAULT_ENCODER_HOURS,
        "origin": origin.isoformat(),
        "resource_count": results["resources_trained"],
        "targets": TARGETS,
    }
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.model_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results["manifest"] = str(manifest_path)
    return results


def forecast_usage(
    settings: Settings,
    hours: int = 24,
    batch_size: int = 128,
) -> list[dict[str, Any]]:
    (
        _Trainer,
        _seed_everything,
        _EarlyStopping,
        _ModelCheckpoint,
        TemporalFusionTransformer,
        TimeSeriesDataSet,
        _GroupNormalizer,
        _QuantileLoss,
    ) = _model_components()
    manifest_path = settings.model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "TFT models are not trained yet; call POST /api/v1/forecast/train first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregation_hours = int(manifest["aggregation_hours"])
    trained_hours = int(manifest["prediction_hours"])
    if hours != trained_hours:
        raise ValueError(
            f"These models predict {trained_hours} hours; retrain with the "
            "requested prediction_hours to change the horizon."
        )
    panel = _load_panel(
        settings,
        aggregation_hours,
        start_time=pd.Timestamp(manifest["origin"]),
    )
    last_time = panel["timestamp"].max()
    future_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=aggregation_hours),
        periods=hours // aggregation_hours,
        freq=f"{aggregation_hours}h",
    )
    resources = panel[["resource_id"] + STATIC_CATEGORICALS + STATIC_REALS].drop_duplicates(
        "resource_id"
    )
    future = pd.DataFrame(
        {
            "resource_id": np.repeat(resources["resource_id"], len(future_times)),
            "timestamp": np.tile(future_times, len(resources)),
        }
    ).merge(resources, on="resource_id", how="left", validate="many_to_one")
    future["time_idx"] = (
        (
            (future["timestamp"] - pd.Timestamp(manifest["origin"])).dt.total_seconds()
            / (aggregation_hours * 3600)
        )
        .round()
        .astype("int64")
    )
    future = _add_calendar_features(future)

    output: list[dict[str, Any]] = []
    for name, target in TARGETS.items():
        checkpoint_path = settings.model_dir / name / "best.ckpt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Trained {name} model is missing: {checkpoint_path}")
        model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        prediction_data = pd.concat(
            [
                panel,
                future.assign(**{target: 0.0}),
            ],
            ignore_index=True,
        )
        dataset = TimeSeriesDataSet.from_parameters(
            model.dataset_parameters,
            prediction_data,
            predict=True,
            stop_randomization=True,
        )
        result = model.predict(
            dataset,
            mode="quantiles",
            return_index=True,
            batch_size=batch_size,
            trainer_kwargs={
                "accelerator": "auto",
                "devices": "auto",
                "logger": False,
                "enable_progress_bar": False,
            },
        )
        prediction = result.output.detach().cpu().numpy()
        index = result.index.reset_index(drop=True)
        if prediction.ndim != 3 or prediction.shape[-1] != len(QUANTILES):
            raise ValueError(f"Unexpected quantile output shape for {name}: {prediction.shape}")
        for sample_idx, item in index.iterrows():
            first_time = int(item["time_idx"])
            for step, quantiles in enumerate(prediction[sample_idx]):
                timestamp = pd.Timestamp(manifest["origin"]) + pd.Timedelta(
                    hours=(first_time + step) * aggregation_hours
                )
                low, median, high = np.clip(quantiles, 0, 100)
                output.append(
                    {
                        "resource_id": item["resource_id"],
                        "timestamp": timestamp.isoformat(),
                        "target": name,
                        "q10": float(low),
                        "q50": float(median),
                        "q90": float(high),
                    }
                )
    return output


def trained_prediction_hours(settings: Settings) -> int:
    """Return the horizon stored with the trained models."""
    manifest_path = settings.model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "TFT models are not trained yet; call POST /api/v1/forecast/train first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hours = int(manifest["prediction_hours"])
    if hours < 1:
        raise ValueError("The trained model manifest has an invalid prediction horizon.")
    return hours

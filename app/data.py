"""Read configured inventory, pricing, and telemetry datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Settings
from app.database import inventory_frame, pricing_frame


def read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def load_inventory(path: Path) -> pd.DataFrame:
    inventory = read_csv(
        path,
        {
            "resource_id",
            "type",
            "plan",
            "region",
            "environment",
            "vcpu",
            "ram_gb",
            "disk_size_gb",
        },
    )
    if inventory["resource_id"].duplicated().any():
        raise ValueError("Inventory resource_id values must be unique.")
    for column in ("vcpu", "ram_gb", "disk_size_gb"):
        inventory[column] = pd.to_numeric(inventory[column], errors="raise")
    if (inventory[["vcpu", "ram_gb", "disk_size_gb"]] < 0).any().any():
        raise ValueError("Inventory capacity values must be non-negative.")
    return inventory


def load_pricing(path: Path) -> pd.DataFrame:
    pricing = read_csv(path, {"plan", "type", "vcpu", "ram_gb", "rate", "storage_rate"})
    if pricing["plan"].duplicated().any():
        raise ValueError("Pricing plan values must be unique.")
    for column in ("vcpu", "ram_gb", "rate", "storage_rate"):
        pricing[column] = pd.to_numeric(pricing[column], errors="raise")
    if (
        not pricing[["vcpu", "ram_gb", "rate", "storage_rate"]]
        .map(lambda value: pd.notna(value) and value >= 0)
        .all()
        .all()
    ):
        raise ValueError("Pricing capacities and rates must be non-negative numbers.")
    return pricing


def load_configured_inventory(settings: Settings) -> pd.DataFrame:
    """Load inventory from PostgreSQL when configured, otherwise from the CSV."""
    if settings.database_url:
        inventory = inventory_frame(settings)
        if inventory.empty:
            raise ValueError("PostgreSQL resource inventory is empty.")
        if inventory["resource_id"].duplicated().any():
            raise ValueError("Inventory resource_id values must be unique.")
        for column in ("vcpu", "ram_gb", "disk_size_gb"):
            inventory[column] = pd.to_numeric(inventory[column], errors="raise")
        return inventory
    return load_inventory(settings.inventory_path)


def load_configured_pricing(settings: Settings) -> pd.DataFrame:
    """Load pricing from PostgreSQL when configured, otherwise from the CSV."""
    if settings.database_url:
        pricing = pricing_frame(settings)
        if pricing.empty:
            raise ValueError("PostgreSQL pricing catalog is empty.")
        if pricing["plan"].duplicated().any():
            raise ValueError("Pricing plan values must be unique.")
        for column in ("vcpu", "ram_gb", "rate", "storage_rate"):
            pricing[column] = pd.to_numeric(pricing[column], errors="raise")
        return pricing
    return load_pricing(settings.pricing_path)


def load_telemetry(path: Path, resource_ids: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry file not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Telemetry dataset must be a .parquet or .csv file.")
    required = {
        "resource_id",
        "timestamp",
        "cpu_utilization_pct",
        "memory_utilization_pct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if resource_ids is not None:
        frame = frame[frame["resource_id"].isin(resource_ids)].copy()
    if frame.empty:
        raise ValueError("No telemetry rows matched the configured inventory.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    for column in ("cpu_utilization_pct", "memory_utilization_pct"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[column].isna().any() or ((frame[column] < 0) | (frame[column] > 100)).any():
            raise ValueError(f"{column} must contain percentages from 0 to 100.")
    return frame.sort_values(["resource_id", "timestamp"])


def inventory_records(inventory: pd.DataFrame) -> list[dict[str, Any]]:
    return inventory.where(pd.notna(inventory), None).to_dict(orient="records")

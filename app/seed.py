"""Load the configured CSV and Parquet datasets into PostgreSQL/TimescaleDB."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb
from pyarrow import parquet

from app.config import settings
from app.data import load_inventory, load_pricing
from app.database import connect

TELEMETRY_COLUMNS = (
    "cpu_utilization_pct",
    "memory_utilization_pct",
    "disk_utilization_pct",
    "disk_iops",
    "network_in_mbps",
    "network_out_mbps",
)
BATCH_SIZE = 20_000


def _json_attributes(record: dict[str, Any], excluded: set[str]) -> Jsonb:
    attributes = {}
    for key, value in record.items():
        if key in excluded or pd.isna(value):
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            attributes[key] = value
        else:
            attributes[key] = str(value)
    return Jsonb(attributes)


def _seed_inventory(connection: psycopg.Connection, path: Path) -> int:
    load_inventory(path)
    frame = pd.read_csv(path)
    columns = (
        "resource_id",
        "resource_type",
        "plan",
        "region",
        "environment",
        "team",
        "vcpu",
        "ram_gb",
        "disk_size_gb",
        "attributes",
    )
    sql = """
        INSERT INTO resource_inventory
            (resource_id, resource_type, plan, region, environment, team,
             vcpu, ram_gb, disk_size_gb, attributes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (resource_id) DO UPDATE SET
            resource_type = EXCLUDED.resource_type, plan = EXCLUDED.plan,
            region = EXCLUDED.region, environment = EXCLUDED.environment,
            team = EXCLUDED.team, vcpu = EXCLUDED.vcpu, ram_gb = EXCLUDED.ram_gb,
            disk_size_gb = EXCLUDED.disk_size_gb, attributes = EXCLUDED.attributes
    """
    excluded = set(columns) | {"type"}
    for record in frame.to_dict(orient="records"):
        values = (
            str(record["resource_id"]),
            str(record["type"]),
            str(record["plan"]),
            str(record["region"]),
            str(record["environment"]),
            None if pd.isna(record.get("team")) else str(record["team"]),
            float(record["vcpu"]),
            float(record["ram_gb"]),
            float(record["disk_size_gb"]),
            _json_attributes(record, excluded),
        )
        connection.execute(sql, values)
    return len(frame)


def _seed_pricing(connection: psycopg.Connection, path: Path) -> int:
    load_pricing(path)
    frame = pd.read_csv(path)
    required = {"plan", "type", "vcpu", "ram_gb", "rate", "storage_rate"}
    sql = """
        INSERT INTO pricing_catalog
            (plan, resource_type, vcpu, ram_gb, rate, storage_rate, attributes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (plan) DO UPDATE SET
            resource_type = EXCLUDED.resource_type, vcpu = EXCLUDED.vcpu,
            ram_gb = EXCLUDED.ram_gb, rate = EXCLUDED.rate,
            storage_rate = EXCLUDED.storage_rate, attributes = EXCLUDED.attributes
    """
    excluded = set(required)
    for record in frame.to_dict(orient="records"):
        values = (
            str(record["plan"]),
            str(record["type"]),
            float(record["vcpu"]),
            float(record["ram_gb"]),
            float(record["rate"]),
            float(record["storage_rate"]),
            _json_attributes(record, excluded),
        )
        connection.execute(sql, values)
    return len(frame)


def _seed_telemetry(connection: psycopg.Connection, path: Path) -> int:
    connection.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS telemetry_seed_stage (
            resource_id text NOT NULL,
            timestamp timestamptz NOT NULL,
            cpu_utilization_pct double precision,
            memory_utilization_pct double precision,
            disk_utilization_pct double precision,
            disk_iops double precision,
            network_in_mbps double precision,
            network_out_mbps double precision,
            attributes jsonb NOT NULL DEFAULT '{}'::jsonb
        ) ON COMMIT DELETE ROWS
        """
    )
    connection.commit()
    if path.suffix.lower() == ".parquet":
        batches = iter(parquet.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE))
    elif path.suffix.lower() == ".csv":
        batches = (batch for batch in pd.read_csv(path, chunksize=BATCH_SIZE))
    else:
        raise ValueError("Telemetry dataset must be CSV or Parquet.")

    required = {"resource_id", "timestamp", "cpu_utilization_pct", "memory_utilization_pct"}
    count = 0
    for batch in batches:
        frame = batch.to_pandas() if hasattr(batch, "to_pandas") else batch
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Telemetry is missing required columns: {sorted(missing)}")
        for name in TELEMETRY_COLUMNS:
            if name not in frame:
                continue
            frame[name] = pd.to_numeric(frame[name], errors="raise")
            if name in {"cpu_utilization_pct", "memory_utilization_pct"}:
                if frame[name].isna().any() or not frame[name].between(0, 100).all():
                    raise ValueError(f"{name} must contain percentages from 0 to 100.")
            elif (frame[name].dropna() < 0).any():
                raise ValueError(f"{name} must be non-negative.")
        extras = [
            column
            for column in frame.columns
            if column not in {"resource_id", "timestamp", *TELEMETRY_COLUMNS}
        ]
        with connection.transaction():
            with connection.cursor() as cursor:
                with cursor.copy(
                    """
                    COPY telemetry_seed_stage
                        (resource_id, timestamp, cpu_utilization_pct, memory_utilization_pct,
                         disk_utilization_pct, disk_iops, network_in_mbps, network_out_mbps,
                         attributes)
                    FROM STDIN
                    """
                ) as copy:
                    for record in frame.to_dict(orient="records"):
                        timestamp = pd.Timestamp(record["timestamp"]).to_pydatetime()
                        metrics = [
                            None
                            if column not in record or pd.isna(record[column])
                            else float(record[column])
                            for column in TELEMETRY_COLUMNS
                        ]
                        attributes = {
                            name: record[name]
                            for name in extras
                            if not pd.isna(record[name])
                            and isinstance(record[name], (str, int, float, bool))
                        }
                        copy.write_row(
                            (
                                str(record["resource_id"]),
                                timestamp,
                                *metrics,
                                Jsonb(attributes),
                            )
                        )
                cursor.execute(
                    """
                    INSERT INTO telemetry
                        (resource_id, timestamp, cpu_utilization_pct, memory_utilization_pct,
                         disk_utilization_pct, disk_iops, network_in_mbps, network_out_mbps,
                         attributes)
                    SELECT resource_id, timestamp, cpu_utilization_pct, memory_utilization_pct,
                           disk_utilization_pct, disk_iops, network_in_mbps, network_out_mbps,
                           attributes
                    FROM telemetry_seed_stage
                    ON CONFLICT (resource_id, timestamp) DO UPDATE SET
                        cpu_utilization_pct = EXCLUDED.cpu_utilization_pct,
                        memory_utilization_pct = EXCLUDED.memory_utilization_pct,
                        disk_utilization_pct = EXCLUDED.disk_utilization_pct,
                        disk_iops = EXCLUDED.disk_iops,
                        network_in_mbps = EXCLUDED.network_in_mbps,
                        network_out_mbps = EXCLUDED.network_out_mbps,
                        attributes = EXCLUDED.attributes
                    """
                )
        count += len(frame)
        print(f"Imported {count:,} telemetry rows", flush=True)
    return count


def seed_all() -> dict[str, int]:
    paths = {
        "inventory": settings.inventory_path,
        "pricing": settings.pricing_path,
        "telemetry": settings.telemetry_path,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Seed input does not exist: {path}")

    with connect(settings) as connection:
        inventory_count = _seed_inventory(connection, paths["inventory"])
        pricing_count = _seed_pricing(connection, paths["pricing"])
        connection.commit()
        telemetry_count = _seed_telemetry(connection, paths["telemetry"])
        connection.execute(
            """
            INSERT INTO dataset_imports (dataset_name, source_path, imported_rows)
            VALUES (%s, %s, %s), (%s, %s, %s), (%s, %s, %s)
            ON CONFLICT (dataset_name) DO UPDATE SET
                source_path = EXCLUDED.source_path, imported_rows = EXCLUDED.imported_rows,
                imported_at = now()
            """,
            (
                "inventory",
                str(paths["inventory"]),
                inventory_count,
                "pricing",
                str(paths["pricing"]),
                pricing_count,
                "telemetry",
                str(paths["telemetry"]),
                telemetry_count,
            ),
        )
    return {
        "inventory": inventory_count,
        "pricing": pricing_count,
        "telemetry": telemetry_count,
    }


if __name__ == "__main__":
    try:
        totals = seed_all()
    except (OSError, ValueError, psycopg.Error) as exc:
        print(f"Dataset seeding failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(totals, indent=2))

"""PostgreSQL access helpers for inventory, pricing, telemetry, and chat history."""

from __future__ import annotations

from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from app.config import Settings


def connect(settings: Settings) -> psycopg.Connection[dict[str, Any]]:
    if not settings.database_url:
        raise RuntimeError("PostgreSQL is not configured. Set DATABASE_URL.")
    return psycopg.connect(settings.database_url, row_factory=dict_row)


def query_rows(
    settings: Settings, statement: str, parameters: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    with connect(settings) as connection, connection.cursor() as cursor:
        cursor.execute(statement, parameters)
        return list(cursor.fetchall())


def inventory_frame(settings: Settings) -> pd.DataFrame:
    rows = query_rows(
        settings,
        """
        SELECT attributes || jsonb_build_object(
            'resource_id', resource_id, 'type', resource_type, 'plan', plan,
            'region', region, 'environment', environment, 'team', team,
            'vcpu', vcpu, 'ram_gb', ram_gb, 'disk_size_gb', disk_size_gb
        ) AS resource
        FROM resource_inventory ORDER BY resource_id
        """,
    )
    return pd.DataFrame([row["resource"] for row in rows])


def pricing_frame(settings: Settings) -> pd.DataFrame:
    rows = query_rows(
        settings,
        """
        SELECT attributes || jsonb_build_object(
            'plan', plan, 'type', resource_type, 'vcpu', vcpu,
            'ram_gb', ram_gb, 'rate', rate, 'storage_rate', storage_rate
        ) AS price
        FROM pricing_catalog ORDER BY plan
        """,
    )
    return pd.DataFrame([row["price"] for row in rows])


def telemetry_columns(settings: Settings) -> set[str]:
    rows = query_rows(
        settings,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'telemetry'
        """,
    )
    return {str(row["column_name"]) for row in rows}


def telemetry_rows(
    settings: Settings,
    statement: str,
    parameters: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    return query_rows(settings, statement, parameters)

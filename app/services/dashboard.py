"""Dashboard metrics, history, rightsizing actions, and explainable alerts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from app.config import Settings
from app.data import load_inventory, load_pricing

METRICS = {
    "cpu_utilization_pct": {"unit": "percent", "label": "CPU"},
    "memory_utilization_pct": {"unit": "percent", "label": "Memory"},
    "disk_utilization_pct": {"unit": "percent", "label": "Storage"},
    "disk_iops": {"unit": "IOPS", "label": "Disk IOPS"},
    "network_in_mbps": {"unit": "Mbps", "label": "Network in"},
    "network_out_mbps": {"unit": "Mbps", "label": "Network out"},
}
DIMENSIONS = ("region", "environment", "team")
MONTHLY_HOURS = 730


def _telemetry_scan(settings: Settings) -> tuple[pl.LazyFrame, set[str]]:
    path = settings.telemetry_path
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pl.scan_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pl.scan_csv(path, try_parse_dates=True)
    else:
        raise ValueError("Telemetry dataset must be a .parquet or .csv file.")
    schema = frame.collect_schema()
    columns = set(schema.names())
    missing = {"resource_id", "timestamp"} - columns
    if missing:
        raise ValueError(f"Telemetry is missing columns: {sorted(missing)}")
    if schema["timestamp"] == pl.String:
        frame = frame.with_columns(pl.col("timestamp").str.to_datetime(strict=True))
    return frame.with_columns(pl.col("resource_id").cast(pl.String)), columns


def _filtered_inventory(
    inventory: pd.DataFrame,
    region: str | None,
    environment: str | None,
    team: str | None,
) -> pd.DataFrame:
    selected = inventory.copy()
    for column, value in (
        ("region", region),
        ("environment", environment),
        ("team", team),
    ):
        if value is not None:
            if column not in selected:
                return selected.iloc[0:0]
            selected = selected[
                selected[column].fillna("").astype(str).str.casefold() == value.casefold()
            ]
    return selected


def _nullable_metric_exprs(columns: set[str]) -> list[pl.Expr]:
    return [
        pl.col(name).cast(pl.Float64, strict=True).alias(name)
        if name in columns
        else pl.lit(None, dtype=pl.Float64).alias(name)
        for name in METRICS
    ]


def _aggregate_stats(
    settings: Settings,
    resource_ids: list[str],
    hours: int,
) -> tuple[list[dict[str, Any]], pd.Timestamp, pd.Timestamp, set[str]]:
    if not resource_ids:
        return [], pd.NaT, pd.NaT, set()
    telemetry, columns = _telemetry_scan(settings)
    available_metrics = set(METRICS).intersection(columns)
    filtered = telemetry.filter(pl.col("resource_id").is_in(resource_ids))
    end_rows = filtered.select(pl.col("timestamp").max()).collect().to_dicts()
    end_value = end_rows[0]["timestamp"] if end_rows else None
    if end_value is None:
        return [], pd.NaT, pd.NaT, available_metrics
    end = pd.Timestamp(end_value)
    start = end - pd.Timedelta(hours=hours)
    filtered = filtered.filter(
        (pl.col("timestamp") >= start.to_pydatetime())
        & (pl.col("timestamp") <= end.to_pydatetime())
    ).with_columns(_nullable_metric_exprs(columns))
    aggregations: list[pl.Expr] = [
        pl.len().alias("sample_count"),
        pl.col("timestamp").min().alias("first_seen"),
        pl.col("timestamp").max().alias("last_seen"),
    ]
    for name in METRICS:
        aggregations.extend(
            [
                pl.col(name).mean().alias(f"{name}_mean"),
                pl.col(name).quantile(0.95, interpolation="linear").alias(f"{name}_p95"),
                pl.col(name).quantile(0.99, interpolation="linear").alias(f"{name}_p99"),
                pl.col(name).max().alias(f"{name}_max"),
                pl.col(name).last().alias(f"{name}_latest"),
            ]
        )
    records = (
        filtered.sort(["resource_id", "timestamp"])
        .group_by("resource_id", maintain_order=True)
        .agg(aggregations)
        .collect()
        .to_dicts()
    )
    return records, start, end, available_metrics


def _monthly_rate(resource: dict[str, Any], rates: dict[str, dict[str, Any]]) -> float | None:
    plan = str(resource["plan"])
    pricing = rates.get(plan)
    if pricing is None:
        return None
    resource_type = str(resource["type"])
    if resource_type == "VM" or resource_type == "public_ip":
        return float(pricing["rate"]) * MONTHLY_HOURS
    if resource_type == "block_volume":
        return float(pricing["storage_rate"]) * float(resource.get("disk_size_gb", 0))
    return None


def dashboard_resources(
    settings: Settings,
    hours: int = 24 * 30,
    region: str | None = None,
    environment: str | None = None,
    team: str | None = None,
) -> dict[str, Any]:
    if not 1 <= hours <= 24 * 90:
        raise ValueError("Dashboard hours must be from 1 to 2160.")
    inventory = load_inventory(settings.inventory_path)
    selected = _filtered_inventory(inventory, region, environment, team)
    ids = selected["resource_id"].astype(str).tolist()
    stats, start, end, available_metrics = _aggregate_stats(settings, ids, hours)
    stats_by_id = {str(row["resource_id"]): row for row in stats}
    pricing = load_pricing(settings.pricing_path)
    rates = pricing.set_index("plan").to_dict(orient="index")

    resources: list[dict[str, Any]] = []
    for resource in selected.to_dict(orient="records"):
        resource_id = str(resource["resource_id"])
        values = stats_by_id.get(resource_id)
        item: dict[str, Any] = {
            **{key: None if pd.isna(value) else value for key, value in resource.items()},
            "resource_id": resource_id,
            "monthly_cost_estimate": _monthly_rate(resource, rates),
            "telemetry_available": values is not None,
            "available_metrics": sorted(available_metrics),
        }
        if values is not None:
            item["sample_count"] = values["sample_count"]
            item["first_seen"] = pd.Timestamp(values["first_seen"]).isoformat()
            item["last_seen"] = pd.Timestamp(values["last_seen"]).isoformat()
            item["metrics"] = {
                name: {
                    "latest": values.get(f"{name}_latest"),
                    "mean": values.get(f"{name}_mean"),
                    "p95": values.get(f"{name}_p95"),
                    "p99": values.get(f"{name}_p99"),
                    "max": values.get(f"{name}_max"),
                    "unit": definition["unit"],
                }
                for name, definition in METRICS.items()
                if name in available_metrics
            }
        else:
            item["sample_count"] = 0
            item["metrics"] = {}
        resources.append(item)
    return {
        "count": len(resources),
        "filters": {"region": region, "environment": environment, "team": team},
        "window": {
            "hours": hours,
            "start": start.isoformat() if not pd.isna(start) else None,
            "end": end.isoformat() if not pd.isna(end) else None,
        },
        "available_metrics": sorted(available_metrics),
        "resources": resources,
    }


def dashboard_summary(
    settings: Settings,
    hours: int = 24 * 30,
    region: str | None = None,
    environment: str | None = None,
    team: str | None = None,
) -> dict[str, Any]:
    result = dashboard_resources(settings, hours, region, environment, team)
    resources = result["resources"]
    by_type: dict[str, int] = {}
    for resource in resources:
        resource_type = str(resource["type"])
        by_type[resource_type] = by_type.get(resource_type, 0) + 1
    summary: dict[str, Any] = {}
    for name in result["available_metrics"]:
        measured = [
            resource["metrics"][name]
            for resource in resources
            if name in resource["metrics"] and resource["metrics"][name]["p95"] is not None
        ]
        summary[name] = {
            "average_latest": (
                round(float(np.mean([item["latest"] for item in measured])), 3)
                if measured
                else None
            ),
            "average_resource_p95": (
                round(float(np.mean([item["p95"] for item in measured])), 3) if measured else None
            ),
            "max_resource_p99": (
                round(float(max(item["p99"] for item in measured)), 3) if measured else None
            ),
            "unit": METRICS[name]["unit"],
            "resource_count": len(measured),
        }
    return {
        "resource_count": len(resources),
        "resources_with_telemetry": sum(resource["telemetry_available"] for resource in resources),
        "resources_by_type": by_type,
        "filters": result["filters"],
        "window": result["window"],
        "metrics": summary,
    }


def resource_history(
    settings: Settings,
    resource_id: str,
    hours: int = 24 * 7,
    interval_minutes: int = 60,
) -> dict[str, Any]:
    if not 1 <= hours <= 24 * 90:
        raise ValueError("History hours must be from 1 to 2160.")
    if interval_minutes not in {5, 15, 60, 240, 1440}:
        raise ValueError("interval_minutes must be one of 5, 15, 60, 240, or 1440.")
    inventory = load_inventory(settings.inventory_path)
    matches = inventory[inventory["resource_id"].astype(str) == resource_id]
    if matches.empty:
        raise KeyError(f"Unknown resource_id: {resource_id}")
    telemetry, columns = _telemetry_scan(settings)
    available_metrics = sorted(set(METRICS).intersection(columns))
    filtered = telemetry.filter(pl.col("resource_id") == resource_id)
    end_rows = filtered.select(pl.col("timestamp").max()).collect().to_dicts()
    if not end_rows or end_rows[0]["timestamp"] is None:
        raise KeyError(f"No telemetry history found for resource_id: {resource_id}")
    end = pd.Timestamp(end_rows[0]["timestamp"])
    start = end - pd.Timedelta(hours=hours)
    group_columns = [
        pl.col(name).cast(pl.Float64, strict=True).alias(name)
        if name in columns
        else pl.lit(None, dtype=pl.Float64).alias(name)
        for name in METRICS
    ]
    rows = (
        filtered.filter(
            (pl.col("timestamp") >= start.to_pydatetime())
            & (pl.col("timestamp") <= end.to_pydatetime())
        )
        .with_columns(group_columns)
        .sort("timestamp")
        .group_by_dynamic(
            "timestamp",
            every=f"{interval_minutes}m",
            group_by="resource_id",
            label="left",
            closed="left",
        )
        .agg(
            [pl.len().alias("sample_count")] + [pl.col(name).mean().alias(name) for name in METRICS]
        )
        .sort("timestamp")
        .collect()
        .to_dicts()
    )
    points = [
        {
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "sample_count": row["sample_count"],
            "metrics": {
                name: {"value": row.get(name), "unit": METRICS[name]["unit"]}
                for name in available_metrics
            },
        }
        for row in rows
    ]
    return {
        "resource": matches.iloc[0].where(pd.notna(matches.iloc[0]), None).to_dict(),
        "window": {"hours": hours, "start": start.isoformat(), "end": end.isoformat()},
        "interval_minutes": interval_minutes,
        "available_metrics": available_metrics,
        "points": points,
    }


def _confidence(sample_count: int, first_seen: Any, last_seen: Any) -> tuple[str, str]:
    if sample_count < 288:
        return "LOW", "Fewer than 24 hours of five-minute telemetry are available."
    observed_hours = (pd.Timestamp(last_seen) - pd.Timestamp(first_seen)).total_seconds() / 3600
    if sample_count >= 2016 and observed_hours >= 7 * 24:
        return "HIGH", "At least seven days of telemetry support the selected percentile."
    return "MEDIUM", "At least 24 hours, but less than seven days, support the percentile."


def _hourly_low_windows(
    settings: Settings,
    resource_ids: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, tuple[int, int]]:
    if not resource_ids:
        return {}
    telemetry, columns = _telemetry_scan(settings)
    if "cpu_utilization_pct" not in columns:
        return {}
    rows = (
        telemetry.filter(
            pl.col("resource_id").is_in(resource_ids)
            & (pl.col("timestamp") >= start.to_pydatetime())
            & (pl.col("timestamp") <= end.to_pydatetime())
        )
        .with_columns(pl.col("cpu_utilization_pct").cast(pl.Float64, strict=True))
        .group_by(["resource_id", pl.col("timestamp").dt.hour().alias("hour")])
        .agg(pl.col("cpu_utilization_pct").quantile(0.95).alias("cpu_p95"))
        .collect()
        .to_dicts()
    )
    by_resource: dict[str, dict[int, float]] = {}
    for row in rows:
        by_resource.setdefault(str(row["resource_id"]), {})[int(row["hour"])] = float(
            row["cpu_p95"]
        )
    schedules: dict[str, tuple[int, int]] = {}
    for resource_id, hourly in by_resource.items():
        low_hours = {hour for hour, value in hourly.items() if value < 5}
        if len(low_hours) < 4 or len(low_hours) == 24:
            continue
        candidates: list[tuple[int, int, int]] = []
        for start_hour in range(24):
            if start_hour not in low_hours or (start_hour - 1) % 24 in low_hours:
                continue
            duration = 0
            while (start_hour + duration) % 24 in low_hours and duration < 24:
                duration += 1
            candidates.append((duration, start_hour, (start_hour + duration) % 24))
        if candidates:
            duration, start_hour, end_hour = max(candidates)
            if duration >= 4:
                schedules[resource_id] = (start_hour, end_hour)
    return schedules


def actionable_recommendations(
    settings: Settings,
    hours: int = 24 * 30,
    peak_percentile: int = 99,
    headroom: float = 1.3,
    region: str | None = None,
    environment: str | None = None,
    team: str | None = None,
) -> dict[str, Any]:
    if peak_percentile not in {95, 99}:
        raise ValueError("peak_percentile must be 95 or 99.")
    if not np.isfinite(headroom) or not 1 <= headroom <= 3:
        raise ValueError("headroom must be between 1 and 3.")
    dashboard = dashboard_resources(settings, hours, region, environment, team)
    inventory = load_inventory(settings.inventory_path)
    selected = _filtered_inventory(inventory, region, environment, team)
    pricing = load_pricing(settings.pricing_path)
    vm_catalog = pricing[pricing["type"] == "VM"]
    all_prices_by_plan = pricing.set_index("plan").to_dict(orient="index")
    resource_stats = {item["resource_id"]: item for item in dashboard["resources"]}
    end = pd.Timestamp(dashboard["window"]["end"]) if dashboard["window"]["end"] else pd.NaT
    start = pd.Timestamp(dashboard["window"]["start"]) if dashboard["window"]["start"] else pd.NaT
    schedules = (
        _hourly_low_windows(
            settings,
            selected["resource_id"].astype(str).tolist(),
            start,
            end,
        )
        if not pd.isna(start) and not pd.isna(end)
        else {}
    )
    recommendations: list[dict[str, Any]] = []
    for resource in selected.to_dict(orient="records"):
        resource_id = str(resource["resource_id"])
        stats = resource_stats[resource_id]
        metrics = stats["metrics"]
        sample_count = int(stats["sample_count"])
        confidence, confidence_reason = (
            _confidence(sample_count, stats.get("first_seen"), stats.get("last_seen"))
            if sample_count
            else ("LOW", "No telemetry is available for this resource.")
        )
        if resource["type"] == "VM":
            cpu = metrics.get("cpu_utilization_pct", {}).get(f"p{peak_percentile}")
            memory = metrics.get("memory_utilization_pct", {}).get(f"p{peak_percentile}")
            disk = metrics.get("disk_utilization_pct", {}).get(f"p{peak_percentile}")
            if cpu is None or memory is None:
                recommendations.append(
                    {
                        "resource_id": resource_id,
                        "action": "collect_telemetry",
                        "reason": "CPU and memory percentiles are required before a safe resize recommendation.",
                        "monthly_savings_estimate": 0.0,
                        "risk_percentage": None,
                        "risk_level": "UNKNOWN",
                        "confidence_level": "LOW",
                        "confidence_reason": confidence_reason,
                    }
                )
                continue
            network_in = metrics.get("network_in_mbps", {}).get(f"p{peak_percentile}")
            network_out = metrics.get("network_out_mbps", {}).get(f"p{peak_percentile}")
            disk_iops = metrics.get("disk_iops", {}).get(f"p{peak_percentile}")
            is_idle = (
                float(cpu) <= 5
                and float(memory) <= 15
                and disk is not None
                and float(disk) <= 20
                and disk_iops is not None
                and float(disk_iops) <= 1
                and network_in is not None
                and float(network_in) <= 0.1
                and network_out is not None
                and float(network_out) <= 0.1
            )
            if is_idle:
                current_cost = _monthly_rate(resource, all_prices_by_plan)
                production = str(resource.get("environment", "")).casefold() in {
                    "prod",
                    "production",
                }
                recommendations.append(
                    {
                        "resource_id": resource_id,
                        "action": "review_idle_resource" if production else "stop_idle_resource",
                        "current_plan": resource["plan"],
                        "monthly_savings_estimate": 0.0
                        if production
                        else round(current_cost or 0, 2),
                        "risk_percentage": round(max(float(cpu), float(memory)), 2),
                        "risk_level": "LOW",
                        "risk_definition": (
                            "Measured p99 CPU and memory utilization; this is not a probability of failure."
                        ),
                        "confidence_level": confidence,
                        "confidence_reason": confidence_reason,
                        "reason": (
                            f"Over the selected {hours}-hour window, p{peak_percentile} CPU "
                            f"was {cpu:.1f}%, memory {memory:.1f}%, disk use {disk:.1f}%, "
                            f"disk IOPS {disk_iops:.2f}, and network throughput stayed "
                            "below 0.1 Mbps. "
                            + (
                                "Production resources are flagged for owner review only."
                                if production
                                else "Consider stopping it after confirming it is not required."
                            )
                        ),
                    }
                )
                continue
            current_vcpu = int(resource["vcpu"])
            current_ram = int(resource["ram_gb"])
            required_vcpu = max(1, int(np.ceil(current_vcpu * float(cpu) / 100 * headroom)))
            required_ram = max(1, int(np.ceil(current_ram * float(memory) / 100 * headroom)))
            candidates = vm_catalog[
                (vm_catalog["vcpu"] >= required_vcpu) & (vm_catalog["ram_gb"] >= required_ram)
            ].sort_values(["rate", "plan"])
            current_cost = _monthly_rate(resource, all_prices_by_plan)
            if candidates.empty:
                recommendations.append(
                    {
                        "resource_id": resource_id,
                        "action": "no_catalog_match",
                        "current_plan": resource["plan"],
                        "reason": (
                            f"p{peak_percentile} demand with {headroom:.0%} headroom "
                            f"requires at least {required_vcpu} vCPU and {required_ram} GB RAM, "
                            "which no catalog plan provides."
                        ),
                        "monthly_savings_estimate": 0.0,
                        "risk_percentage": max(float(cpu), float(memory), float(disk or 0)),
                        "risk_level": "HIGH",
                        "confidence_level": confidence,
                        "confidence_reason": confidence_reason,
                        "peak_percentile": peak_percentile,
                    }
                )
                continue
            target = candidates.iloc[0].to_dict()
            target_plan = str(target["plan"])
            target_cpu = float(cpu) * current_vcpu / int(target["vcpu"])
            target_memory = float(memory) * current_ram / int(target["ram_gb"])
            target_disk = float(disk or 0)
            risk_percentage = round(min(100.0, max(target_cpu, target_memory, target_disk)), 2)
            risk_level = (
                "HIGH" if risk_percentage >= 85 else "MEDIUM" if risk_percentage >= 70 else "LOW"
            )
            target_cost = float(target["rate"]) * MONTHLY_HOURS
            savings = round(max(0.0, (current_cost or target_cost) - target_cost), 2)
            if target_plan == str(resource["plan"]):
                action = "maintain"
                reason = (
                    f"The current plan meets p{peak_percentile} CPU and memory demand "
                    f"with {headroom:.0%} headroom."
                )
            elif target_cost < (current_cost or 0):
                action = "resize_instance"
                reason = (
                    f"The cheapest catalog plan meets the p{peak_percentile} peak "
                    f"({cpu:.1f}% CPU, {memory:.1f}% memory) plus {headroom:.0%} headroom."
                )
            else:
                action = "resize_instance"
                reason = (
                    f"The current plan does not meet p{peak_percentile} peak demand "
                    f"({cpu:.1f}% CPU, {memory:.1f}% memory) plus {headroom:.0%} headroom."
                )
            recommendations.append(
                {
                    "resource_id": resource_id,
                    "action": action,
                    "current_plan": resource["plan"],
                    "recommended_plan": target_plan,
                    "target_configuration": {
                        "vcpu": int(target["vcpu"]),
                        "ram_gb": int(target["ram_gb"]),
                    },
                    "peak_percentile": peak_percentile,
                    "peak_cpu_pct": round(float(cpu), 2),
                    "peak_memory_pct": round(float(memory), 2),
                    "peak_disk_pct": round(float(disk), 2) if disk is not None else None,
                    "required_vcpu": required_vcpu,
                    "required_ram_gb": required_ram,
                    "target_peak_cpu_pct": round(target_cpu, 2),
                    "target_peak_memory_pct": round(target_memory, 2),
                    "risk_percentage": risk_percentage,
                    "risk_level": risk_level,
                    "risk_definition": (
                        "Maximum projected p95/p99 CPU, memory, or disk utilization on the target plan; "
                        "this is a capacity-pressure index, not a probability of failure."
                    ),
                    "monthly_savings_estimate": savings,
                    "confidence_level": confidence,
                    "confidence_reason": confidence_reason,
                    "reason": reason,
                    "schedule": None,
                }
            )
            if (
                str(resource.get("environment", "")).casefold()
                in {"dev", "development", "test", "testing", "staging"}
                and resource_id in schedules
            ):
                schedule_start, schedule_end = schedules[resource_id]
                duration = (schedule_end - schedule_start) % 24
                recommendations.append(
                    {
                        "resource_id": resource_id,
                        "action": "schedule_nonproduction_shutdown",
                        "current_plan": resource["plan"],
                        "schedule": {
                            "start": f"{schedule_start:02d}:00",
                            "end": f"{schedule_end:02d}:00",
                            "days": "daily",
                        },
                        "peak_percentile": peak_percentile,
                        "risk_percentage": round(max(float(cpu), float(memory)), 2),
                        "risk_level": risk_level,
                        "risk_definition": (
                            "Current-plan peak utilization; schedule suggestion is not an execution command."
                        ),
                        "monthly_savings_estimate": round(
                            float(all_prices_by_plan[str(resource["plan"])]["rate"])
                            * duration
                            * 30,
                            2,
                        ),
                        "confidence_level": confidence,
                        "confidence_reason": confidence_reason,
                        "reason": (
                            "The p95 CPU utilization stayed below 5% during these hours "
                            "across the selected history window."
                        ),
                    }
                )
        elif resource["type"] in {"block_volume", "public_ip"}:
            attachment_column = next(
                (
                    name
                    for name in ("attached_to", "attached_resource_id", "attachment_id")
                    if name in selected.columns
                ),
                None,
            )
            attachment = resource.get(attachment_column) if attachment_column else None
            if attachment_column and (pd.isna(attachment) or not str(attachment).strip()):
                monthly_cost = _monthly_rate(resource, all_prices_by_plan)
                recommendations.append(
                    {
                        "resource_id": resource_id,
                        "action": "clean_up_orphan",
                        "current_plan": resource["plan"],
                        "monthly_savings_estimate": round(monthly_cost or 0, 2),
                        "risk_percentage": 0.0,
                        "risk_level": "LOW",
                        "confidence_level": "HIGH",
                        "confidence_reason": (
                            f"The inventory explicitly includes an empty {attachment_column} field."
                        ),
                        "reason": "No attached resource is recorded; verify in the cloud account before cleanup.",
                    }
                )
            else:
                recommendations.append(
                    {
                        "resource_id": resource_id,
                        "action": "review_attachment",
                        "current_plan": resource["plan"],
                        "monthly_savings_estimate": 0.0,
                        "risk_percentage": None,
                        "risk_level": "UNKNOWN",
                        "confidence_level": "LOW",
                        "confidence_reason": "The inventory does not provide attachment metadata.",
                        "reason": (
                            "Attachment status is unknown, so this resource cannot safely be "
                            "classified as orphaned."
                        ),
                    }
                )
    return {
        "count": len(recommendations),
        "peak_percentile": peak_percentile,
        "headroom": headroom,
        "window": dashboard["window"],
        "filters": dashboard["filters"],
        "recommendations": recommendations,
    }


def _bucketed_history(
    settings: Settings,
    resource_ids: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    every: str,
) -> list[dict[str, Any]]:
    if not resource_ids:
        return []
    telemetry, columns = _telemetry_scan(settings)
    names = [name for name in METRICS if name in columns]
    if not names:
        return []
    values = (
        telemetry.filter(
            pl.col("resource_id").is_in(resource_ids)
            & (pl.col("timestamp") >= start.to_pydatetime())
            & (pl.col("timestamp") <= end.to_pydatetime())
        )
        .with_columns([pl.col(name).cast(pl.Float64, strict=True).alias(name) for name in names])
        .sort(["resource_id", "timestamp"])
        .group_by_dynamic(
            "timestamp",
            every=every,
            group_by="resource_id",
            label="left",
            closed="left",
        )
        .agg([pl.col(name).quantile(0.95).alias(name) for name in names])
        .collect()
        .to_dicts()
    )
    return values


def dashboard_alerts(
    settings: Settings,
    hours: int = 24 * 30,
    region: str | None = None,
    environment: str | None = None,
    team: str | None = None,
    monthly_budget_usd: float | None = None,
    previous_month_cost_usd: float | None = None,
) -> dict[str, Any]:
    dashboard = dashboard_resources(settings, hours, region, environment, team)
    inventory = load_inventory(settings.inventory_path)
    selected = _filtered_inventory(inventory, region, environment, team)
    alerts: list[dict[str, Any]] = []
    items = {item["resource_id"]: item for item in dashboard["resources"]}

    for resource_id, item in items.items():
        metrics = item["metrics"]
        for metric_name in (
            "cpu_utilization_pct",
            "memory_utilization_pct",
            "disk_utilization_pct",
        ):
            metric = metrics.get(metric_name)
            if metric and metric.get("p99") is not None and metric["p99"] >= 85:
                alerts.append(
                    {
                        "type": "capacity_limit",
                        "severity": "high" if metric["p99"] >= 95 else "medium",
                        "resource_id": resource_id,
                        "metric": metric_name,
                        "value": round(float(metric["p99"]), 2),
                        "threshold": 85,
                        "reason": (
                            f"{metric_name} p99 reached {metric['p99']:.1f}% in the selected "
                            f"{hours}-hour window, approaching the 100% capacity limit."
                        ),
                    }
                )

    if dashboard["window"]["end"]:
        end = pd.Timestamp(dashboard["window"]["end"])
        start = pd.Timestamp(dashboard["window"]["start"])
        hourly_start = max(start, end - pd.Timedelta(days=8))
        hourly_rows = _bucketed_history(
            settings,
            selected["resource_id"].astype(str).tolist(),
            hourly_start,
            end,
            "1h",
        )
        by_resource: dict[str, list[dict[str, Any]]] = {}
        for row in hourly_rows:
            by_resource.setdefault(str(row["resource_id"]), []).append(row)
        daily_rows = _bucketed_history(
            settings,
            selected["resource_id"].astype(str).tolist(),
            max(start, end - pd.Timedelta(days=14)),
            end,
            "1d",
        )
        daily_by_resource: dict[str, list[dict[str, Any]]] = {}
        for row in daily_rows:
            daily_by_resource.setdefault(str(row["resource_id"]), []).append(row)
        for resource_id, rows in by_resource.items():
            rows.sort(key=lambda row: row["timestamp"])
            if len(rows) >= 25:
                latest = rows[-1]
                baseline_rows = rows[-169:-1]
                for metric_name in (
                    "cpu_utilization_pct",
                    "memory_utilization_pct",
                    "disk_utilization_pct",
                    "network_in_mbps",
                    "network_out_mbps",
                ):
                    current = latest.get(metric_name)
                    baseline = [
                        row.get(metric_name)
                        for row in baseline_rows
                        if row.get(metric_name) is not None
                    ]
                    if current is None or len(baseline) < 24:
                        continue
                    average = float(np.mean(baseline))
                    deviation = float(np.std(baseline))
                    minimum_delta = 10.0 if metric_name.startswith("network_") else 15.0
                    if current >= average + 3 * deviation and current - average >= minimum_delta:
                        alerts.append(
                            {
                                "type": "usage_spike",
                                "severity": "high" if current >= 90 else "medium",
                                "resource_id": resource_id,
                                "metric": metric_name,
                                "value": round(float(current), 2),
                                "baseline_mean": round(average, 2),
                                "reason": (
                                    f"Latest hourly {metric_name} is {current:.1f}, at least three "
                                    f"baseline standard deviations and {minimum_delta:.0f} points "
                                    f"above the preceding hourly average ({average:.1f})."
                                ),
                            }
                        )
            for metric_name in ("cpu_utilization_pct", "memory_utilization_pct"):
                values = [
                    float(row[metric_name])
                    for row in daily_by_resource.get(resource_id, [])
                    if row.get(metric_name) is not None
                ]
                if len(values) >= 5:
                    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
                    if values[-1] >= 70 and slope >= 0.5:
                        alerts.append(
                            {
                                "type": "capacity_trend",
                                "severity": "medium",
                                "resource_id": resource_id,
                                "metric": metric_name,
                                "daily_change_pct_points": round(slope, 2),
                                "reason": (
                                    f"Daily p95 {metric_name} is rising by about {slope:.2f} "
                                    "percentage points per day and is currently near capacity."
                                ),
                            }
                        )

    current_monthly = sum(
        float(item["monthly_cost_estimate"])
        for item in dashboard["resources"]
        if item["monthly_cost_estimate"] is not None
    )
    if monthly_budget_usd is not None and current_monthly > monthly_budget_usd:
        alerts.append(
            {
                "type": "projected_budget_overrun",
                "severity": "high",
                "projected_monthly_cost_usd": round(current_monthly, 2),
                "monthly_budget_usd": monthly_budget_usd,
                "reason": (
                    f"Current catalog rates project ${current_monthly:.2f}/month, "
                    f"which exceeds the supplied ${monthly_budget_usd:.2f} budget."
                ),
            }
        )
    if previous_month_cost_usd is not None and previous_month_cost_usd > 0:
        increase = (current_monthly - previous_month_cost_usd) / previous_month_cost_usd * 100
        if increase >= 20:
            alerts.append(
                {
                    "type": "cost_increase",
                    "severity": "high" if increase >= 50 else "medium",
                    "projected_monthly_cost_usd": round(current_monthly, 2),
                    "previous_month_cost_usd": previous_month_cost_usd,
                    "increase_pct": round(increase, 2),
                    "reason": (
                        f"Current catalog-rate projection is {increase:.1f}% above the "
                        "previous-month spend supplied by the caller."
                    ),
                }
            )
    return {
        "count": len(alerts),
        "window": dashboard["window"],
        "filters": dashboard["filters"],
        "cost_basis": "Current catalog rates; not provider-billed usage.",
        "alerts": alerts,
    }

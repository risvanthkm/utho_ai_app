"""Cost projections and forecast-based VM plan recommendations."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

HOURS_PER_DAY = 24
DAYS_PER_STORAGE_MONTH = 30


def _forecast_peaks(forecasts: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    peaks: dict[str, dict[str, float]] = {}
    for row in forecasts:
        resource_id = str(row["resource_id"])
        target = str(row["target"])
        q90 = float(row["q90"])
        if target not in {"cpu", "memory"} or not math.isfinite(q90) or not 0 <= q90 <= 100:
            raise ValueError("Forecasts require finite CPU and memory q90 percentages.")
        peaks.setdefault(resource_id, {})[target] = max(
            q90, peaks.get(resource_id, {}).get(target, 0.0)
        )
    return peaks


def _recommended_plan(
    resource: dict[str, Any],
    catalog: pd.DataFrame,
    peak: dict[str, float] | None,
    headroom: float,
) -> tuple[str | None, str, str]:
    current_plan = str(resource["plan"])
    if peak is None or not {"cpu", "memory"}.issubset(peak):
        return None, "MAINTAIN", "No complete CPU and memory forecast; retain the current plan."

    current_vcpu = int(resource["vcpu"])
    current_ram = int(resource["ram_gb"])
    needed_vcpu = max(1, math.ceil(current_vcpu * peak["cpu"] / 100 * headroom))
    needed_ram = max(1, math.ceil(current_ram * peak["memory"] / 100 * headroom))
    candidates = catalog[
        (catalog["vcpu"] >= needed_vcpu) & (catalog["ram_gb"] >= needed_ram)
    ].sort_values(["rate", "plan"])
    if candidates.empty:
        return (
            None,
            "NO_CATALOG_MATCH",
            (
                f"Forecast demand needs at least {needed_vcpu} vCPU and "
                f"{needed_ram} GB RAM, which no catalog VM plan provides."
            ),
        )

    selected = candidates.iloc[0]
    current_candidates = catalog[catalog["plan"] == current_plan]
    if current_candidates.empty:
        return (
            str(selected["plan"]),
            "NO_CATALOG_MATCH",
            f"Current plan {current_plan} is absent from the VM pricing catalog.",
        )
    selected_plan = str(selected["plan"])
    selected_rate = float(selected["rate"])
    current_rate = float(current_candidates.iloc[0]["rate"])
    if selected_plan == current_plan:
        return (
            selected_plan,
            "MAINTAIN",
            "Current plan is the least-cost catalog plan meeting forecast demand.",
        )
    if selected_rate < current_rate:
        return (
            selected_plan,
            "DOWNSIZE",
            "A cheaper plan meets q90 CPU and memory demand plus safety headroom.",
        )
    return (
        selected_plan,
        "UPSCALE",
        "Forecast q90 demand plus safety headroom exceeds the current plan.",
    )


def _schedule_suggestion(
    resource_id: str,
    environment: str,
    forecasts: list[dict[str, Any]],
    interval_hours: int = 4,
) -> dict[str, str] | None:
    if environment.casefold() not in {"dev", "development", "test", "testing", "staging"}:
        return None
    cpu_rows = sorted(
        (
            row
            for row in forecasts
            if str(row["resource_id"]) == resource_id and row["target"] == "cpu"
        ),
        key=lambda row: row["timestamp"],
    )
    if len(cpu_rows) < 2:
        return None
    low_intervals: list[list[dict[str, Any]]] = []
    current_window: list[dict[str, Any]] = []
    for row in cpu_rows:
        if float(row["q90"]) < 5:
            if current_window:
                previous_time = pd.Timestamp(current_window[-1]["timestamp"])
                current_time = pd.Timestamp(row["timestamp"])
                if current_time - previous_time == pd.Timedelta(hours=interval_hours):
                    current_window.append(row)
                else:
                    low_intervals.append(current_window)
                    current_window = [row]
            else:
                current_window = [row]
        else:
            if current_window:
                low_intervals.append(current_window)
                current_window = []
    if current_window:
        low_intervals.append(current_window)
    windows = [window for window in low_intervals if len(window) >= 2]
    if not windows:
        return None
    window = max(windows, key=len)
    start = pd.Timestamp(window[0]["timestamp"])
    end = pd.Timestamp(window[-1]["timestamp"]) + pd.Timedelta(hours=interval_hours)
    return {
        "start": start.strftime("%H:%M"),
        "end": end.strftime("%H:%M"),
        "days": "daily",
        "basis": "forecast q90 CPU below 5% for consecutive intervals",
    }


def build_recommendations(
    inventory: pd.DataFrame,
    pricing: pd.DataFrame,
    forecasts: list[dict[str, Any]],
    hours: int,
    headroom: float,
) -> list[dict[str, Any]]:
    if hours < 1:
        raise ValueError("Projection hours must be positive.")
    if not math.isfinite(headroom) or headroom < 1:
        raise ValueError("Headroom must be a finite value greater than or equal to 1.")
    catalog = pricing[pricing["type"] == "VM"].copy()
    catalog["rate"] = pd.to_numeric(catalog["rate"], errors="raise")
    peaks = _forecast_peaks(forecasts)
    period_hours = hours
    recommendations = []
    for resource in inventory[inventory["type"] == "VM"].to_dict(orient="records"):
        resource_id = str(resource["resource_id"])
        current_plan = str(resource["plan"])
        current_rows = catalog[catalog["plan"] == current_plan]
        if current_rows.empty:
            raise ValueError(f"Current VM plan {current_plan} for {resource_id} is not in pricing.")
        current_rate = float(current_rows.iloc[0]["rate"])
        peak = peaks.get(resource_id)
        plan, action, reason = _recommended_plan(resource, catalog, peak, headroom)
        if plan is None:
            recommended_rate = None
            recommended_cost = None
            savings = None
            target_vcpu = None
            target_ram = None
        else:
            target = catalog[catalog["plan"] == plan].iloc[0]
            recommended_rate = float(target["rate"])
            recommended_cost = round(recommended_rate * period_hours, 2)
            savings = round(current_rate * period_hours - recommended_cost, 2)
            target_vcpu = int(target["vcpu"])
            target_ram = int(target["ram_gb"])
        recommendations.append(
            {
                "resource_id": resource_id,
                "environment": str(resource.get("environment", "")),
                "region": str(resource.get("region", "")),
                "current_plan": current_plan,
                "current_vcpu": int(resource["vcpu"]),
                "current_ram_gb": int(resource["ram_gb"]),
                "recommended_plan": plan,
                "recommended_vcpu": target_vcpu,
                "recommended_ram_gb": target_ram,
                "action": action,
                "estimated_cost_current": round(current_rate * period_hours, 2),
                "estimated_cost_recommended": recommended_cost,
                "estimated_savings": savings,
                "peak_cpu_q90_pct": peak.get("cpu") if peak else None,
                "peak_memory_q90_pct": peak.get("memory") if peak else None,
                "schedule": _schedule_suggestion(
                    resource_id,
                    str(resource.get("environment", "")),
                    forecasts,
                ),
                "reason": reason,
            }
        )
    return recommendations


def inventory_costs(
    inventory: pd.DataFrame,
    pricing: pd.DataFrame,
    days: int,
    recommendations: list[dict[str, Any]] | None = None,
    shutdown_start_minutes: int | None = None,
    shutdown_end_minutes: int | None = None,
    weekdays_only: bool = False,
    shutdown_resource_ids: set[str] | None = None,
) -> dict[str, Any]:
    if days < 1:
        raise ValueError("Projection days must be positive.")
    if (shutdown_start_minutes is None) != (shutdown_end_minutes is None):
        raise ValueError("Both shutdown start and end times are required.")
    shutdown_hours_per_day = 0.0
    if shutdown_start_minutes is not None and shutdown_end_minutes is not None:
        if shutdown_start_minutes == shutdown_end_minutes:
            raise ValueError("Shutdown start and end times must differ.")
        shutdown_minutes = (shutdown_end_minutes - shutdown_start_minutes) % (24 * 60)
        shutdown_hours_per_day = shutdown_minutes / 60

    rates = pricing.set_index("plan")
    recommendation_by_id = {item["resource_id"]: item for item in (recommendations or [])}
    component_totals = {
        "vm_compute": 0.0,
        "public_ip": 0.0,
        "storage": 0.0,
    }
    resource_rows = []
    for resource in inventory.to_dict(orient="records"):
        resource_id = str(resource["resource_id"])
        plan = str(resource["plan"])
        if plan not in rates.index:
            raise ValueError(f"Missing pricing for inventory plan: {plan}")
        rate = float(rates.loc[plan, "rate"])
        storage_rate = float(rates.loc[plan, "storage_rate"])
        resource_type = str(resource["type"])
        vm_hours = days * 24
        resource_is_scheduled = (
            shutdown_resource_ids is None or resource_id in shutdown_resource_ids
        )
        if resource_type == "VM" and shutdown_hours_per_day and resource_is_scheduled:
            active_days = days * (5 / 7 if weekdays_only else 1)
            vm_hours -= active_days * shutdown_hours_per_day
        vm_cost = round(rate * vm_hours, 2) if resource_type == "VM" else 0.0
        ip_cost = round(rate * days * 24, 2) if resource_type == "public_ip" else 0.0
        disk_gb = float(resource.get("disk_size_gb", 0) or 0)
        storage_cost = round(storage_rate * disk_gb * days / 30, 2)

        recommendation = recommendation_by_id.get(resource_id)
        recommended_plan = (
            recommendation.get("recommended_plan")
            if recommendation and recommendation.get("recommended_plan")
            else plan
        )
        recommended_rate = float(rates.loc[recommended_plan, "rate"])
        if resource_type == "VM" and shutdown_hours_per_day and resource_is_scheduled:
            recommended_vm_cost = round(recommended_rate * vm_hours, 2)
        else:
            recommended_vm_cost = (
                round(recommended_rate * days * 24, 2) if resource_type == "VM" else 0.0
            )
        recommended_ip_cost = ip_cost
        recommended_storage_rate = float(rates.loc[recommended_plan, "storage_rate"])
        recommended_storage_cost = round(recommended_storage_rate * disk_gb * days / 30, 2)
        current_total = round(vm_cost + ip_cost + storage_cost, 2)
        recommended_total = round(
            recommended_vm_cost + recommended_ip_cost + recommended_storage_cost, 2
        )
        component_totals["vm_compute"] += vm_cost
        component_totals["public_ip"] += ip_cost
        component_totals["storage"] += storage_cost
        resource_rows.append(
            {
                "resource_id": resource_id,
                "type": resource_type,
                "environment": resource.get("environment"),
                "region": resource.get("region"),
                "current_plan": plan,
                "recommended_plan": recommended_plan,
                "disk_size_gb": disk_gb,
                "vm_compute_cost": vm_cost,
                "public_ip_cost": ip_cost,
                "storage_cost": storage_cost,
                "current_total_cost": current_total,
                "recommended_total_cost": recommended_total,
                "estimated_savings": round(current_total - recommended_total, 2),
            }
        )

    recommended_totals = {
        "vm_compute": sum(row["vm_compute_cost"] for row in resource_rows if row["type"] == "VM"),
        "public_ip": sum(
            row["public_ip_cost"] for row in resource_rows if row["type"] == "public_ip"
        ),
        "storage": sum(row["storage_cost"] for row in resource_rows),
    }
    current_total = round(sum(component_totals.values()), 2)
    recommended_total = round(sum(recommended_totals.values()), 2)
    return {
        "days": days,
        "currency": "USD",
        "assumptions": {
            "vm_and_ip_pricing": "hourly",
            "storage_pricing": "per GB-month, prorated over 30 days",
            "prices_constant": True,
            "shutdown_days": "weekdays only" if weekdays_only else "every day",
            "shutdown_hours_per_day": shutdown_hours_per_day,
        },
        "current_cost": {key: round(value, 2) for key, value in component_totals.items()}
        | {"total": current_total},
        "recommended_cost": {key: round(value, 2) for key, value in recommended_totals.items()}
        | {"total": recommended_total},
        "estimated_savings": round(current_total - recommended_total, 2),
        "resources": resource_rows,
    }

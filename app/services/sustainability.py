"""Approximate energy use and operational-carbon changes from VM resizing."""

from __future__ import annotations

from typing import Any

from app.config import Settings


def estimate_sustainability(
    recommendations: list[dict[str, Any]],
    settings: Settings,
    hours: int,
    scheduled_shutdown_resource_ids: set[str] | None = None,
    shutdown_hours_per_day: float = 0.0,
    weekdays_only: bool = False,
    days: int | None = None,
) -> dict[str, Any]:
    if hours < 1:
        raise ValueError("Energy projection hours must be positive.")
    if (
        settings.watts_per_vcpu_at_idle < 0
        or settings.watts_per_vcpu_at_full_load < settings.watts_per_vcpu_at_idle
        or settings.power_usage_effectiveness < 1
        or settings.grid_carbon_kg_per_kwh < 0
    ):
        raise ValueError("Energy and carbon assumptions in environment are invalid.")
    if shutdown_hours_per_day < 0 or shutdown_hours_per_day > 24:
        raise ValueError("Daily shutdown hours must be between 0 and 24.")
    shutdown_days = (
        min(hours / 24, days) * (5 / 7 if weekdays_only else 1)
        if days is not None
        else hours / 24 * (5 / 7 if weekdays_only else 1)
    )
    shutdown_total_hours = shutdown_hours_per_day * shutdown_days

    def kwh(vcpu: int, load_pct: float | None, runtime_hours: float) -> float:
        utilization = min(max(load_pct or 0.0, 0.0), 100.0) / 100
        watts = vcpu * (
            settings.watts_per_vcpu_at_idle
            + utilization * (settings.watts_per_vcpu_at_full_load - settings.watts_per_vcpu_at_idle)
        )
        return watts * settings.power_usage_effectiveness * runtime_hours / 1000

    current_energy = 0.0
    recommended_energy = 0.0
    for item in recommendations:
        current_load = item.get("peak_cpu_q90_pct")
        current_energy += kwh(int(item["current_vcpu"]), current_load, hours)
        target_vcpu = item.get("recommended_vcpu") or item["current_vcpu"]
        if item.get("peak_cpu_q90_pct") is not None and item.get("recommended_vcpu"):
            demand_vcpu = int(item["current_vcpu"]) * float(current_load) / 100
            new_load = min(demand_vcpu / int(target_vcpu) * 100, 100)
        else:
            new_load = current_load
        scheduled = (
            scheduled_shutdown_resource_ids is not None
            and str(item.get("resource_id")) in scheduled_shutdown_resource_ids
        )
        runtime_hours = max(0.0, hours - shutdown_total_hours if scheduled else hours)
        recommended_energy += kwh(int(target_vcpu), new_load, runtime_hours)
    saved_energy = current_energy - recommended_energy
    current_carbon = current_energy * settings.grid_carbon_kg_per_kwh
    recommended_carbon = recommended_energy * settings.grid_carbon_kg_per_kwh
    carbon_saved = current_carbon - recommended_carbon
    score = min(max(100 * carbon_saved / current_carbon, 0.0), 100.0) if current_carbon else 0.0
    return {
        "estimated_energy_saved_kwh": round(saved_energy, 3),
        "estimated_carbon_saved_kg_co2e": round(carbon_saved, 3),
        "sustainability_score": round(score, 2),
        "current_estimated_energy_kwh": round(current_energy, 3),
        "recommended_estimated_energy_kwh": round(recommended_energy, 3),
        "assumptions": [
            (
                f"{settings.watts_per_vcpu_at_idle:g} W/vCPU idle and "
                f"{settings.watts_per_vcpu_at_full_load:g} W/vCPU at full load"
            ),
            f"PUE {settings.power_usage_effectiveness:g}",
            f"Grid intensity {settings.grid_carbon_kg_per_kwh:g} kg CO2e/kWh",
            "Energy is a capacity/utilization estimate, not measured facility power.",
            "Scheduled shutdown energy savings are included only for selected VMs.",
        ],
    }

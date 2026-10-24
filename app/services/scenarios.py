"""Deterministic recurring shutdown scenario parsing and simulation."""

from __future__ import annotations

import re
from datetime import time
from typing import Any

import pandas as pd

from app.services.optimization import inventory_costs

TIME_RANGE = re.compile(
    r"\b(?P<start>\d{1,2}(?::\d{2})?\s*(?:am|pm))\s*"
    r"(?:to|until|through|-)\s*"
    r"(?P<end>\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
    re.IGNORECASE,
)
ENVIRONMENT = re.compile(
    r"\b(?:all\s+)?(?P<environment>dev|development|staging|test|testing|prod|production)"
    r"(?:\s+(?:vms?|instances?|servers?))?\b",
    re.IGNORECASE,
)


def _parse_clock(value: str) -> time:
    compact = re.sub(r"\s+", "", value).upper()
    formats = ("%I%p", "%I:%M%p")
    for pattern in formats:
        try:
            from datetime import datetime

            return datetime.strptime(compact, pattern).time()
        except ValueError:
            continue
    raise ValueError(f"Could not parse time of day: {value}")


def parse_scenario_text(text: str) -> dict[str, Any]:
    time_match = TIME_RANGE.search(text)
    if not time_match:
        raise ValueError(
            "Describe a shutdown time range, for example 'shut down dev VMs from 8 PM to 8 AM'."
        )
    environment_match = ENVIRONMENT.search(text)
    if not environment_match:
        raise ValueError(
            "Scenario text must identify a resource environment such as dev or staging."
        )
    environment = environment_match.group("environment").lower()
    normalized_environment = {
        "development": "dev",
        "testing": "test",
        "production": "prod",
    }.get(environment, environment)
    return {
        "environment": normalized_environment,
        "shutdown_start": _parse_clock(time_match.group("start")),
        "shutdown_end": _parse_clock(time_match.group("end")),
    }


def simulate_scenario(
    inventory: pd.DataFrame,
    pricing: pd.DataFrame,
    recommendations: list[dict[str, Any]],
    days: int,
    environment: str | None,
    resource_ids: list[str] | None,
    shutdown_start: time,
    shutdown_end: time,
    weekdays_only: bool,
    use_recommended_plans: bool,
) -> dict[str, Any]:
    selected = inventory[inventory["type"] == "VM"].copy()
    if environment is not None:
        selected = selected[
            selected["environment"].astype(str).str.casefold() == environment.casefold()
        ]
    if resource_ids is not None:
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("resource_ids must not contain duplicates.")
        unknown = sorted(set(resource_ids) - set(inventory["resource_id"].astype(str)))
        if unknown:
            raise ValueError(f"Unknown resource IDs in scenario: {unknown}")
        selected = selected[selected["resource_id"].astype(str).isin(resource_ids)]
    if selected.empty:
        raise ValueError("The scenario did not select any VMs.")
    selected_ids = set(selected["resource_id"].astype(str))
    adjusted_inventory = inventory.copy()
    plan_by_id = {
        item["resource_id"]: item["recommended_plan"]
        for item in recommendations
        if item.get("recommended_plan")
    }
    if use_recommended_plans:
        chosen_ids = set(inventory.loc[inventory["type"] == "VM", "resource_id"].astype(str))
        adjusted_inventory.loc[
            adjusted_inventory["resource_id"].astype(str).isin(chosen_ids), "plan"
        ] = (
            adjusted_inventory.loc[
                adjusted_inventory["resource_id"].astype(str).isin(chosen_ids), "resource_id"
            ]
            .map(plan_by_id)
            .fillna(
                adjusted_inventory.loc[
                    adjusted_inventory["resource_id"].astype(str).isin(chosen_ids), "plan"
                ]
            )
        )

    start_minutes = shutdown_start.hour * 60 + shutdown_start.minute
    end_minutes = shutdown_end.hour * 60 + shutdown_end.minute
    current = inventory_costs(
        inventory,
        pricing,
        days,
        shutdown_start_minutes=start_minutes,
        shutdown_end_minutes=end_minutes,
        weekdays_only=weekdays_only,
        shutdown_resource_ids=selected_ids,
    )
    proposed = inventory_costs(
        adjusted_inventory,
        pricing,
        days,
        shutdown_start_minutes=start_minutes,
        shutdown_end_minutes=end_minutes,
        weekdays_only=weekdays_only,
        shutdown_resource_ids=selected_ids,
    )
    baseline = inventory_costs(inventory, pricing, days)
    return {
        "scenario": {
            "environment": environment,
            "resource_ids": sorted(selected_ids),
            "shutdown_start": shutdown_start.isoformat(timespec="minutes"),
            "shutdown_end": shutdown_end.isoformat(timespec="minutes"),
            "weekdays_only": weekdays_only,
            "days": days,
            "plans": "forecast-recommended" if use_recommended_plans else "current",
        },
        "baseline": baseline,
        "scenario_cost": proposed,
        "scenario_only_cost": current,
        "estimated_savings_vs_baseline": round(
            baseline["current_cost"]["total"] - proposed["recommended_cost"]["total"],
            2,
        ),
        "note": (
            "Shutdown savings assume VM compute billing stops during scheduled "
            "hours. Storage and public IP charges continue. No cloud resources are "
            "actually stopped."
        ),
    }

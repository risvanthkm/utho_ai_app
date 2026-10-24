"""Versioned HTTP endpoints for forecasting and optimization."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.data import inventory_records, load_inventory, load_pricing
from app.schemas import (
    CostForecastRequest,
    DescriptionRequest,
    ForecastRequest,
    RecommendationRequest,
    ScenarioRequest,
    TerraformRequest,
    TrainRequest,
)
from app.services.descriptions import generate_description
from app.services.forecasting import (
    forecast_usage,
    train_models,
    trained_prediction_hours,
)
from app.services.optimization import build_recommendations, inventory_costs
from app.services.scenarios import parse_scenario_text, simulate_scenario
from app.services.sustainability import estimate_sustainability
from app.services.terraform import generate_vm_terraform

router = APIRouter(prefix="/api/v1")


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, (ValueError, KeyError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=503, detail=str(exc))
    raise exc


def _forecast_or_error(hours: int) -> list[dict[str, Any]]:
    try:
        return forecast_usage(settings, hours=hours)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise _http_error(exc) from exc


def _recommendations(
    hours: int,
    headroom: float,
    inventory: pd.DataFrame,
    pricing: pd.DataFrame,
    forecasts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return build_recommendations(inventory, pricing, forecasts, hours, headroom)


def _scenario_values(request: ScenarioRequest) -> dict[str, Any]:
    values: dict[str, Any] = {
        "environment": request.environment,
        "shutdown_start": request.shutdown_start,
        "shutdown_end": request.shutdown_end,
    }
    if request.scenario_text:
        parsed = parse_scenario_text(request.scenario_text)
        for key in ("environment", "shutdown_start", "shutdown_end"):
            if values[key] is not None and values[key] != parsed[key]:
                raise ValueError(f"scenario_text conflicts with the explicit {key} field.")
            values[key] = parsed[key]
    if values["shutdown_start"] is None or values["shutdown_end"] is None:
        raise ValueError("Provide shutdown_start and shutdown_end or a supported scenario_text.")
    if values["environment"] is None and not request.resource_ids:
        raise ValueError("Select an environment or provide resource_ids for the shutdown scenario.")
    return values


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/resources")
def list_resources() -> dict[str, Any]:
    try:
        inventory = load_inventory(settings.inventory_path)
        return {"count": len(inventory), "resources": inventory_records(inventory)}
    except (FileNotFoundError, ValueError) as exc:
        raise _http_error(exc) from exc


@router.post("/forecast/train")
async def train_forecasters(request: TrainRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            train_models,
            settings,
            request.aggregation_hours,
            request.prediction_hours,
            request.epochs,
            request.batch_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise _http_error(exc) from exc


@router.post("/forecast/usage")
def get_usage_forecast(request: ForecastRequest) -> dict[str, Any]:
    forecasts = _forecast_or_error(request.hours)
    return {
        "hours": request.hours,
        "resource_count": len({row["resource_id"] for row in forecasts}),
        "forecasts": forecasts,
    }


@router.post("/optimization/recommendations")
def get_recommendations(request: RecommendationRequest) -> dict[str, Any]:
    try:
        inventory = load_inventory(settings.inventory_path)
        pricing = load_pricing(settings.pricing_path)
        forecasts = _forecast_or_error(request.hours)
        recommendations = _recommendations(
            request.hours, request.headroom, inventory, pricing, forecasts
        )
        sustainability = estimate_sustainability(recommendations, settings, request.hours)
        return {
            "hours": request.hours,
            "count": len(recommendations),
            "recommendations": recommendations,
            "sustainability": sustainability,
        }
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise _http_error(exc) from exc


@router.post("/costs/forecast")
def get_cost_forecast(request: CostForecastRequest) -> dict[str, Any]:
    try:
        inventory = load_inventory(settings.inventory_path)
        pricing = load_pricing(settings.pricing_path)
        forecast_hours = trained_prediction_hours(settings)
        forecasts = _forecast_or_error(forecast_hours)
        recommendations = _recommendations(
            forecast_hours, request.headroom, inventory, pricing, forecasts
        )
        projection = inventory_costs(
            inventory,
            pricing,
            request.days,
            recommendations=recommendations,
        )
        sustainability = estimate_sustainability(recommendations, settings, request.days * 24)
        projection["sustainability"] = sustainability
        projection["forecast_basis"] = (
            f"Peak q90 utilization from a {forecast_hours}-hour forecast; "
            "the selected plans and prices are held constant over the cost horizon."
        )
        return projection
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise _http_error(exc) from exc


@router.post("/scenarios/simulate")
def simulate_what_if(request: ScenarioRequest) -> dict[str, Any]:
    try:
        values = _scenario_values(request)
        inventory = load_inventory(settings.inventory_path)
        pricing = load_pricing(settings.pricing_path)
        forecast_hours = trained_prediction_hours(settings)
        forecasts = _forecast_or_error(forecast_hours)
        recommendations = _recommendations(
            forecast_hours, request.headroom, inventory, pricing, forecasts
        )
        result = simulate_scenario(
            inventory,
            pricing,
            recommendations,
            request.days,
            values["environment"],
            request.resource_ids,
            values["shutdown_start"],
            values["shutdown_end"],
            request.weekdays_only,
            request.use_recommended_plans,
        )
        selected_ids = set(result["scenario"]["resource_ids"])
        start = values["shutdown_start"]
        end = values["shutdown_end"]
        shutdown_hours = (
            ((end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)) % (24 * 60)
        ) / 60
        sustainability = estimate_sustainability(
            recommendations,
            settings,
            request.days * 24,
            scheduled_shutdown_resource_ids=selected_ids,
            shutdown_hours_per_day=shutdown_hours,
            weekdays_only=request.weekdays_only,
            days=request.days,
        )
        result["sustainability"] = sustainability
        result["scenario"]["shutdown_start"] = values["shutdown_start"].isoformat(
            timespec="minutes"
        )
        result["scenario"]["shutdown_end"] = values["shutdown_end"].isoformat(timespec="minutes")
        return result
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise _http_error(exc) from exc


@router.post("/terraform")
def terraform_for_resource(request: TerraformRequest) -> dict[str, str]:
    try:
        inventory = load_inventory(settings.inventory_path)
        resource = inventory[inventory["resource_id"].astype(str) == request.resource_id]
        if resource.empty:
            raise ValueError(f"Unknown resource_id: {request.resource_id}")
        pricing = load_pricing(settings.pricing_path)
        result = generate_vm_terraform(
            request.resource_id,
            request.plan,
            pricing.to_dict(orient="records"),
            request.region or settings.default_region,
            request.ami_id,
            request.subnet_id,
        )
        return result
    except (FileNotFoundError, ValueError) as exc:
        raise _http_error(exc) from exc


@router.post("/descriptions")
def describe_recommendation(request: DescriptionRequest) -> dict[str, str]:
    try:
        description = generate_description(settings, request.title, request.facts)
        return {"description": description, "model": settings.google_model}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/catalog")
def get_pricing_catalog() -> dict[str, Any]:
    try:
        pricing = load_pricing(settings.pricing_path)
        return {"count": len(pricing), "plans": pricing.to_dict(orient="records")}
    except (FileNotFoundError, ValueError) as exc:
        raise _http_error(exc) from exc

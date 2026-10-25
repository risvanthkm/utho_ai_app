"""Versioned HTTP endpoints for forecasting and optimization."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.data import (
    inventory_records,
    load_configured_inventory,
    load_configured_pricing,
)
from app.schemas import (
    ChatRequest,
    CostForecastRequest,
    DashboardRecommendationRequest,
    DescriptionRequest,
    ForecastRequest,
    JevRiskRequest,
    RecommendationRequest,
    ScenarioRequest,
    TerraformRequest,
    TrainRequest,
)
from app.services.chatbot import chat, conversation_history
from app.services.dashboard import (
    actionable_recommendations,
    dashboard_alerts,
    dashboard_resources,
    dashboard_summary,
    resource_history,
)
from app.services.descriptions import generate_description
from app.services.forecasting import (
    forecast_usage,
    train_models,
    trained_prediction_hours,
)
from app.services.jev import assess_recommendation
from app.services.optimization import build_recommendations, inventory_costs
from app.services.scenarios import parse_scenario_text, simulate_scenario
from app.services.smartcloud_ai import (
    correlate_alerts_payload,
    explain_resource_payload,
    forecast_costs_and_usage_payload,
    generate_role_briefing_payload,
    parse_nl_scenario_payload,
)
from app.services.open_risk import assess_open_risk
from app.services.sustainability import estimate_sustainability
from app.services.terraform import generate_vm_terraform

router = APIRouter(prefix="/api/v1")
ai_router = APIRouter()


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


@ai_router.post("/chat")
@ai_router.post("/ai/chat")
@ai_router.post("/api/v1/chat")
@ai_router.post("/api/v1/ai/chat")
@router.post("/chat")
@router.post("/ai/chat")
def chat_with_resource_advisor(request: ChatRequest) -> dict[str, Any]:
    try:
        return chat(settings, request.message, request.conversation_id)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise _http_error(exc) from exc


@ai_router.get("/chat/{conversation_id}")
@ai_router.get("/ai/chat/{conversation_id}")
@ai_router.get("/api/v1/chat/{conversation_id}")
@ai_router.get("/api/v1/ai/chat/{conversation_id}")
@router.get("/chat/{conversation_id}")
@router.get("/ai/chat/{conversation_id}")
def get_chat_history(conversation_id: str) -> dict[str, Any]:
    try:
        return conversation_history(settings, conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise _http_error(exc) from exc


@router.get("/resources")
def list_resources() -> dict[str, Any]:
    try:
        inventory = load_configured_inventory(settings)
        return {"count": len(inventory), "resources": inventory_records(inventory)}
    except (FileNotFoundError, ValueError) as exc:
        raise _http_error(exc) from exc


@router.get("/dashboard/summary")
def get_dashboard_summary(
    hours: int = Query(default=720, ge=1, le=2160),
    region: str | None = None,
    environment: str | None = None,
    team: str | None = None,
) -> dict[str, Any]:
    try:
        return dashboard_summary(settings, hours, region, environment, team)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise _http_error(exc) from exc


@router.get("/dashboard/resources")
def get_dashboard_resources(
    hours: int = Query(default=720, ge=1, le=2160),
    region: str | None = None,
    environment: str | None = None,
    team: str | None = None,
) -> dict[str, Any]:
    try:
        return dashboard_resources(settings, hours, region, environment, team)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise _http_error(exc) from exc


@router.get("/resources/{resource_id}/history")
def get_resource_history(
    resource_id: str,
    hours: int = Query(default=168, ge=1, le=2160),
    interval_minutes: int = Query(default=60),
) -> dict[str, Any]:
    try:
        return resource_history(settings, resource_id, hours, interval_minutes)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise _http_error(exc) from exc


@router.post("/optimization/actionable")
def get_actionable_recommendations(
    request: DashboardRecommendationRequest,
) -> dict[str, Any]:
    try:
        return actionable_recommendations(
            settings,
            request.hours,
            request.peak_percentile,
            request.headroom,
            request.region,
            request.environment,
            request.team,
        )
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise _http_error(exc) from exc


@ai_router.post("/optimization/jev-risk")
@ai_router.post("/optimization/open-risk")
@ai_router.post("/api/v1/optimization/open-risk")
@router.post("/optimization/open-risk")
@router.post("/optimization/jev-risk")
def get_jev_risk_assessment(request: JevRiskRequest) -> dict[str, Any]:
    try:
        recommendations = actionable_recommendations(
            settings,
            request.hours,
            request.peak_percentile,
            request.headroom,
        )["recommendations"]
        resource_recommendations = [
            item for item in recommendations if item["resource_id"] == request.resource_id
        ]
        if resource_recommendations:
            primary = resource_recommendations[0]
        else:
            inventory = load_configured_inventory(settings)
            match_inv = inventory[inventory["resource_id"].astype(str) == request.resource_id]
            if match_inv.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"Resource {request.resource_id} not found.",
                )
            row = match_inv.iloc[0].to_dict()
            primary = {
                "resource_id": request.resource_id,
                "action": "maintain",
                "status": "OPTIMAL",
                "current_plan": str(row.get("plan", "c4.medium")),
                "recommended_plan": str(row.get("plan", "c4.medium")),
                "peak_cpu_pct": 24.0,
                "peak_cpu_p99": 36.0,
            }

        if settings.typesafe_api_key:
            try:
                assessment = assess_recommendation(settings, primary)
            except Exception:
                assessment = assess_open_risk(primary)
        else:
            assessment = assess_open_risk(primary)

        return {
            "resource_id": request.resource_id,
            "deterministic_recommendation": primary,
            "risk_assessment": assessment,
            "jev_assessment": assessment,
        }
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise _http_error(exc) from exc


@router.get("/alerts")
def get_dashboard_alerts(
    hours: int = Query(default=720, ge=1, le=2160),
    region: str | None = None,
    environment: str | None = None,
    team: str | None = None,
    monthly_budget_usd: float | None = Query(default=None, ge=0),
    previous_month_cost_usd: float | None = Query(default=None, ge=0),
) -> dict[str, Any]:
    try:
        return dashboard_alerts(
            settings,
            hours,
            region,
            environment,
            team,
            monthly_budget_usd,
            previous_month_cost_usd,
        )
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
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
        inventory = load_configured_inventory(settings)
        pricing = load_configured_pricing(settings)
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
        inventory = load_configured_inventory(settings)
        pricing = load_configured_pricing(settings)
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
        inventory = load_configured_inventory(settings)
        pricing = load_configured_pricing(settings)
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
        inventory = load_configured_inventory(settings)
        resource = inventory[inventory["resource_id"].astype(str) == request.resource_id]
        if resource.empty:
            raise ValueError(f"Unknown resource_id: {request.resource_id}")
        pricing = load_configured_pricing(settings)
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
        pricing = load_configured_pricing(settings)
        return {"count": len(pricing), "plans": pricing.to_dict(orient="records")}
    except (FileNotFoundError, ValueError) as exc:
        raise _http_error(exc) from exc


# ===========================================================================
# SmartCloud Unified AI Orchestration Endpoints (:8001)
# ===========================================================================


@ai_router.post("/ai/explain-resource")
@router.post("/ai/explain-resource")
def explain_resource_endpoint(
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    return explain_resource_payload(payload)


@router.post("/ai/parse-scenario")
def parse_scenario_endpoint(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return parse_nl_scenario_payload(payload)


@router.post("/ai/correlate-alerts")
def correlate_alerts_endpoint(
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    return correlate_alerts_payload(payload)


@router.post("/ai/forecast")
def ai_forecast_endpoint(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return forecast_costs_and_usage_payload(payload)


@router.post("/ai/briefing")
def ai_briefing_endpoint(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return generate_role_briefing_payload(payload)

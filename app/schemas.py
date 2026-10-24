"""Validated request and response models for the API."""

from __future__ import annotations

from datetime import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainRequest(APIModel):
    aggregation_hours: int = Field(default=4, ge=1, le=24)
    epochs: int = Field(default=5, ge=1, le=100)
    batch_size: int = Field(default=64, ge=1, le=1024)
    prediction_hours: int = Field(default=24, ge=4, le=168)


class ForecastRequest(APIModel):
    hours: int = Field(default=24, ge=4, le=168)


class RecommendationRequest(APIModel):
    hours: int = Field(default=24, ge=4, le=168)
    headroom: float = Field(default=1.30, ge=1.0, le=3.0)


class DashboardRecommendationRequest(APIModel):
    hours: int = Field(default=24 * 30, ge=1, le=24 * 90)
    peak_percentile: Literal[95, 99] = 99
    headroom: float = Field(default=1.30, ge=1.0, le=3.0)
    region: str | None = None
    environment: str | None = None
    team: str | None = None


class JevRiskRequest(APIModel):
    resource_id: str = Field(min_length=1, max_length=128)
    hours: int = Field(default=24 * 30, ge=1, le=24 * 90)
    peak_percentile: Literal[95, 99] = 99
    headroom: float = Field(default=1.30, ge=1.0, le=3.0)


class CostForecastRequest(APIModel):
    days: Literal[30, 60, 90] = 30
    headroom: float = Field(default=1.30, ge=1.0, le=3.0)


class ScenarioRequest(APIModel):
    days: int = Field(default=30, ge=1, le=365)
    environment: str | None = None
    resource_ids: list[str] | None = None
    shutdown_start: time | None = None
    shutdown_end: time | None = None
    weekdays_only: bool = False
    use_recommended_plans: bool = True
    headroom: float = Field(default=1.30, ge=1.0, le=3.0)
    scenario_text: str | None = Field(default=None, max_length=500)


class TerraformRequest(APIModel):
    resource_id: str = Field(min_length=1, max_length=128)
    plan: str = Field(min_length=1, max_length=64)
    region: str | None = None
    ami_id: str = Field(min_length=1, max_length=128)
    subnet_id: str = Field(min_length=1, max_length=128)


class DescriptionRequest(APIModel):
    title: str = Field(min_length=1, max_length=120)
    facts: dict[str, str | int | float | bool | None]


class SustainabilitySummary(BaseModel):
    estimated_energy_saved_kwh: float
    estimated_carbon_saved_kg_co2e: float
    sustainability_score: float = Field(ge=0, le=100)
    assumptions: list[str]


class ResourceRecommendation(BaseModel):
    resource_id: str
    current_plan: str
    recommended_plan: str | None
    action: Literal["DOWNSIZE", "UPSCALE", "MAINTAIN", "NO_CATALOG_MATCH"]
    estimated_cost_current: float
    estimated_cost_recommended: float | None
    estimated_savings: float | None
    peak_cpu_q90_pct: float | None
    peak_memory_q90_pct: float | None
    schedule: dict[str, str] | None
    reason: str

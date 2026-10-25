"""Strict Pydantic v2 standardized schemas for FinSort telemetry and metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StandardMetricRecord(BaseModel):
    """Single telemetry metric observation adhering to FinSort standardized schema."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    resource_id: str = Field(..., min_length=1, max_length=128, description="Target resource identifier, e.g. res-vm-001")
    ts: datetime = Field(..., description="UTC-aware observation timestamp", alias="timestamp")
    cpu_utilization_pct: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage [0-100%]")
    memory_utilization_pct: float = Field(..., ge=0.0, le=100.0, description="RAM usage percentage [0-100%]")
    disk_utilization_pct: float = Field(default=25.0, ge=0.0, le=100.0, description="Disk volume usage percentage [0-100%]")
    disk_iops: float = Field(default=0.0, ge=0.0, description="Disk operations per second")
    network_in_mbps: float = Field(default=0.0, ge=0.0, description="Inbound network throughput in megabits per second")
    network_out_mbps: float = Field(default=0.0, ge=0.0, description="Outbound network throughput in megabits per second")

    @field_validator("ts", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError(f"Invalid timestamp format: {v}")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


class DailyAggregateRecord(BaseModel):
    """Daily continuous aggregate metrics for forecasting and reporting."""

    model_config = ConfigDict(extra="forbid")

    resource_id: str
    day: datetime
    cpu_avg: float = Field(..., ge=0.0, le=100.0)
    cpu_p95: float = Field(..., ge=0.0, le=100.0)
    cpu_p99: float = Field(..., ge=0.0, le=100.0)
    mem_avg: float = Field(..., ge=0.0, le=100.0)
    mem_p95: float = Field(..., ge=0.0, le=100.0)
    mem_p99: float = Field(..., ge=0.0, le=100.0)
    sample_count: int = Field(..., ge=1)

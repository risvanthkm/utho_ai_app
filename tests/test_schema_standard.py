"""Test strict Pydantic v2 standardized metrics schema."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from app.models.schemas import StandardMetricRecord, DailyAggregateRecord
from app.generate_telemetry import generate_synthesized_metrics


def test_standard_metric_record_valid():
    rec = StandardMetricRecord(
        resource_id="res-vm-001",
        ts=datetime.now(timezone.utc),
        cpu_utilization_pct=45.5,
        memory_utilization_pct=60.2,
        disk_utilization_pct=30.0,
        disk_iops=120.0,
        network_in_mbps=12.5,
        network_out_mbps=8.4,
    )
    assert rec.resource_id == "res-vm-001"
    assert rec.cpu_utilization_pct == 45.5
    assert rec.ts.tzinfo is not None


def test_standard_metric_record_extra_forbidden():
    with pytest.raises(ValidationError):
        StandardMetricRecord(
            resource_id="res-vm-001",
            ts=datetime.now(timezone.utc),
            cpu_utilization_pct=45.5,
            memory_utilization_pct=60.2,
            unknown_field=123,  # Forbidden by extra="forbid"
        )


def test_standard_metric_record_out_of_bounds():
    with pytest.raises(ValidationError):
        StandardMetricRecord(
            resource_id="res-vm-001",
            ts=datetime.now(timezone.utc),
            cpu_utilization_pct=150.0,  # Invalid: > 100
            memory_utilization_pct=60.2,
        )


def test_synthesizer_emits_valid_schema():
    df = generate_synthesized_metrics()
    assert len(df) > 0
    assert "resource_id" in df.columns
    assert "ts" in df.columns
    assert "cpu_utilization_pct" in df.columns
    assert "memory_utilization_pct" in df.columns
    # Check types
    assert df["cpu_utilization_pct"].dtype in ("float64", "float32")
    assert df["memory_utilization_pct"].dtype in ("float64", "float32")

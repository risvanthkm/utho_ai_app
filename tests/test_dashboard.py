from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.config import Settings
from app.main import app
from app.services.dashboard import (
    actionable_recommendations,
    dashboard_alerts,
    dashboard_resources,
    dashboard_summary,
    resource_history,
)
from app.services.jev import assess_recommendation


@pytest.fixture
def dashboard_settings(tmp_path):
    inventory = pd.DataFrame(
        [
            {
                "resource_id": "vm-busy",
                "type": "VM",
                "plan": "small",
                "region": "region-a",
                "environment": "prod",
                "team": "payments",
                "vcpu": 2,
                "ram_gb": 4,
                "disk_size_gb": 40,
                "attached_to": "host-a",
            },
            {
                "resource_id": "vm-idle",
                "type": "VM",
                "plan": "small",
                "region": "region-a",
                "environment": "dev",
                "team": "payments",
                "vcpu": 2,
                "ram_gb": 4,
                "disk_size_gb": 40,
                "attached_to": "host-b",
            },
            {
                "resource_id": "volume-orphan",
                "type": "block_volume",
                "plan": "gp3",
                "region": "region-b",
                "environment": "dev",
                "team": "platform",
                "vcpu": 0,
                "ram_gb": 0,
                "disk_size_gb": 100,
                "attached_to": "",
            },
        ]
    )
    inventory_path = tmp_path / "inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    pd.DataFrame(
        [
            {
                "plan": "small",
                "type": "VM",
                "vcpu": 2,
                "ram_gb": 4,
                "rate": 0.10,
                "storage_rate": 0.0,
            },
            {
                "plan": "medium",
                "type": "VM",
                "vcpu": 4,
                "ram_gb": 8,
                "rate": 0.20,
                "storage_rate": 0.0,
            },
            {
                "plan": "gp3",
                "type": "block_volume",
                "vcpu": 0,
                "ram_gb": 0,
                "rate": 0.0,
                "storage_rate": 0.08,
            },
        ]
    ).to_csv(tmp_path / "pricing.csv", index=False)
    timestamps = [datetime(2026, 1, 1) + timedelta(minutes=5 * index) for index in range(864)]
    rows = []
    for index, timestamp in enumerate(timestamps):
        busy = index >= 852 or index % 37 == 0
        for resource_id in ("vm-busy", "vm-idle"):
            idle = resource_id == "vm-idle"
            rows.append(
                {
                    "resource_id": resource_id,
                    "timestamp": timestamp,
                    "cpu_utilization_pct": 0 if idle else (95 if busy else 20),
                    "memory_utilization_pct": 0 if idle else (90 if busy else 40),
                    "disk_utilization_pct": 0 if idle else 30,
                    "disk_iops": 0 if idle else 15,
                    "network_in_mbps": 0 if idle else (20 if busy else 3),
                    "network_out_mbps": 0 if idle else (25 if busy else 4),
                }
            )
    telemetry_path = tmp_path / "telemetry.parquet"
    pd.DataFrame(rows).to_parquet(telemetry_path, index=False)
    return Settings(
        inventory_path=inventory_path,
        pricing_path=tmp_path / "pricing.csv",
        telemetry_path=telemetry_path,
        model_dir=tmp_path / "models",
        default_region="us-east-1",
        google_api_key=None,
        google_model="gemini-test",
        watts_per_vcpu_at_idle=5,
        watts_per_vcpu_at_full_load=15,
        power_usage_effectiveness=1.4,
        grid_carbon_kg_per_kwh=0.4,
    )


def test_dashboard_filters_and_exposes_disk_network_percentiles(dashboard_settings):
    result = dashboard_resources(
        dashboard_settings,
        hours=72,
        region="REGION-A",
        environment="prod",
        team="PAYMENTS",
    )
    assert result["count"] == 1
    resource = result["resources"][0]
    assert resource["resource_id"] == "vm-busy"
    assert resource["metrics"]["disk_utilization_pct"]["p99"] == 30
    assert resource["metrics"]["network_in_mbps"]["unit"] == "Mbps"
    assert resource["metrics"]["cpu_utilization_pct"]["p99"] > 90


def test_resource_history_returns_downsampled_metric_series(dashboard_settings):
    result = resource_history(dashboard_settings, "vm-busy", hours=24, interval_minutes=60)
    assert result["interval_minutes"] == 60
    assert len(result["points"]) <= 25
    assert "network_out_mbps" in result["points"][0]["metrics"]


def test_actionable_recommendations_use_p99_and_find_idle_and_orphans(dashboard_settings):
    result = actionable_recommendations(
        dashboard_settings, hours=72, peak_percentile=99, headroom=1.3
    )
    by_id = {}
    for recommendation in result["recommendations"]:
        by_id.setdefault(recommendation["resource_id"], []).append(recommendation)
    busy = by_id["vm-busy"][0]
    assert busy["action"] == "resize_instance"
    assert busy["recommended_plan"] == "medium"
    assert busy["peak_cpu_pct"] > 90
    assert busy["risk_percentage"] == max(
        busy["target_peak_cpu_pct"], busy["target_peak_memory_pct"], busy["peak_disk_pct"]
    )
    assert by_id["vm-idle"][0]["action"] == "stop_idle_resource"
    orphan = by_id["volume-orphan"][0]
    assert orphan["action"] == "clean_up_orphan"
    assert orphan["monthly_savings_estimate"] == 8


def test_dashboard_summary_and_alerts_explain_thresholds(dashboard_settings):
    summary = dashboard_summary(dashboard_settings, hours=72, region="region-a")
    assert summary["resources_with_telemetry"] == 2
    alerts = dashboard_alerts(
        dashboard_settings,
        hours=72,
        region="region-a",
        monthly_budget_usd=1,
        previous_month_cost_usd=1,
    )
    kinds = {alert["type"] for alert in alerts["alerts"]}
    assert "capacity_limit" in kinds
    assert "usage_spike" in kinds
    assert "projected_budget_overrun" in kinds
    assert "cost_increase" in kinds
    assert all(alert.get("reason") for alert in alerts["alerts"])


def test_jev_risk_is_a_scored_assessment_and_not_failure_probability(
    dashboard_settings, monkeypatch
):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "jev-test",
                "answers": {
                    "recommendation_risk": {
                        "score": 1.5,
                        "confidence": 0.7,
                        "probabilities": {"0": 0.1, "1": 0.6, "2": 0.2, "3": 0.1},
                    }
                },
            }

    def fake_post(url, headers, json, timeout):
        assert url == "https://api.typesafe.ai/v1/systemone"
        assert headers["Authorization"] == "Bearer test-token"
        assert "resource_id" not in json["state"]
        return FakeResponse()

    monkeypatch.setattr("app.services.jev.httpx.post", fake_post)
    configured = replace(
        dashboard_settings,
        typesafe_api_key="test-token",
        typesafe_model="jev-test",
    )
    assessment = assess_recommendation(
        configured,
        {
            "resource_id": "private-id",
            "action": "resize_instance",
            "risk_percentage": 40,
            "risk_level": "LOW",
        },
    )
    assert assessment["risk_score_percentage"] == 50
    assert assessment["model_confidence"] == 0.7
    assert "not a measured probability" in assessment["explanation"]


def test_jev_requires_explicit_api_key(dashboard_settings):
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        assess_recommendation(dashboard_settings, {"action": "maintain"})


def test_dashboard_http_routes_expose_filtered_views(dashboard_settings, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr("app.api.routes.settings", dashboard_settings)
    client = TestClient(app)

    summary = client.get(
        "/api/v1/dashboard/summary",
        params={"hours": 72, "region": "region-a", "team": "payments"},
    )
    assert summary.status_code == 200
    assert summary.json()["resource_count"] == 2

    history = client.get(
        "/api/v1/resources/vm-busy/history",
        params={"hours": 24, "interval_minutes": 60},
    )
    assert history.status_code == 200
    assert history.json()["points"]

    recommendations = client.post(
        "/api/v1/optimization/actionable",
        json={"hours": 72, "peak_percentile": 99, "headroom": 1.3},
    )
    assert recommendations.status_code == 200
    assert recommendations.json()["count"] >= 3

    alerts = client.get(
        "/api/v1/alerts",
        params={"hours": 72, "region": "region-a", "monthly_budget_usd": 1},
    )
    assert alerts.status_code == 200
    assert alerts.json()["count"] > 0

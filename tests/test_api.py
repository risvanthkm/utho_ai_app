from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    response = TestClient(app).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cost_horizon_is_limited_to_documented_options():
    response = TestClient(app).post("/api/v1/costs/forecast", json={"days": 45})
    assert response.status_code == 422



def test_smartcloud_root_health_and_ai_endpoints():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["service"] == "smartcloud-ai-engine"
    assert "sre_resource_audit" in body["features"]

    explain = client.post(
        "/ai/explain-resource",
        json={
            "resource": {
                "id": "res-vm-021",
                "name": "prod-payment-webhook-01",
                "type": "vm",
                "plan": "m8.xlarge",
                "region": "in-mumbai-1",
                "env": "prod",
                "vcpu": 8,
                "ram_gb": 32,
                "monthly_cost_inr": 9600,
                "optimization_status": "PROTECTED_PEAK_LOAD",
                "metrics_summary": {
                    "cpu_avg": 13.8,
                    "cpu_p95": 86.4,
                    "cpu_p99": 93.2,
                    "mem_p95": 68.0,
                },
            }
        },
    )
    assert explain.status_code == 200
    exp_json = explain.json()
    assert exp_json["downsizing_safe"] is False
    assert "jev_assessment" in exp_json
    assert len(exp_json["quantile_forecast_24h"]) == 6
    assert len(exp_json["rollback_runbook"]) >= 2

    scenario = client.post(
        "/api/v1/ai/parse-scenario",
        json={
            "query": "What if we shut down all dev VMs in Bangalore from 8 PM to 8 AM and keep 30% peak headroom?"
        },
    )
    assert scenario.status_code == 200
    scen_json = scenario.json()
    assert scen_json["parsed_params"]["safety_headroom_buffer_pct"] == 30.0
    assert scen_json["detected_scope"]["region"] == "in-bangalore-1"

    alerts = client.post("/api/v1/ai/correlate-alerts", json={})
    assert alerts.status_code == 200
    assert alerts.json()["count"] >= 3
    assert alerts.json()["alerts"][0]["time_to_failure_hours"] > 0

    forecast = client.post("/api/v1/ai/forecast", json={"days": 60, "headroom": 1.25})
    assert forecast.status_code == 200
    assert forecast.json()["days"] == 60
    assert forecast.json()["currency"] == "INR"

    briefing = client.post("/api/v1/ai/briefing", json={"role": "engineering"})
    assert briefing.status_code == 200
    assert briefing.json()["role"] == "engineering"
    assert len(briefing.json()["bullet_points"]) == 3

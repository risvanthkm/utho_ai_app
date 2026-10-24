from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    response = TestClient(app).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cost_horizon_is_limited_to_documented_options():
    response = TestClient(app).post("/api/v1/costs/forecast", json={"days": 45})
    assert response.status_code == 422

import pandas as pd
import pytest

from app.services.optimization import build_recommendations, inventory_costs


@pytest.fixture
def catalog():
    return pd.DataFrame(
        [
            {"plan": "tiny", "type": "VM", "vcpu": 2, "ram_gb": 2, "rate": 0.02, "storage_rate": 0},
            {
                "plan": "medium",
                "type": "VM",
                "vcpu": 4,
                "ram_gb": 8,
                "rate": 0.08,
                "storage_rate": 0,
            },
            {
                "plan": "large",
                "type": "VM",
                "vcpu": 8,
                "ram_gb": 16,
                "rate": 0.16,
                "storage_rate": 0,
            },
            {
                "plan": "disk",
                "type": "block_volume",
                "vcpu": 0,
                "ram_gb": 0,
                "rate": 0,
                "storage_rate": 0.08,
            },
            {
                "plan": "ip",
                "type": "public_ip",
                "vcpu": 0,
                "ram_gb": 0,
                "rate": 0.005,
                "storage_rate": 0,
            },
        ]
    )


@pytest.fixture
def inventory():
    return pd.DataFrame(
        [
            {
                "resource_id": "vm-a",
                "type": "VM",
                "plan": "large",
                "vcpu": 8,
                "ram_gb": 16,
                "disk_size_gb": 100,
                "environment": "dev",
                "region": "test",
            },
            {
                "resource_id": "vol-a",
                "type": "block_volume",
                "plan": "disk",
                "vcpu": 0,
                "ram_gb": 0,
                "disk_size_gb": 100,
                "environment": "dev",
                "region": "test",
            },
            {
                "resource_id": "ip-a",
                "type": "public_ip",
                "plan": "ip",
                "vcpu": 0,
                "ram_gb": 0,
                "disk_size_gb": 0,
                "environment": "dev",
                "region": "test",
            },
        ]
    )


def test_forecast_downsize_and_upsize(catalog):
    inventory = pd.DataFrame(
        [
            {
                "resource_id": "small-load",
                "type": "VM",
                "plan": "large",
                "vcpu": 8,
                "ram_gb": 16,
                "environment": "prod",
                "region": "test",
            },
            {
                "resource_id": "high-load",
                "type": "VM",
                "plan": "tiny",
                "vcpu": 2,
                "ram_gb": 2,
                "environment": "prod",
                "region": "test",
            },
        ]
    )
    forecasts = [
        {"resource_id": "small-load", "target": "cpu", "q90": 10},
        {"resource_id": "small-load", "target": "memory", "q90": 10},
        {"resource_id": "high-load", "target": "cpu", "q90": 95},
        {"resource_id": "high-load", "target": "memory", "q90": 90},
    ]
    result = build_recommendations(inventory, catalog, forecasts, 24, 1.3)
    assert [row["action"] for row in result] == ["DOWNSIZE", "UPSCALE"]
    assert result[0]["recommended_plan"] == "medium"
    assert result[1]["recommended_plan"] == "medium"


def test_missing_forecast_keeps_plan(catalog, inventory):
    result = build_recommendations(inventory, catalog, [], 24, 1.3)
    vm = next(row for row in result if row["resource_id"] == "vm-a")
    assert vm["action"] == "MAINTAIN"
    assert vm["recommended_plan"] is None


def test_costs_split_compute_storage_and_ip(catalog, inventory):
    result = inventory_costs(inventory, catalog, 30)
    assert result["current_cost"]["vm_compute"] == 115.2
    assert result["current_cost"]["storage"] == 8.0
    assert result["current_cost"]["public_ip"] == 3.6
    assert result["current_cost"]["total"] == 126.8


def test_shutdown_schedule_saves_only_selected_vm_hours(catalog, inventory):
    result = inventory_costs(
        inventory,
        catalog,
        30,
        shutdown_start_minutes=20 * 60,
        shutdown_end_minutes=8 * 60,
        shutdown_resource_ids={"vm-a"},
    )
    assert result["current_cost"]["vm_compute"] == pytest.approx(57.6)
    assert result["current_cost"]["storage"] == 8.0
    assert result["current_cost"]["public_ip"] == 3.6

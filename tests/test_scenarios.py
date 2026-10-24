from datetime import time

import pandas as pd
import pytest

from app.services.scenarios import parse_scenario_text, simulate_scenario


def test_parse_shutdown_question():
    result = parse_scenario_text("What if we shut down all dev VMs from 8 PM to 8 AM?")
    assert result == {
        "environment": "dev",
        "shutdown_start": time(20, 0),
        "shutdown_end": time(8, 0),
    }


def test_reject_scenario_without_environment_or_time():
    with pytest.raises(ValueError, match="identify a resource environment"):
        parse_scenario_text("Stop instances from 8 PM to 8 AM")
    with pytest.raises(ValueError, match="shutdown time range"):
        parse_scenario_text("Shut down dev VMs overnight")


def test_simulate_dev_shutdown_and_recommended_plan():
    inventory = pd.DataFrame(
        [
            {
                "resource_id": "dev-1",
                "type": "VM",
                "plan": "large",
                "vcpu": 8,
                "ram_gb": 16,
                "disk_size_gb": 100,
                "environment": "dev",
            },
            {
                "resource_id": "prod-1",
                "type": "VM",
                "plan": "large",
                "vcpu": 8,
                "ram_gb": 16,
                "disk_size_gb": 100,
                "environment": "prod",
            },
        ]
    )
    pricing = pd.DataFrame(
        [
            {"plan": "tiny", "type": "VM", "vcpu": 2, "ram_gb": 2, "rate": 0.02, "storage_rate": 0},
            {
                "plan": "large",
                "type": "VM",
                "vcpu": 8,
                "ram_gb": 16,
                "rate": 0.16,
                "storage_rate": 0,
            },
        ]
    )
    recommendations = [
        {"resource_id": "dev-1", "recommended_plan": "tiny"},
        {"resource_id": "prod-1", "recommended_plan": "large"},
    ]
    result = simulate_scenario(
        inventory,
        pricing,
        recommendations,
        30,
        "dev",
        None,
        time(20),
        time(8),
        False,
        True,
    )
    assert result["scenario"]["resource_ids"] == ["dev-1"]
    assert result["baseline"]["current_cost"]["total"] == 230.4
    assert result["scenario_cost"]["recommended_cost"]["total"] == 122.4
    assert result["estimated_savings_vs_baseline"] == 108.0

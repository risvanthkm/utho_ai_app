"""Empirical challenger stress-tests for NL scenario parser and fleet/catalog grounding."""
import pytest
from app.services.smartcloud_ai import parse_nl_scenario_payload, UTHO_CATALOG


def test_adversarial_empty_and_garbage_payload():
    """Verify empty, malformed, or garbage input does not crash the service."""
    # Empty dictionary
    res1 = parse_nl_scenario_payload({})
    assert isinstance(res1, dict)
    assert "parsed_params" in res1
    assert res1["parsed_params"]["dev_off_hours_enabled"] is True

    # None values
    res2 = parse_nl_scenario_payload({"query": None, "fleet_summary": None, "catalog": None})
    assert isinstance(res2, dict)
    assert res2["parsed_params"]["safety_headroom_buffer_pct"] == 25.0

    # Garbage strings
    res3 = parse_nl_scenario_payload({"query": "!@#$%^&*()_+{}[]:;<>?,./~`"})
    assert isinstance(res3, dict)
    assert res3["parsed_params"]["dev_shutdown_hours_per_week"] == 118


def test_grounded_fleet_and_catalog_in_payload():
    """Verify scenario parser accepts custom fleet summary and pricing catalog."""
    custom_fleet = {
        "total_resources": 100,
        "vm_count": 80,
        "block_volume_count": 15,
        "public_ip_count": 5,
        "monthly_baseline_inr": 850000.0,
        "regions": ["in-noida-1", "in-mumbai-1"],
        "envs": ["prod", "dev"],
    }
    custom_catalog = {
        "c2.small": {"monthly_inr": 1000.0, "vcpu": 1, "ram_gb": 2},
        "m8.xlarge": {"monthly_inr": 10000.0, "vcpu": 8, "ram_gb": 32},
    }
    payload = {
        "query": "Shutdown dev VMs in Noida from 9 PM to 9 AM and keep 35% peak headroom",
        "fleet_summary": custom_fleet,
        "catalog": custom_catalog,
    }
    res = parse_nl_scenario_payload(payload)
    assert isinstance(res, dict)
    assert res["parsed_params"]["safety_headroom_buffer_pct"] == 35.0
    assert "in-noida-1" in res["detected_scope"]["region"] or res["detected_scope"]["region"] == "all"
    assert res["parsed_params"]["dev_off_hours_enabled"] is True
    # 9 PM to 9 AM is 12 hours/day -> 12*5 + 48 = 108 hrs/wk
    assert res["parsed_params"]["dev_shutdown_hours_per_week"] == 108


def test_adversarial_queries_headroom_bounds():
    """Verify headroom buffer is clamped within [10.0, 60.0]% safe envelope."""
    # Query specifying 5% (below min 10%)
    res_low = parse_nl_scenario_payload({"query": "Keep 5% peak headroom buffer"})
    assert res_low["parsed_params"]["safety_headroom_buffer_pct"] == 10.0

    # Query specifying 90% (above max 60%)
    res_high = parse_nl_scenario_payload({"query": "Keep 90% peak headroom safety margin"})
    assert res_high["parsed_params"]["safety_headroom_buffer_pct"] == 60.0


def test_disable_dev_and_staging_keywords():
    """Verify keywords that disable dev or staging off-hours are recognized."""
    # Dev running 24/7
    res_247 = parse_nl_scenario_payload({"query": "Keep dev running 24/7 with no shutdown"})
    assert res_247["parsed_params"]["dev_off_hours_enabled"] is False
    assert res_247["parsed_params"]["dev_shutdown_hours_per_week"] == 0

    # Disable staging weekend
    res_stage = parse_nl_scenario_payload({"query": "Disable staging weekend halt and keep 20% headroom"})
    assert res_stage["parsed_params"]["staging_weekend_off_enabled"] is False
    assert res_stage["parsed_params"]["staging_shutdown_hours_per_week"] == 0
    assert res_stage["parsed_params"]["safety_headroom_buffer_pct"] == 20.0


def test_profile_keywords():
    """Verify aggressive and conservative profile keywords."""
    res_agg = parse_nl_scenario_payload({"query": "Maximum savings aggressive rightsizing"})
    assert res_agg["parsed_params"]["safety_headroom_buffer_pct"] == 15.0
    assert res_agg["parsed_params"]["purge_orphaned_older_than_days"] == 15

    res_con = parse_nl_scenario_payload({"query": "Ultra-conservative zero risk profile"})
    assert res_con["parsed_params"]["safety_headroom_buffer_pct"] == 40.0

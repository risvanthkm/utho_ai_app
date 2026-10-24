import json
from pathlib import Path

from app.config import Settings
from app.services.forecasting import trained_prediction_hours
from app.services.sustainability import estimate_sustainability
from app.services.terraform import generate_vm_terraform


def test_terraform_is_parameterized_for_approved_inputs():
    output = generate_vm_terraform(
        "i-123",
        "t3.small",
        [{"plan": "t3.small", "type": "VM", "vcpu": 2, "ram_gb": 2}],
        "us-east-1",
        "ami-123",
        "subnet-456",
    )
    assert 'instance_type = "t3.small"' in output["terraform"]
    assert 'default     = "ami-123"' in output["terraform"]
    assert "does not migrate or destroy" in output["warning"]


def test_terraform_resource_label_sanitizes_dots_and_hyphens():
    output = generate_vm_terraform(
        "vm.prod-1",
        "t3.small",
        [{"plan": "t3.small", "type": "VM", "vcpu": 2, "ram_gb": 2}],
        "us-east-1",
        "ami-123",
        "subnet-456",
    )
    assert 'resource "aws_instance" "rightsized_vm_prod_1"' in output["terraform"]
    assert "aws_instance.rightsized_vm_prod_1.id" in output["terraform"]


def test_terraform_rejects_injected_resource_name():
    try:
        generate_vm_terraform(
            'vm"\nresource "aws_s3_bucket" "bad"',
            "t3.small",
            [{"plan": "t3.small", "type": "VM", "vcpu": 2, "ram_gb": 2}],
            "us-east-1",
            "ami-123",
            "subnet-456",
        )
    except ValueError as exc:
        assert "resource_id" in str(exc)
    else:
        raise AssertionError("Terraform resource name injection must be rejected.")


def test_sustainability_reports_reductions():
    settings = Settings(
        inventory_path=Path("inventory.csv"),
        pricing_path=Path("pricing.csv"),
        telemetry_path=Path("telemetry.parquet"),
        model_dir=Path("models"),
        default_region="us-east-1",
        google_api_key=None,
        google_model="gemini-test",
        watts_per_vcpu_at_idle=5,
        watts_per_vcpu_at_full_load=15,
        power_usage_effectiveness=1.4,
        grid_carbon_kg_per_kwh=0.4,
    )
    result = estimate_sustainability(
        [{"current_vcpu": 8, "recommended_vcpu": 2, "peak_cpu_q90_pct": 10}],
        settings,
        24,
    )
    assert result["estimated_energy_saved_kwh"] > 0
    assert result["estimated_carbon_saved_kg_co2e"] > 0
    assert 0 <= result["sustainability_score"] <= 100


def test_trained_prediction_hours_uses_model_manifest(tmp_path):
    settings = Settings(
        inventory_path=tmp_path / "inventory.csv",
        pricing_path=tmp_path / "pricing.csv",
        telemetry_path=tmp_path / "telemetry.parquet",
        model_dir=tmp_path,
        default_region="us-east-1",
        google_api_key=None,
        google_model="gemini-test",
        watts_per_vcpu_at_idle=5,
        watts_per_vcpu_at_full_load=15,
        power_usage_effectiveness=1.4,
        grid_carbon_kg_per_kwh=0.4,
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"prediction_hours": 48}), encoding="utf-8")

    assert trained_prediction_hours(settings) == 48

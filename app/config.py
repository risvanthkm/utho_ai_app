"""Environment-based application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_setting(name: str, default: Path) -> Path:
    configured = os.getenv(name)
    if configured is None:
        return default
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """File locations and optional integrations, configurable through environment."""

    inventory_path: Path
    pricing_path: Path
    telemetry_path: Path
    model_dir: Path
    default_region: str
    google_api_key: str | None
    google_model: str
    watts_per_vcpu_at_idle: float
    watts_per_vcpu_at_full_load: float
    power_usage_effectiveness: float
    grid_carbon_kg_per_kwh: float
    typesafe_api_key: str | None = None
    typesafe_model: str = "jev-latest"
    database_url: str | None = None

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            inventory_path=_path_setting(
                "INVENTORY_PATH", PROJECT_ROOT / "datasets" / "raw" / "inventory.csv"
            ),
            pricing_path=_path_setting(
                "PRICING_PATH", PROJECT_ROOT / "datasets" / "raw" / "pricing.csv"
            ),
            telemetry_path=_path_setting(
                "TELEMETRY_PATH",
                PROJECT_ROOT / "datasets" / "raw" / "telemetry.parquet",
            ),
            model_dir=_path_setting("MODEL_DIR", PROJECT_ROOT / "models" / "tft"),
            default_region=os.getenv("AWS_REGION", "us-east-1"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_model=os.getenv("GOOGLE_MODEL", "gemini-3.8-flash"),
            typesafe_api_key=os.getenv("TYPESAFE_API_KEY"),
            typesafe_model=os.getenv("TYPESAFE_MODEL", "jev-latest"),
            database_url=os.getenv("DATABASE_URL") or None,
            watts_per_vcpu_at_idle=float(os.getenv("WATTS_PER_VCPU_IDLE", "5")),
            watts_per_vcpu_at_full_load=float(os.getenv("WATTS_PER_VCPU_FULL", "15")),
            power_usage_effectiveness=float(os.getenv("DATA_CENTER_PUE", "1.4")),
            grid_carbon_kg_per_kwh=float(os.getenv("GRID_CARBON_KG_PER_KWH", "0.4")),
        )


settings = Settings.from_environment()

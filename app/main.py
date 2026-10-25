"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ai_router, router
from app.config import settings
from app.logger import RequestLoggingMiddleware
from app.services.smartcloud_ai import AI_FEATURES


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-seed TimescaleDB on container startup if database is connected and empty
    if settings.database_url:
        try:
            from app.database import connect
            from app.seed import _seed_inventory, _seed_pricing
            with connect(settings) as conn:
                row = conn.execute("SELECT count(*) as c FROM resource_inventory").fetchone()
                if row and row["c"] == 0:
                    if settings.inventory_path.is_file():
                        _seed_inventory(conn, settings.inventory_path)
                    if settings.pricing_path.is_file():
                        _seed_pricing(conn, settings.pricing_path)
                    conn.commit()
        except Exception as exc:
            print(f"Startup database bootstrap notice: {exc}")
    yield


app = FastAPI(
    title="Cloud Resource Advisor API",
    description=(
        "Forecast VM usage and cost, recommend safe capacity plans, simulate "
        "scheduled shutdowns, estimate sustainability, and generate Terraform."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

app.include_router(router)
app.include_router(ai_router)


@app.get("/health")
def root_health() -> dict[str, Any]:
    db_connected = False
    if settings.database_url:
        try:
            from app.database import connect
            with connect(settings) as conn:
                res = conn.execute("SELECT 1").fetchone()
                db_connected = bool(res)
        except Exception:
            db_connected = False

    all_features = list(AI_FEATURES)
    if "chatbot_copilot" not in all_features:
        all_features.append("chatbot_copilot")
    if "timescaledb_telemetry" not in all_features:
        all_features.append("timescaledb_telemetry")

    return {
        "status": "ok",
        "service": "smartcloud-ai-engine",
        "port": 8001,
        "database_connected": db_connected,
        "features": all_features,
    }


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": app.title, "docs": "/docs", "health": "/health"}

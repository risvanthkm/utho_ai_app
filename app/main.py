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
    import os
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

    model_loaded = False
    try:
        from app.services.tft_forecasting import load_model
        model, _ = load_model()
        model_loaded = model is not None
    except Exception as exc:
        print(f"Healthcheck model warning: {exc}")

    rag_size = 0
    try:
        from app.services.rag_service import _rag_chunks
        rag_size = len(_rag_chunks)
    except Exception as exc:
        print(f"Healthcheck RAG warning: {exc}")

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or settings.google_api_key

    return {
        "status": "ok",
        "service": "smartcloud-ai-engine",
        "port": 8001,
        "database_connected": db_connected,
        "db_connected": db_connected,
        "llm_configured": bool(api_key),
        "google_model": settings.google_model,
        "forecast_model_loaded": model_loaded,
        "rag_index_size": rag_size,
        "features": all_features,
    }


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": app.title, "docs": "/docs", "health": "/health"}

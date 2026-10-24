"""FastAPI application entry point."""

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ai_router, router
from app.services.smartcloud_ai import AI_FEATURES

app = FastAPI(
    title="Cloud Resource Advisor API",
    description=(
        "Forecast VM usage and cost, recommend safe capacity plans, simulate "
        "scheduled shutdowns, estimate sustainability, and generate Terraform."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(ai_router)


@app.get("/health")
def root_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "smartcloud-ai-engine",
        "port": 8001,
        "features": AI_FEATURES,
    }


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": app.title, "docs": "/docs", "health": "/health"}

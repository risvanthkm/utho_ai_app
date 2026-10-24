"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Cloud Resource Advisor API",
    description=(
        "Forecast VM usage and cost, recommend safe capacity plans, simulate "
        "scheduled shutdowns, estimate sustainability, and generate Terraform."
    ),
    version="1.0.0",
)
app.include_router(router)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": app.title, "docs": "/docs", "health": "/api/v1/health"}

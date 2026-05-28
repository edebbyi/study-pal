"""main.py: FastAPI application entrypoint for StudyPal backend APIs."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI

from src.api.models import HealthResponse
from src.core.version import APP_VERSION
from src.api.routers.ask import router as ask_router
from src.api.routers.documents import router as documents_router
from src.api.routers.evaluation import router as evaluation_router
from src.api.routers.observability import router as observability_router
from src.api.routers.publishing import router as publishing_router
from src.api.routers.runs import router as runs_router


app = FastAPI(
    title="StudyPal API",
    version=APP_VERSION,
    description="Backend API skeleton for StudyPal and future Publishing Mode workflows.",
)

api_router = APIRouter(prefix="/api")


@api_router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Simple health endpoint for uptime checks."""
    return HealthResponse()


api_router.include_router(documents_router)
api_router.include_router(ask_router)
api_router.include_router(publishing_router)
api_router.include_router(runs_router)
api_router.include_router(evaluation_router)
api_router.include_router(observability_router)
app.include_router(api_router)

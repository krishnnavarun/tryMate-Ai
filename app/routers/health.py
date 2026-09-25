from fastapi import APIRouter

from app import __version__
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check (no API key needed)")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)

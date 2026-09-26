"""FastAPI app: wires settings, middleware, error handlers and routers together.

Run locally:
    uvicorn app.main:app --reload --port 8000
or:
    python -m app.main          (uses PORT from .env)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartParser

from app import __version__
from app.config import MAX_REQUEST_BYTES, get_settings
from app.errors import AppError, register_error_handlers
from app.logging_setup import RequestContextMiddleware, configure_logging
from app.middleware import BodySizeLimitMiddleware
from app.routers import analyze, health, sizing, tryon
from app.services.face import get_face_finder
from app.services.pose import get_pose_detector

logger = logging.getLogger("app")


def keep_uploads_in_memory() -> None:
    """Make sure uploaded photos are never written to disk.

    Starlette stores each uploaded file in a SpooledTemporaryFile, which moves to a real
    temp file on disk once it grows past `spool_max_size` (1 MB by default). Most phone
    photos are bigger than that. Raising the limit above the maximum request size means
    uploads always stay in RAM. BodySizeLimitMiddleware caps the request size, so this
    can't be used to fill up memory with huge uploads.
    """
    MultiPartParser.spool_max_size = MAX_REQUEST_BYTES + 1024 * 1024


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Runs once at startup: load the pose + face models now, so the first /analyze isn't slow.

    If a model file is missing we only log it: /health keeps working, and /analyze
    answers with a clear INTERNAL_ERROR message until the model is downloaded.
    """
    for load in (get_pose_detector, get_face_finder):
        try:
            await run_in_threadpool(load)
        except AppError as err:
            logger.error(err.message)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_format == "json")
    keep_uploads_in_memory()

    if settings.service_api_key == "change-me":
        logger.warning("SERVICE_API_KEY is still 'change-me'. Set a real secret in .env before deploying.")

    app = FastAPI(
        lifespan=lifespan,
        title="tryMate AI Service",
        version=__version__,
        description=(
            "Body measurements, skin tone, size recommendation and virtual try-on.\n\n"
            "Called only by the tryMate store's Express server. Every endpoint except "
            "`/health` needs the `X-API-Key` header (click **Authorize**).\n\n"
            "Try-on shows how a garment **looks**, not how it **fits**; "
            "fit comes from `/recommend-size`."
        ),
    )

    # No CORS middleware on purpose: only the Express server calls this service
    # (server-to-server), so browsers should never be allowed to.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)
    # Added last = runs first: every request (even rejected ones) gets an id + access log line
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)

    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(sizing.router)
    app.include_router(tryon.router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=get_settings().port)

import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import AnyHttpUrl

from app.errors import AppError, ErrorCode
from app.routers import COMMON_ERROR_RESPONSES
from app.schemas import Category, ErrorResponse, TryOnResponse
from app.security import require_api_key
from app.services.image_io import read_image_upload

router = APIRouter(tags=["try-on"], dependencies=[Depends(require_api_key)])

_STUB_RESULT_URL = "https://placehold.co/768x1024/png?text=Try-on+stub"


@router.post(
    "/try-on",
    response_model=TryOnResponse,
    responses={
        **COMMON_ERROR_RESPONSES,
        502: {"model": ErrorResponse, "description": "TRYON_FAILED"},
        504: {"model": ErrorResponse, "description": "TRYON_TIMEOUT"},
    },
    summary="Image of the person wearing the garment (shows the look, not the fit)",
)
async def try_on(
    person_image: Annotated[UploadFile, File(description="Photo of the person. JPEG, PNG or WEBP, max 10 MB.")],
    category: Annotated[Category, Form()],
    garment_image: Annotated[
        UploadFile | None, File(description="Garment photo. Send this OR garment_image_url.")
    ] = None,
    garment_image_url: Annotated[
        AnyHttpUrl | None, Form(description="Garment image URL. Send this OR garment_image.")
    ] = None,
    garment_description: Annotated[
        str | None, Form(max_length=300, examples=["navy blue cotton polo shirt"])
    ] = None,
) -> TryOnResponse:
    started = time.perf_counter()

    # Exactly one garment source is allowed, so there is never any doubt which one was used.
    if garment_image is None and garment_image_url is None:
        raise AppError(ErrorCode.INVALID_INPUT, "Send garment_image or garment_image_url.")
    if garment_image is not None and garment_image_url is not None:
        raise AppError(ErrorCode.INVALID_INPUT, "Send only one of garment_image or garment_image_url, not both.")

    await read_image_upload(person_image, "person_image")
    if garment_image is not None:
        await read_image_upload(garment_image, "garment_image")

    # ---- PHASE 1 STUB -------------------------------------------------------
    # Returns a placeholder image URL. Phase 5 adds the provider interface,
    # Replicate IDM-VTON, the mock provider (TRYON_MOCK) and timeouts/retries.
    latency_ms = int((time.perf_counter() - started) * 1000)
    return TryOnResponse(
        result_image_url=_STUB_RESULT_URL,
        result_image_base64=None,
        latency_ms=latency_ms,
        provider="stub",
    )

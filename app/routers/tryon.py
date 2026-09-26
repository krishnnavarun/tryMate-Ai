import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import AnyHttpUrl

from app.errors import AppError, ErrorCode
from app.rate_limit import limit_tryon
from app.routers import COMMON_ERROR_RESPONSES
from app.schemas import Category, ErrorResponse, TryOnResponse
from app.security import require_api_key
from app.services.image_io import read_image_upload
from app.tryon import get_tryon_provider
from app.tryon.base import TryOnProvider, TryOnRequest

router = APIRouter(tags=["try-on"], dependencies=[Depends(require_api_key)])


@router.post(
    "/try-on",
    response_model=TryOnResponse,
    responses={
        **COMMON_ERROR_RESPONSES,
        429: {"model": ErrorResponse, "description": "RATE_LIMITED: too many try-ons this minute"},
        502: {"model": ErrorResponse, "description": "TRYON_FAILED"},
        504: {"model": ErrorResponse, "description": "TRYON_TIMEOUT"},
    },
    summary="Image of the person wearing the garment (shows the look, not the fit)",
    description=(
        "Send `garment_image` **or** `garment_image_url` (exactly one). Takes ~10–60 s with the "
        "real provider (timeout 120 s). Exactly one of `result_image_url` / `result_image_base64` "
        "is set. This shows how the garment LOOKS; fit comes from /recommend-size."
    ),
)
async def try_on(
    person_image: Annotated[UploadFile, File(description="Photo of the person. JPEG, PNG or WEBP, max 10 MB.")],
    category: Annotated[Category, Form()],
    provider: Annotated[TryOnProvider, Depends(get_tryon_provider)],
    _rate_limit: Annotated[None, Depends(limit_tryon)],
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

    request = TryOnRequest(
        person_image=await read_image_upload(person_image, "person_image"),
        category=category,
        garment_image=await read_image_upload(garment_image, "garment_image") if garment_image else None,
        garment_image_url=str(garment_image_url) if garment_image_url else None,
        garment_description=garment_description,
    )
    result = await provider.run(request)

    return TryOnResponse(
        result_image_url=result.image_url,
        result_image_base64=result.image_base64,
        latency_ms=int((time.perf_counter() - started) * 1000),
        provider=provider.name,
    )

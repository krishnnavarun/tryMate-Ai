from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app.routers import COMMON_ERROR_RESPONSES
from app.schemas import (
    HEIGHT_MAX_CM,
    HEIGHT_MIN_CM,
    WEIGHT_MAX_KG,
    WEIGHT_MIN_KG,
    AnalyzeResponse,
    ColorSuggestion,
    SkinTone,
)
from app.security import require_api_key
from app.services.body_analysis import analyze_body
from app.services.image_io import read_image_upload
from app.services.pose import PoseDetector, get_pose_detector

router = APIRouter(tags=["analyze"], dependencies=[Depends(require_api_key)])

# Placeholder until Phase 3 (face → skin tone → colours)
_STUB_SKIN_TONE = SkinTone(tone="medium", undertone="warm", hex="#C68E6A")
_STUB_COLORS = [
    ColorSuggestion(name="Olive", hex="#708238"),
    ColorSuggestion(name="Mustard", hex="#D4A017"),
    ColorSuggestion(name="Rust", hex="#B7410E"),
    ColorSuggestion(name="Camel", hex="#C19A6B"),
    ColorSuggestion(name="Teal", hex="#008080"),
    ColorSuggestion(name="Cream", hex="#FFFDD0"),
]
_STUB_SKIN_WARNING = "Skin tone and color suggestions are placeholders (not implemented yet)."


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses=COMMON_ERROR_RESPONSES,
    summary="Body measurements + skin tone from a full-body photo",
)
async def analyze(
    image: Annotated[UploadFile, File(description="Full-body front photo. JPEG, PNG or WEBP, max 10 MB.")],
    height_cm: Annotated[float, Form(ge=HEIGHT_MIN_CM, le=HEIGHT_MAX_CM, description="User's real height")],
    detector: Annotated[PoseDetector, Depends(get_pose_detector)],
    weight_kg: Annotated[
        float | None, Form(ge=WEIGHT_MIN_KG, le=WEIGHT_MAX_KG, description="Optional; not used yet")
    ] = None,
    debug: Annotated[bool, Form(description="Also return an image with landmarks and measured lines drawn")] = False,
) -> AnalyzeResponse:
    # Validates type + size; the bytes are only held in memory.
    image_bytes = await read_image_upload(image, "image")

    # Pose detection + measurement is CPU work (~0.1–0.5 s). Running it in a worker
    # thread keeps the server responsive to other requests meanwhile.
    body = await run_in_threadpool(analyze_body, image_bytes, height_cm, detector, debug)

    return AnalyzeResponse(
        measurements=body.measurements,
        skin_tone=_STUB_SKIN_TONE,  # Phase 3
        color_suggestions=_STUB_COLORS,  # Phase 3
        confidence=body.confidence,
        warnings=[*body.warnings, _STUB_SKIN_WARNING],
        debug_image_base64=body.debug_image_base64,
    )

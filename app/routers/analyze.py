from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app.routers import COMMON_ERROR_RESPONSES
from app.schemas import HEIGHT_MAX_CM, HEIGHT_MIN_CM, WEIGHT_MAX_KG, WEIGHT_MIN_KG, AnalyzeResponse
from app.security import require_api_key
from app.services.body_analysis import analyze_body
from app.services.face import FaceFinder, get_face_finder
from app.services.image_io import read_image_upload
from app.services.pose import PoseDetector, get_pose_detector

router = APIRouter(tags=["analyze"], dependencies=[Depends(require_api_key)])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses=COMMON_ERROR_RESPONSES,
    summary="Body measurements + skin tone from a full-body photo",
    description=(
        "Returns measurements (cm), skin tone, suggested colors, a confidence score and warnings. "
        "If the body is fine but the face can't be seen, the response is still 200 with "
        "`skin_tone: null`, `color_suggestions: []` and a warning."
    ),
)
async def analyze(
    image: Annotated[UploadFile, File(description="Full-body front photo. JPEG, PNG or WEBP, max 10 MB.")],
    height_cm: Annotated[float, Form(ge=HEIGHT_MIN_CM, le=HEIGHT_MAX_CM, description="User's real height")],
    detector: Annotated[PoseDetector, Depends(get_pose_detector)],
    face_finder: Annotated[FaceFinder, Depends(get_face_finder)],
    weight_kg: Annotated[
        float | None, Form(ge=WEIGHT_MIN_KG, le=WEIGHT_MAX_KG, description="Optional; not used yet")
    ] = None,
    debug: Annotated[bool, Form(description="Also return an image with landmarks and measured lines drawn")] = False,
) -> AnalyzeResponse:
    # Validates type + size; the bytes are only held in memory.
    image_bytes = await read_image_upload(image, "image")

    # Pose, face and colour analysis are CPU work (~0.1–0.5 s). Running them in a worker
    # thread keeps the server responsive to other requests meanwhile.
    result = await run_in_threadpool(analyze_body, image_bytes, height_cm, detector, face_finder, debug)

    return AnalyzeResponse(
        measurements=result.measurements,
        skin_tone=result.skin_tone,
        color_suggestions=result.color_suggestions,
        confidence=result.confidence,
        warnings=result.warnings,
        debug_image_base64=result.debug_image_base64,
    )

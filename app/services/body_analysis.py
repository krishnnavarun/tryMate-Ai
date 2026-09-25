"""Runs the whole body pipeline for /analyze: decode → pose → checks → measurements.

This is plain synchronous, CPU-heavy code; the router runs it in a thread pool so the
server can keep answering other requests meanwhile.
"""

from dataclasses import dataclass

from app.schemas import Measurements
from app.services.debug_image import render_debug_image
from app.services.image_io import decode_image
from app.services.measurements import measure
from app.services.pose import PoseDetector, select_single_full_body


@dataclass
class BodyAnalysis:
    measurements: Measurements
    confidence: float
    warnings: list[str]
    debug_image_base64: str | None


def analyze_body(image_bytes: bytes, height_cm: float, detector: PoseDetector, debug: bool = False) -> BodyAnalysis:
    image = decode_image(image_bytes, "image")
    pose = select_single_full_body(detector.detect(image))  # raises NO_PERSON / MULTIPLE / PARTIAL
    result = measure(pose, height_cm)
    return BodyAnalysis(
        measurements=result.measurements,
        confidence=result.confidence,
        warnings=result.warnings,
        debug_image_base64=render_debug_image(image, pose, result) if debug else None,
    )

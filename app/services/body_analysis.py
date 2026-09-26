"""Runs the whole /analyze pipeline: decode → pose → checks → measurements → face → skin tone.

This is plain synchronous, CPU-heavy code; the router runs it in a thread pool so the
server can keep answering other requests meanwhile.

FACE_NOT_FOUND decision: if the body is fine but no face is found (turned head, hair,
mask, tiny face), we still return the measurements, with `skin_tone: null`,
`color_suggestions: []` and a warning. Measurements are the main value of a scan; failing
the whole request because of the face would make users re-take a photo that was good for
sizing. The FACE_NOT_FOUND error code stays in the contract but /analyze doesn't raise it.
"""

from dataclasses import dataclass, field

from app.schemas import ColorSuggestion, Measurements, SkinTone
from app.services.colors import suggest_colors
from app.services.debug_image import render_debug_image
from app.services.face import FaceFinder
from app.services.image_io import decode_image
from app.services.measurements import measure
from app.services.pose import PoseDetector, select_single_full_body
from app.services.skin_tone import analyze_skin

WARN_NO_FACE = "We couldn't see your face clearly, so skin tone and color suggestions are missing"


@dataclass
class BodyAnalysis:
    measurements: Measurements
    confidence: float
    warnings: list[str]
    skin_tone: SkinTone | None
    color_suggestions: list[ColorSuggestion] = field(default_factory=list)
    debug_image_base64: str | None = None


def analyze_body(
    image_bytes: bytes,
    height_cm: float,
    detector: PoseDetector,
    face_finder: FaceFinder,
    debug: bool = False,
) -> BodyAnalysis:
    image = decode_image(image_bytes, "image")
    pose = select_single_full_body(detector.detect(image))  # raises NO_PERSON / MULTIPLE / PARTIAL
    body = measure(pose, height_cm)
    warnings = list(body.warnings)

    face = face_finder.find(image, pose)
    skin = analyze_skin(image, face) if face is not None else None
    skin_tone, colors, circles = None, [], []
    if skin is None:
        warnings.append(WARN_NO_FACE)
    else:
        skin_tone = skin.skin_tone
        colors = suggest_colors(skin_tone.tone, skin_tone.undertone)
        warnings.extend(skin.warnings)
        circles = skin.sample_circles

    return BodyAnalysis(
        measurements=body.measurements,
        confidence=body.confidence,
        warnings=warnings,
        skin_tone=skin_tone,
        color_suggestions=colors,
        debug_image_base64=render_debug_image(image, pose, body, circles) if debug else None,
    )

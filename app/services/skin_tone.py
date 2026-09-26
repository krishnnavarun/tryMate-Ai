"""Face → skin colour → tone (how light/deep) + undertone (warm / cool / neutral) + hex.

HOW IT WORKS
============

1. Where to sample
   Three small circles on the face mesh: mid-forehead (point 151) and the centre of each
   cheek (points 205 and 425). These spots are skin on almost everyone, and far enough from
   the eyes, eyebrows, lips, nostrils and hairline. Circle radius = 7% of the face width.

2. Cleaning the pixels
   - Keep pixels in the classic YCrCb skin range (Cr 133–173, Cb 77–127; Chai & Ngan, 1999).
     That throws away stray hair, eyebrow, beard and background pixels.
   - Drop the darkest 15% and brightest 10% (shadows and shiny highlights).
   - Take the MEDIAN colour in CIELAB. The median ignores the odd outlier pixel.

3. CIELAB in one paragraph
   L* = lightness (0 black … 100 white), a* = green (−) ↔ red (+), b* = blue (−) ↔ yellow (+).
   Unlike RGB, distances in LAB roughly match what people see, which makes it the standard
   space for describing skin colour.

4. Tone: Individual Typology Angle (ITA°), a standard dermatology measure
       ITA = atan((L* − 50) / b*) × 180/π
   Categories from Chardon et al. (1991) / Del Bino et al. (2006):
       > 55 very light · 41–55 light · 28–41 intermediate · 10–28 tan · −30–10 brown · < −30 dark
   We label them fair / light / medium / tan / brown / deep.

5. Undertone: hue angle h = atan2(b*, a*)
   Skin in photos sits between red and yellow (h ≈ 30–60°). More yellow/golden (higher h) = warm,
   more pink/red (lower h) = cool, in between = neutral. There's no single scientific
   standard for undertone; the thresholds below are a simple, explainable heuristic —
   tune UNDERTONE_COOL_BELOW / UNDERTONE_WARM_ABOVE with real photos.

LIGHTING (the biggest source of error)
   A camera doesn't measure skin colour, it records skin × light × exposure. The same person
   looks darker in a dim room and "warmer" under yellow bulbs. ITA's published thresholds
   come from calibrated instruments, so on photos they only hold for well-lit pictures.
   What we do about it:
   - "White patch" correction: the brightest near-grey areas (white walls, a white shirt)
     are assumed to be white, and the skin pixels are corrected by the same amount
     (bounded, and skipped if the photo has no such areas). See white_balance_gains().
   - Warnings for strong colour casts and dark photos, asking for daylight.
   Results are most reliable in soft daylight, facing a window, no filters. Test photos taken
   indoors or against the light come out too deep — that is a known limitation.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas import SkinTone
from app.services.face import Face

# Face-mesh points at the centre of each sample circle
SAMPLE_POINTS = {"forehead": 151, "right_cheek": 205, "left_cheek": 425}
SAMPLE_RADIUS = 0.07  # × face width

# Tone thresholds (ITA°, Del Bino et al. 2006), from lightest to deepest
TONE_BANDS = [(55, "fair"), (41, "light"), (28, "medium"), (10, "tan"), (-30, "brown")]
DEEPEST_TONE = "deep"

# Undertone thresholds on the hue angle (degrees) — heuristic, tune with real photos
UNDERTONE_COOL_BELOW = 40.0
UNDERTONE_WARM_ABOVE = 52.0

# White balance / exposure correction (see white_balance_gains)
WB_TARGET = 235.0  # the brightest near-grey areas are scaled to this RGB level
WB_MAX_CHROMA = 12.0  # "near-grey" = LAB chroma below this
WB_MIN_FRACTION = 0.005  # need at least 0.5% of the photo as reference
WB_GAIN_LIMITS = (0.8, 1.6)

MIN_SKIN_PIXELS = 30
SMALL_FACE_PX = 40  # faces narrower than this give noisy colours

WARN_SMALL_FACE = "Your face is small in the photo — skin tone may be less accurate"
WARN_FEW_PIXELS = "Couldn't see enough clear skin on your face — skin tone may be less accurate"
WARN_COLOR_CAST = "The lighting has a strong colour tint — for accurate colours use daylight"
WARN_DARK_PHOTO = "The photo is quite dark — for accurate colours use brighter, even light"


@dataclass
class SkinResult:
    skin_tone: SkinTone
    ita: float
    hue: float
    warnings: list[str]
    sample_circles: list[tuple[float, float, float]]  # (x, y, radius) for the debug image


def _to_lab(pixels_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 pixels (N×3) → CIELAB floats (L 0–100, a/b roughly −128…127)."""
    if len(pixels_rgb) == 0:
        return np.empty((0, 3), dtype=np.float32)
    as_image = (pixels_rgb.reshape(-1, 1, 3).astype(np.float32)) / 255.0
    return cv2.cvtColor(as_image, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def lab_to_hex(lab: np.ndarray) -> str:
    rgb = cv2.cvtColor(np.array([[lab]], dtype=np.float32), cv2.COLOR_LAB2RGB)[0, 0]
    r, g, b = (int(round(float(np.clip(c, 0, 1)) * 255)) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def sample_skin_pixels(image_rgb: np.ndarray, face: Face) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """All RGB pixels inside the three sample circles, and the circles themselves."""
    height, width = image_rgb.shape[:2]
    radius = max(2.0, SAMPLE_RADIUS * face.width_px)
    mask = np.zeros((height, width), dtype=np.uint8)
    circles = []
    for index in SAMPLE_POINTS.values():
        x, y = face.points[index]
        cv2.circle(mask, (int(round(x)), int(round(y))), int(round(radius)), 1, -1)
        circles.append((float(x), float(y), radius))
    return image_rgb[mask.astype(bool)], circles


def clean_skin_pixels(pixels_rgb: np.ndarray) -> np.ndarray:
    """Keep skin-coloured, well-exposed pixels. Returns LAB values (N×3)."""
    ycrcb = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    cr, cb = ycrcb[:, 1], ycrcb[:, 2]
    skin_like = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    lab = _to_lab(pixels_rgb[skin_like])
    if len(lab) < MIN_SKIN_PIXELS:
        return lab
    low, high = np.percentile(lab[:, 0], [15, 90])
    return lab[(lab[:, 0] >= low) & (lab[:, 0] <= high)]


def classify_tone(ita: float) -> str:
    for threshold, label in TONE_BANDS:
        if ita > threshold:
            return label
    return DEEPEST_TONE


def classify_undertone(hue: float) -> str:
    if hue < UNDERTONE_COOL_BELOW:
        return "cool"
    if hue > UNDERTONE_WARM_ABOVE:
        return "warm"
    return "neutral"


def ita_angle(lightness: float, b: float) -> float:
    # atan2 instead of atan(x / b) avoids dividing by zero when b* is 0
    return math.degrees(math.atan2(lightness - 50.0, b))


def white_balance_gains(image_rgb: np.ndarray) -> np.ndarray:
    """Per-channel (R, G, B) gains that correct exposure and colour cast ("white patch" method).

    Idea: the brightest near-grey areas of a photo (white walls, shutters, a white shirt)
    should look white. If they look dim or tinted, the whole photo is, so scale R, G and B
    to make that reference a clean light grey (235). Blown-out pure white pixels are skipped
    (they carry no information), and the gains are limited so a photo with no real white
    areas can't be pushed too far. Returns [1, 1, 1] when there's no usable reference.
    """
    longest = max(image_rgb.shape[:2])
    small = cv2.resize(image_rgb, None, fx=256 / longest, fy=256 / longest, interpolation=cv2.INTER_AREA)
    rgb = small.reshape(-1, 3).astype(np.float32)
    lab = _to_lab(small.reshape(-1, 3))
    lightness, chroma = lab[:, 0], np.hypot(lab[:, 1], lab[:, 2])
    not_clipped = rgb.max(axis=1) < 250
    candidates = not_clipped & (chroma < WB_MAX_CHROMA) & (lightness >= np.percentile(lightness[not_clipped], 97))
    if candidates.sum() < WB_MIN_FRACTION * len(rgb):
        return np.ones(3, dtype=np.float32)
    reference = rgb[candidates].mean(axis=0)
    return np.clip(WB_TARGET / np.maximum(reference, 1.0), *WB_GAIN_LIMITS).astype(np.float32)


def lighting_warnings(image_rgb: np.ndarray) -> list[str]:
    """Warn about strong colour casts and dark photos (see LIGHTING above)."""
    small = cv2.resize(image_rgb, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    lab = _to_lab(small)
    usable = lab[(lab[:, 0] > 15) & (lab[:, 0] < 95)]  # ignore black and blown-out pixels
    warnings = []
    if len(usable) and (abs(np.mean(usable[:, 1])) > 10 or abs(np.mean(usable[:, 2])) > 15):
        warnings.append(WARN_COLOR_CAST)
    if np.median(lab[:, 0]) < 30:
        warnings.append(WARN_DARK_PHOTO)
    return warnings


def analyze_skin(image_rgb: np.ndarray, face: Face) -> SkinResult | None:
    """Skin tone of the face, or None if no pixels could be sampled (face outside the photo)."""
    warnings = []
    pixels, circles = sample_skin_pixels(image_rgb, face)
    if len(pixels) == 0:
        return None
    gains = white_balance_gains(image_rgb)
    pixels = np.clip(pixels.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    lab = clean_skin_pixels(pixels)
    if len(lab) < MIN_SKIN_PIXELS:
        # Not enough skin-like pixels (odd lighting, makeup, tiny face): use everything we sampled
        warnings.append(WARN_FEW_PIXELS)
        lab = _to_lab(pixels)
    if face.width_px < SMALL_FACE_PX:
        warnings.append(WARN_SMALL_FACE)
    warnings.extend(lighting_warnings(image_rgb))

    median = np.median(lab, axis=0)
    lightness, a, b = (float(v) for v in median)
    ita = ita_angle(lightness, b)
    hue = math.degrees(math.atan2(b, a))

    skin_tone = SkinTone(tone=classify_tone(ita), undertone=classify_undertone(hue), hex=lab_to_hex(median))
    return SkinResult(skin_tone=skin_tone, ita=ita, hue=hue, warnings=warnings, sample_circles=circles)

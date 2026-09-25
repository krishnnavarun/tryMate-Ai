"""Pose landmarks + segmentation mask + the user's height → body measurements in cm.

HOW IT WORKS
============

1. Scale (pixels → cm)
   A photo has no built-in scale, so we use the user's real height:
       cm_per_px = height_cm / (sole_y − head_top_y)
   Pose landmarks stop at the eyes/ears, so the top of the head comes from the
   segmentation mask (the highest "person" pixel above the face). The soles come from the
   lowest mask pixel around the feet. If the mask looks wrong we fall back to landmarks.

2. Lengths (straight from landmarks × cm_per_px)
   shoulder_cm  distance between the two shoulder landmarks × SHOULDER_WIDTH_FACTOR
                (the landmarks are the shoulder joints; the factor converts to the bony
                shoulder-tip width that tailors measure; see the constant below)
   torso_cm     shoulder-line midpoint → hip-line midpoint
   arm_cm       shoulder → elbow → wrist (averaged over the visible arms)
   leg_cm       hip → knee → ankle (averaged over the visible legs)

3. Chest and waist circumference
   A front photo shows how WIDE the chest/waist is, not how far around it is.
   We measure the silhouette width at chest and waist height, then multiply by a ratio
   (circumference ÷ breadth) taken from ANSUR II, a 2012 survey of 4,082 US Army men
   with dozens of tape/caliper measurements each:
       chest circumference ≈ 3.658 × chest breadth   (sd of the ratio 0.20)
       waist circumference ≈ 2.879 × waist breadth   (sd of the ratio 0.10)
   Even with a perfect breadth, this alone gives an average error of ~4.7 cm (chest)
   and ~2.6 cm (waist), because bodies differ in depth. (We also tried an ellipse from
   breadth + assumed depth: no more accurate, so the simpler ratio wins.)
   Chest and waist heights also come from ANSUR II: on average the chest line sits 27.7%
   and the waist (navel) line 71.2% of the way down from the shoulder to the hip.

4. Finding the torso edges (and arms touching the body)
   Starting from the body's centre line we walk outwards along the row until the mask
   says "not person". We never walk past the arm (its position comes from the shoulder →
   elbow → wrist landmarks), so the arm is never counted as chest. The mask model works at
   low resolution and often blurs a thin gap between arm and torso into a mere dip in
   value; the deepest dip before the arm is then used as the edge.
   If there's neither an edge nor a dip before the arm, the arm really touches the body
   at that height: we estimate the breadth from shoulder width instead (ANSUR II ratios
   0.697 chest/shoulder, 0.785 waist/shoulder) and add a warning. Standing with the arms
   ~30° away from the body (an "A" pose) avoids this.

Expected accuracy (honest): with a good photo (fitted clothes, arms slightly away, camera
at chest height, facing straight on), roughly ±2–3 cm for lengths and ±4–8 cm for chest/
waist circumference. Loose clothes, bad posture or a tilted camera make it worse.
The constants below are the knobs to tune once you compare results with a measuring tape.

Constants are for men (the MVP catalogue is men's tops). Women's ANSUR II ratios are
3.516 (chest) and 2.871 (waist).
"""

from dataclasses import dataclass, field

import numpy as np

from app.schemas import Measurements
from app.services.pose import MIN_VISIBILITY, L, Pose

# ---- Ratios from ANSUR II (US Army 2012, men, n = 4082) -------------------------------
CHEST_CIRC_PER_BREADTH = 3.658
WAIST_CIRC_PER_BREADTH = 2.879
CHEST_LEVEL = 0.277  # fraction of the way from the shoulder line down to the hip line
WAIST_LEVEL = 0.712
CHEST_BREADTH_PER_SHOULDER = 0.697  # fallbacks when arms hide the chest/waist edges
WAIST_BREADTH_PER_SHOULDER = 0.785
ARM_PER_HEIGHT = 0.338  # sleeve outseam ÷ stature, fallback when arms aren't visible
LEG_PER_HEIGHT = 0.47  # hip joint → ankle ÷ stature (trochanter height 0.513 − ankle ~0.04)

# ---- Tuning knobs (calibrate against tape measurements) --------------------------------
# MediaPipe's shoulder landmarks sit at the shoulder JOINTS, a few cm inside the bony
# shoulder tips (acromions) that shoulder width is measured between. On a test photo,
# 0.816 × silhouette width across the deltoids came out 1.11–1.21× the landmark distance,
# so 1.15 is the starting value. Adjust after comparing with real tape measurements.
SHOULDER_WIDTH_FACTOR = 1.15
MASK_THRESHOLD = 0.5  # mask value above which a pixel counts as "person"
# A dip in the mask below this (between body and arm) marks a gap the mask blurred over
DIP_THRESHOLD = 0.9
# Fallback for the top of the head: it sits about 0.62 × (eyes → shoulder line) above the eyes
HEAD_TOP_ABOVE_EYES = 0.62

# Chest/waist breadth ÷ shoulder width (after SHOULDER_WIDTH_FACTOR) outside these limits
# means the silhouette is wrong. In ANSUR II the ratios are 0.70 (chest) and 0.79 (waist),
# up to ~1.0 for heavier men; clothing adds a bit more.
BREADTH_RATIO_LIMITS = (0.45, 1.2)

WARN_ARMS_CHEST = "Arms close to body — chest was estimated from shoulder width and may be less accurate"
WARN_ARMS_WAIST = "Arms or hands touching the waist — waist was estimated and may be less accurate"
WARN_HEAD = "Couldn't clearly find the top of your head — measurements may be slightly off"
WARN_FEET = "Couldn't clearly find your feet — measurements may be slightly off"
WARN_TURNED = "Your body seems turned — stand facing the camera straight on"
WARN_SMALL = "You look small in the photo — stand closer so your body fills most of the frame"
WARN_ARM_LENGTH = "Arms not clearly visible — arm length was estimated from your height"
WARN_LEG_LENGTH = "Legs not clearly visible — leg length was estimated from your height"


@dataclass
class DebugLine:
    """A line to draw on the debug image."""

    start: tuple[float, float]
    end: tuple[float, float]
    label: str


@dataclass
class MeasurementResult:
    measurements: Measurements
    confidence: float
    warnings: list[str]
    cm_per_px: float
    lines: list[DebugLine] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------


def _mid(pose: Pose, a: int, b: int) -> np.ndarray:
    return (pose[a].xy + pose[b].xy) / 2


def _dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def _visible(pose: Pose, *indices: int) -> bool:
    return all(pose[i].visibility >= MIN_VISIBILITY for i in indices)


def _x_on_segment_at_y(p: np.ndarray, q: np.ndarray, y: float) -> float | None:
    """x where the segment p→q crosses height y, or None if it doesn't reach that height."""
    (x1, y1), (x2, y2) = p, q
    if min(y1, y2) > y or max(y1, y2) < y or abs(y2 - y1) < 1e-6:
        return None
    t = (y - y1) / (y2 - y1)
    return float(x1 + t * (x2 - x1))


def _mask_profile(mask: np.ndarray, y: float) -> np.ndarray:
    """The mask values along row y, averaged with 2 rows above and below (less noise)."""
    row = int(round(y))
    top, bottom = max(0, row - 2), min(mask.shape[0], row + 3)
    return mask[top:bottom].mean(axis=0)


def _find_edge(profile: np.ndarray, x_center: int, direction: int, limit: int) -> int | None:
    """Walk from the body centre outwards (direction −1 = left, +1 = right) and find the torso edge.

    - Normal case: the mask drops below MASK_THRESHOLD → that's the edge of the body.
    - Arm next to the body: the mask model works at low resolution, so a small gap between
      arm and torso often isn't fully empty in the mask; it only shows up as a dip
      (e.g. 1.0 → 0.7 → 1.0). If we reach the arm (`limit`) without a real edge, we use
      the deepest dip on the way, if it's clearly below 1.
    - No edge and no dip before the arm → the arm really touches the body here → None.
    """
    lowest_value, lowest_x = 1.0, None
    x = x_center
    while x != limit:
        x += direction
        if not 0 <= x < len(profile):
            return None
        value = profile[x]
        if value < MASK_THRESHOLD:
            return x - direction  # last pixel that still belongs to the body
        if value < lowest_value:
            lowest_value, lowest_x = value, x
    return lowest_x if lowest_value < DIP_THRESHOLD else None


# ---------------------------------------------------------------------------
# Head top and soles → scale
# ---------------------------------------------------------------------------


def _head_top_y(pose: Pose, shoulder_px: float) -> tuple[float, bool]:
    """(y of the top of the head, found_in_mask)."""
    eyes_y = (pose[L.LEFT_EYE].y + pose[L.RIGHT_EYE].y) / 2
    shoulders_y = _mid(pose, L.LEFT_SHOULDER, L.RIGHT_SHOULDER)[1]
    eyes_to_shoulders = max(1.0, shoulders_y - eyes_y)
    estimate = eyes_y - HEAD_TOP_ABOVE_EYES * eyes_to_shoulders

    if pose.mask is not None:
        # Look in a column band about one head wide, centred on the nose
        half_band = max(3.0, 0.2 * shoulder_px)
        x1 = int(max(0, pose[L.NOSE].x - half_band))
        x2 = int(min(pose.width, pose[L.NOSE].x + half_band + 1))
        band = pose.mask[: int(pose[L.NOSE].y), x1:x2] > MASK_THRESHOLD
        rows = np.flatnonzero(band.any(axis=1))
        if rows.size:
            top = float(rows[0])
            # Accept only if it's a sensible distance above the eyes
            # (not the image edge because of a messy background, not inside the face)
            if eyes_y - 1.2 * eyes_to_shoulders <= top <= eyes_y - 0.3 * eyes_to_shoulders:
                return top, True
    return estimate, False


def _sole_y(pose: Pose, shoulder_px: float, head_top_y: float) -> tuple[float, bool]:
    """(y of the soles of the feet, found_in_mask)."""
    feet = [L.LEFT_ANKLE, L.RIGHT_ANKLE, L.LEFT_HEEL, L.RIGHT_HEEL, L.LEFT_FOOT_INDEX, L.RIGHT_FOOT_INDEX]
    visible_feet = [pose[i] for i in feet if pose[i].visibility >= MIN_VISIBILITY * 0.6] or [
        pose[L.LEFT_ANKLE],
        pose[L.RIGHT_ANKLE],
    ]
    lowest_landmark = max(lm.y for lm in visible_feet)

    if pose.mask is not None:
        margin = 0.15 * shoulder_px
        x1 = int(max(0, min(lm.x for lm in visible_feet) - margin))
        x2 = int(min(pose.width, max(lm.x for lm in visible_feet) + margin + 1))
        ankles_y = int(min(pose[L.LEFT_ANKLE].y, pose[L.RIGHT_ANKLE].y))
        band = pose.mask[ankles_y:, x1:x2] > MASK_THRESHOLD
        rows = np.flatnonzero(band.any(axis=1))
        if rows.size:
            bottom = float(ankles_y + rows[-1])
            # Accept unless it runs far below the feet (e.g. a shadow joined to the mask)
            body_px = lowest_landmark - head_top_y
            if lowest_landmark - 0.02 * body_px <= bottom <= lowest_landmark + 0.04 * body_px:
                return bottom, True
    return lowest_landmark, False


# ---------------------------------------------------------------------------
# Chest / waist breadth
# ---------------------------------------------------------------------------


def _arm_x_at(pose: Pose, y: float, side: str) -> float | None:
    """x of the arm (upper arm or forearm, whichever crosses height y) on one side."""
    shoulder, elbow, wrist = (
        (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST)
        if side == "left"
        else (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST)
    )
    if pose[elbow].visibility < MIN_VISIBILITY:
        return None
    x = _x_on_segment_at_y(pose[shoulder].xy, pose[elbow].xy, y)
    if x is None and pose[wrist].visibility >= MIN_VISIBILITY:
        x = _x_on_segment_at_y(pose[elbow].xy, pose[wrist].xy, y)
    return x


def _breadth_px(pose: Pose, level: float, shoulder_px: float) -> tuple[float | None, tuple[int, int, float] | None]:
    """Torso width (px) at `level` between the shoulder line (0) and the hip line (1).

    Returns (width, or None if it can't be measured reliably; (left_x, right_x, y) to draw).
    """
    if pose.mask is None:
        return None, None

    shoulders = _mid(pose, L.LEFT_SHOULDER, L.RIGHT_SHOULDER)
    hips = _mid(pose, L.LEFT_HIP, L.RIGHT_HIP)
    x_center, y = shoulders + level * (hips - shoulders)
    xc = int(round(x_center))
    profile = _mask_profile(pose.mask, y)
    if not 0 <= xc < len(profile) or profile[xc] < MASK_THRESHOLD:
        return None, None

    # Where is each arm at this height? The edge search stops there, so the arm is never
    # counted as chest/waist.
    limits = {-1: 0, +1: len(profile) - 1}
    for side in ("left", "right"):
        arm_x = _arm_x_at(pose, y, side)
        if arm_x is None:
            continue
        if abs(arm_x - x_center) < 0.2 * shoulder_px:
            return None, None  # arm in front of the body (e.g. crossed arms)
        direction = 1 if arm_x > x_center else -1
        limits[direction] = int(round(arm_x))

    left = _find_edge(profile, xc, -1, limits[-1])
    right = _find_edge(profile, xc, +1, limits[+1])
    if left is None or right is None:
        return None, None

    width = float(right - left + 1)
    low, high = BREADTH_RATIO_LIMITS
    if not low <= width / (shoulder_px * SHOULDER_WIDTH_FACTOR) <= high:
        return None, (left, right, y)
    return width, (left, right, y)


# ---------------------------------------------------------------------------
# Limb lengths
# ---------------------------------------------------------------------------


def _limb_px(pose: Pose, chains: list[tuple[int, int, int]]) -> float | None:
    """Average length (px) of the visible limbs, each measured joint to joint."""
    lengths = [
        _dist(pose[a].xy, pose[b].xy) + _dist(pose[b].xy, pose[c].xy)
        for a, b, c in chains
        if _visible(pose, a, b, c)
    ]
    return float(np.mean(lengths)) if lengths else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def measure(pose: Pose, height_cm: float) -> MeasurementResult:
    warnings: list[str] = []
    penalty = 1.0  # multiplied into the confidence for every fallback/problem
    lines: list[DebugLine] = []

    shoulder_l, shoulder_r = pose[L.LEFT_SHOULDER].xy, pose[L.RIGHT_SHOULDER].xy
    shoulders = _mid(pose, L.LEFT_SHOULDER, L.RIGHT_SHOULDER)
    hips = _mid(pose, L.LEFT_HIP, L.RIGHT_HIP)
    shoulder_px = _dist(shoulder_l, shoulder_r)
    torso_px = _dist(shoulders, hips)

    # ---- 1. Scale -------------------------------------------------------------
    head_top, head_ok = _head_top_y(pose, shoulder_px)
    sole, feet_ok = _sole_y(pose, shoulder_px, head_top)
    if not head_ok:
        warnings.append(WARN_HEAD)
        penalty *= 0.9
    if not feet_ok:
        warnings.append(WARN_FEET)
        penalty *= 0.9

    body_px = sole - head_top
    cm_per_px = height_cm / body_px
    x_mid = float(shoulders[0])
    lines.append(DebugLine((x_mid, head_top), (x_mid, sole), f"height {height_cm:.0f}"))

    # ---- 2. Lengths --------------------------------------------------------------
    shoulder_cm = shoulder_px * cm_per_px * SHOULDER_WIDTH_FACTOR
    torso_cm = torso_px * cm_per_px
    lines.append(DebugLine(tuple(shoulder_r), tuple(shoulder_l), f"shoulder {shoulder_cm:.1f}"))
    lines.append(DebugLine(tuple(shoulders), tuple(hips), f"torso {torso_cm:.1f}"))

    arm_px = _limb_px(
        pose,
        [(L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST), (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST)],
    )
    if arm_px is None:
        arm_cm = ARM_PER_HEIGHT * height_cm
        warnings.append(WARN_ARM_LENGTH)
        penalty *= 0.95
    else:
        arm_cm = arm_px * cm_per_px

    leg_px = _limb_px(
        pose, [(L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE), (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE)]
    )
    if leg_px is None:
        leg_cm = LEG_PER_HEIGHT * height_cm
        warnings.append(WARN_LEG_LENGTH)
        penalty *= 0.95
    else:
        leg_cm = leg_px * cm_per_px

    # ---- 3/4. Chest and waist -------------------------------------------------------
    chest_px, chest_run = _breadth_px(pose, CHEST_LEVEL, shoulder_px)
    if chest_px is None:
        chest_breadth_cm = CHEST_BREADTH_PER_SHOULDER * shoulder_cm
        warnings.append(WARN_ARMS_CHEST)
        penalty *= 0.85
    else:
        chest_breadth_cm = chest_px * cm_per_px
    chest_cm = CHEST_CIRC_PER_BREADTH * chest_breadth_cm

    waist_px, waist_run = _breadth_px(pose, WAIST_LEVEL, shoulder_px)
    if waist_px is None:
        waist_breadth_cm = WAIST_BREADTH_PER_SHOULDER * shoulder_cm
        warnings.append(WARN_ARMS_WAIST)
        penalty *= 0.85
    else:
        waist_breadth_cm = waist_px * cm_per_px
    waist_cm = WAIST_CIRC_PER_BREADTH * waist_breadth_cm

    for run, name, value, used in ((chest_run, "chest", chest_cm, chest_px), (waist_run, "waist", waist_cm, waist_px)):
        if run:
            left, right, y = run
            label = f"{name} {value:.1f}" if used else f"{name} (estimated) {value:.1f}"
            lines.append(DebugLine((left, y), (right, y), label))

    # ---- Pose quality -------------------------------------------------------------
    # Facing the camera, shoulder width is ~0.6–0.8 × torso length. Much less means the
    # body is turned sideways, so every width is too small.
    if shoulder_px / max(torso_px, 1.0) < 0.45:
        warnings.append(WARN_TURNED)
        penalty *= 0.8
    # Small in the frame = few pixels per cm = coarse measurements
    if body_px < 0.5 * pose.height:
        warnings.append(WARN_SMALL)
        penalty *= 0.9

    used = [
        L.NOSE, L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_ELBOW, L.RIGHT_ELBOW, L.LEFT_WRIST, L.RIGHT_WRIST,
        L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE,
    ]  # fmt: skip
    visibility = float(np.mean([pose[i].visibility for i in used]))
    confidence = round(float(np.clip(visibility * penalty, 0.0, 1.0)), 2)

    measurements = Measurements(
        shoulder_cm=round(shoulder_cm, 1),
        chest_cm=round(chest_cm, 1),
        waist_cm=round(waist_cm, 1),
        torso_cm=round(torso_cm, 1),
        arm_cm=round(arm_cm, 1),
        leg_cm=round(leg_cm, 1),
    )
    return MeasurementResult(measurements, confidence, warnings, cm_per_px, lines)

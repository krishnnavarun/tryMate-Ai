"""Body measurements + a product's size chart → a score and a fit note for every size.

HOW IT WORKS
============

1. Which body measurement each chart field is compared with (FIELDS below)
     chest    → chest_cm                      (circumference)
     waist    → waist_cm                      (circumference)
     shoulder → shoulder_cm                   (width)
     length   → torso_cm × LENGTH_PER_TORSO   (upper body / dresses: ideal garment length)
     inseam   → leg_cm                        (lower body)
   Fields we can't estimate from a photo (e.g. hip) are ignored.

2. The ideal point inside each range depends on the fit preference
   A chart range [min, max] says "this size fits bodies from min to max cm".
     regular → ideal = middle of the range
     slim    → ideal = 75% of the way up: the body fills the garment → snug
     loose   → ideal = 25% of the way up: the garment has room to spare
   (Length ignores the preference: slim or loose, sleeves and hems shouldn't change.)

3. Field closeness: a bell curve around the ideal
       exp(−½ × (d / σ)²),   d = body measurement − ideal
   σ ("sigma", in cm) is how quickly the score drops. It's close to the measurement error
   of /analyze (±4 cm for chest/waist), so a size isn't ruled out by noise alone:
       d = 0 → 1.00 · d = σ → 0.61 · d = 2σ → 0.14

4. Size score: the same bell curve, applied to the weighted average of the fields'
   squared deviations:   score = exp(−½ × Σ wᵢ dᵢ² / Σ wᵢ)   (dᵢ in units of σ)
   The CATEGORY decides the weights: a top is mostly about chest and shoulders, trousers
   about the waist, and so on. Weights are re-normalised over the fields this chart has.
   (Combining before the curve keeps sizes correctly ordered even when the person is far
   outside every size.)

5. Recommended size = highest score. On an exact tie, the larger size wins (a slightly
   loose top is usually better than a tight one).

6. Notes: for each field whose deviation is more than NOTE_THRESHOLD σ, say which way it's
   off. Body bigger than the ideal → "tight at …" (for length: "short"); smaller →
   "loose at …" ("long"). Deviations under 1.5σ get "slightly". No issues → "Good fit".
       e.g. "Tight at chest and shoulders", "Loose at shoulders, slightly long"
"""

import math
from dataclasses import dataclass

from app.errors import AppError, ErrorCode
from app.schemas import Measurements, RecommendSizeRequest, RecommendSizeResponse, SizeFit

# Ideal upper-body garment length ≈ 1.55 × torso_cm (shoulder landmarks → hip landmarks).
# Starting value from a test photo; tune it once you've compared real users and garments.
LENGTH_PER_TORSO = 1.55


@dataclass(frozen=True)
class Field:
    label: str  # used in notes: "tight at <label>"
    sigma: float  # cm
    girth: bool  # True: fit preference applies and notes say tight/loose; False: length (short/long)


FIELDS: dict[str, Field] = {
    "chest": Field("chest", sigma=4.0, girth=True),
    "waist": Field("waist", sigma=4.0, girth=True),
    "shoulder": Field("shoulders", sigma=2.0, girth=True),
    "length": Field("length", sigma=4.0, girth=False),
    "inseam": Field("inseam", sigma=3.0, girth=False),
}

# How much each field matters, per category
WEIGHTS: dict[str, dict[str, float]] = {
    "upper_body": {"chest": 0.45, "shoulder": 0.25, "waist": 0.15, "length": 0.15},
    "lower_body": {"waist": 0.6, "inseam": 0.4},
    "dresses": {"chest": 0.35, "waist": 0.35, "shoulder": 0.1, "length": 0.2},
}

IDEAL_POSITION = {"slim": 0.75, "regular": 0.5, "loose": 0.25}

NOTE_THRESHOLD = 0.75  # in σ: smaller deviations aren't worth mentioning
STRONG_THRESHOLD = 1.5  # in σ: from here on, no "slightly"


def body_value(field: str, m: Measurements, category: str) -> float | None:
    """The body measurement to compare with a chart field, or None if we don't have one."""
    if field == "chest":
        return m.chest_cm
    if field == "waist":
        return m.waist_cm
    if field == "shoulder":
        return m.shoulder_cm
    if field == "length" and category in ("upper_body", "dresses"):
        return m.torso_cm * LENGTH_PER_TORSO
    if field == "inseam" and category == "lower_body":
        return m.leg_cm
    return None


def ideal_point(low: float, high: float, field: Field, fit_preference: str) -> float:
    position = IDEAL_POSITION[fit_preference] if field.girth else 0.5
    return low + position * (high - low)


@dataclass
class FieldResult:
    name: str
    deviation_sigmas: float  # (body − ideal) / σ; positive = body bigger than ideal
    score: float


def score_size(ranges: dict[str, list[float]], m: Measurements, category: str, fit: str) -> tuple[float, list[FieldResult]]:
    weights = WEIGHTS[category]
    results: list[tuple[float, FieldResult]] = []
    for name, (low, high) in ranges.items():
        field = FIELDS.get(name)
        value = body_value(name, m, category)
        weight = weights.get(name, 0.0)
        if field is None or value is None or weight == 0:
            continue
        d = (value - ideal_point(low, high, field, fit)) / field.sigma
        results.append((weight, FieldResult(name, d, math.exp(-0.5 * d * d))))

    total_weight = sum(w for w, _ in results)
    if total_weight == 0:
        return 0.0, []
    # Combine the fields' squared deviations first, then apply the bell curve once.
    # (Averaging the field scores instead flattens out: when a person is far from every
    # size, all chest scores are ~0 and a minor field like length would pick the size.)
    mean_squared = sum(w * r.deviation_sigmas**2 for w, r in results) / total_weight
    return math.exp(-0.5 * mean_squared), [r for _, r in results]


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def fit_note(results: list[FieldResult]) -> str:
    if not results:
        return "No comparable measurements in this size chart"

    groups: dict[str, list[str]] = {}  # e.g. {"Tight at": ["chest", "shoulders"], "slightly long": []}
    for r in results:
        if abs(r.deviation_sigmas) < NOTE_THRESHOLD:
            continue
        field = FIELDS[r.name]
        slightly = "slightly " if abs(r.deviation_sigmas) < STRONG_THRESHOLD else ""
        if field.girth:
            word = "tight" if r.deviation_sigmas > 0 else "loose"
            groups.setdefault(f"{slightly}{word} at", []).append(field.label)
        else:
            word = "short" if r.deviation_sigmas > 0 else "long"
            label = f"{slightly}{word}" if r.name == "length" else f"{slightly}{word} in the {field.label}"
            groups.setdefault(label, [])

    if not groups:
        return "Good fit"
    parts = [f"{prefix} {_join(labels)}" if labels else prefix for prefix, labels in groups.items()]
    note = ", ".join(parts)
    return note[0].upper() + note[1:]


def recommend_size(request: RecommendSizeRequest) -> RecommendSizeResponse:
    per_size: dict[str, SizeFit] = {}
    best_size, best_score = None, -1.0
    comparable = False

    for size, ranges in request.size_chart.items():
        score, results = score_size(ranges, request.measurements, request.category, request.fit_preference)
        comparable = comparable or bool(results)
        per_size[size] = SizeFit(score=round(score, 2), note=fit_note(results))
        # ">=" so that on a tie the later (larger) size wins
        if score >= best_score:
            best_size, best_score = size, score

    if not comparable:
        usable = ", ".join(name for name, w in WEIGHTS[request.category].items() if w > 0)
        raise AppError(
            ErrorCode.INVALID_INPUT,
            f"size_chart has no fields that can be compared for {request.category} (use: {usable}).",
        )
    return RecommendSizeResponse(recommended_size=best_size, per_size=per_size)

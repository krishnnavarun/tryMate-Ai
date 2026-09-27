"""Body measurements + a product's size chart → a score, a fit note and a fit breakdown per size.

HOW IT WORKS
============

1. Which body measurement each chart field is compared with (FIELDS below)
     chest    → chest_cm                      (circumference)
     waist    → waist_cm                      (circumference)
     shoulder → shoulder_cm                   (width)
     length   → torso_cm × LENGTH_PER_TORSO   (upper body / dresses: ideal garment length)
     sleeve   → arm_cm × SLEEVE_PER_ARM       (upper body: ideal sleeve length, long sleeves)
     inseam   → leg_cm                        (lower body)
   Fields we can't estimate from a photo (e.g. hip) are ignored. `length` and `sleeve` are
   GARMENT measurements (collar/shoulder seam → hem/cuff); the others are body measurements.

2. The ideal point inside each range depends on the fit preference
   A chart range [min, max] says "this size fits bodies from min to max cm".
     regular → ideal = middle of the range
     slim    → ideal = 75% of the way up: the body fills the garment → snug
     loose   → ideal = 25% of the way up: the garment has room to spare
   (Lengths ignore the preference: slim or loose, sleeves and hems shouldn't change.)

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
   Between sizes: when the NEIGHBOURING size scores at least ALTERNATIVE_RATIO of the best,
   it's returned as `alternative_size`, with a note saying which one fits closer.

6. Notes and breakdown: for each field whose deviation is more than NOTE_THRESHOLD σ, say
   which way it's off. Body bigger than the ideal → "tight at …" (lengths: "short"); smaller
   → "loose at …" ("long"). Under 1.5σ gets "slightly". No issues → "Good fit".
       e.g. "Tight at chest and shoulders", "Loose at shoulders, slightly long"
   Every size also gets `fields`: the numbers behind the note (body vs. range, the ideal
   point, the difference in cm and a verdict), so the shop can show exactly where and by
   how much a size is off.
"""

import math
from dataclasses import dataclass

from app.errors import AppError, ErrorCode
from app.schemas import FieldFit, Measurements, RecommendSizeRequest, RecommendSizeResponse, SizeFit

# Ideal upper-body garment length ≈ 1.55 × torso_cm (shoulder landmarks → hip landmarks).
# Starting value from a test photo; scripts/calibrate.py suggests a better one.
LENGTH_PER_TORSO = 1.55
# Ideal long-sleeve length ≈ 1.06 × arm_cm. arm_cm runs from the shoulder JOINT to the wrist
# joint; a sleeve starts at the shoulder seam (further out) and ends just past the wrist.
# Starting value (≈ 3–4 cm on an average arm); scripts/calibrate.py suggests a better one.
SLEEVE_PER_ARM = 1.06


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
    "sleeve": Field("sleeves", sigma=2.5, girth=False),
    "inseam": Field("inseam", sigma=3.0, girth=False),
}

# How much each field matters, per category (re-normalised over the fields a chart has,
# so a chart without sleeves scores exactly as before)
WEIGHTS: dict[str, dict[str, float]] = {
    "upper_body": {"chest": 0.45, "shoulder": 0.25, "waist": 0.15, "length": 0.15, "sleeve": 0.12},
    "lower_body": {"waist": 0.6, "inseam": 0.4},
    "dresses": {"chest": 0.35, "waist": 0.35, "shoulder": 0.1, "length": 0.2},
}

IDEAL_POSITION = {"slim": 0.75, "regular": 0.5, "loose": 0.25}

NOTE_THRESHOLD = 0.75  # in σ: smaller deviations aren't worth mentioning
STRONG_THRESHOLD = 1.5  # in σ: from here on, no "slightly"
# A neighbouring size scoring at least this share of the best one is "between sizes"
ALTERNATIVE_RATIO = 0.75


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
    if field == "sleeve" and category == "upper_body":
        return m.arm_cm * SLEEVE_PER_ARM
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
    value: float = 0.0  # the body value compared
    low: float = 0.0
    high: float = 0.0
    ideal: float = 0.0


def verdict(name: str, deviation_sigmas: float) -> str:
    """good / slightly_tight / tight / slightly_loose / loose (girths),
    good / slightly_short / short / slightly_long / long (lengths)."""
    size = abs(deviation_sigmas)
    if size < NOTE_THRESHOLD:
        return "good"
    strength = "slightly_" if size < STRONG_THRESHOLD else ""
    if FIELDS[name].girth:
        return strength + ("tight" if deviation_sigmas > 0 else "loose")
    return strength + ("short" if deviation_sigmas > 0 else "long")


def score_size(ranges: dict[str, list[float]], m: Measurements, category: str, fit: str) -> tuple[float, list[FieldResult]]:
    weights = WEIGHTS[category]
    results: list[tuple[float, FieldResult]] = []
    for name, (low, high) in ranges.items():
        field = FIELDS.get(name)
        value = body_value(name, m, category)
        weight = weights.get(name, 0.0)
        if field is None or value is None or weight == 0:
            continue
        ideal = ideal_point(low, high, field, fit)
        d = (value - ideal) / field.sigma
        results.append((weight, FieldResult(name, d, math.exp(-0.5 * d * d), value, low, high, ideal)))

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


def field_fits(results: list[FieldResult]) -> list[FieldFit]:
    return [
        FieldFit(
            field=r.name,
            label=FIELDS[r.name].label,
            body_cm=round(r.value, 1),
            size_min=r.low,
            size_max=r.high,
            ideal_cm=round(r.ideal, 1),
            difference_cm=round(r.value - r.ideal, 1),
            verdict=verdict(r.name, r.deviation_sigmas),
        )
        for r in results
    ]


def between_sizes(sizes: list[str], scores: dict[str, float], best: str) -> tuple[str | None, str | None]:
    """(alternative size, note) when a neighbouring size fits almost as well as the best one."""
    i = sizes.index(best)
    neighbours = [sizes[j] for j in (i - 1, i + 1) if 0 <= j < len(sizes)]
    if not neighbours or scores[best] <= 0:
        return None, None
    other = max(neighbours, key=lambda s: scores[s])
    if scores[other] < ALTERNATIVE_RATIO * scores[best]:
        return None, None
    larger = sizes.index(other) > i
    smaller_size, larger_size = (best, other) if larger else (other, best)
    feel = "fits more relaxed" if larger else "fits closer"
    return other, f"You're between {smaller_size} and {larger_size}: {best} is the closer match; {other} {feel}."


def recommend_size(request: RecommendSizeRequest) -> RecommendSizeResponse:
    per_size: dict[str, SizeFit] = {}
    scores: dict[str, float] = {}
    best_size, best_score = None, -1.0
    comparable = False

    for size, ranges in request.size_chart.items():
        score, results = score_size(ranges, request.measurements, request.category, request.fit_preference)
        comparable = comparable or bool(results)
        per_size[size] = SizeFit(score=round(score, 2), note=fit_note(results), fields=field_fits(results))
        scores[size] = score
        # ">=" so that on a tie the later (larger) size wins
        if score >= best_score:
            best_size, best_score = size, score

    if not comparable:
        usable = ", ".join(name for name, w in WEIGHTS[request.category].items() if w > 0)
        raise AppError(
            ErrorCode.INVALID_INPUT,
            f"size_chart has no fields that can be compared for {request.category} (use: {usable}).",
        )
    alternative, note = between_sizes(list(request.size_chart), scores, best_size)
    return RecommendSizeResponse(
        recommended_size=best_size,
        per_size=per_size,
        alternative_size=alternative,
        alternative_note=note,
    )

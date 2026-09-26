"""Calibrate the measurements against a tape measure: photos + tape values -> which constants to change.

Usage:
    python scripts/calibrate.py calibration/samples.csv
    python scripts/calibrate.py calibration/samples.csv --debug-dir calibration/debug

samples.csv (copy scripts/calibration_template.csv) has one row per photo:
    photo            full-body photo, path relative to the CSV file          (required)
    height_cm        your height without shoes                               (required)
    shoulder_cm      tape across the back, from one shoulder tip to the other
    chest_cm         tape around the fullest part of the chest, under the arms
    waist_cm         tape around the waist at navel height
    shirt_length_cm  a T-shirt that fits you well, laid flat: from the highest point of the
                     shoulder (next to the collar) straight down to the hem
    undertone        warm | neutral | cool, if you know it
Leave a cell empty if you don't have that value.

It runs the same pipeline as /analyze on each photo, on this computer (nothing is uploaded,
and nothing is saved unless you pass --debug-dir), compares with your tape values and prints
which constants to change:

    shoulder   SHOULDER_WIDTH_FACTOR                        app/services/measurements.py
    chest      CHEST_CALIBRATION                            app/services/measurements.py
    waist      WAIST_CALIBRATION                            app/services/measurements.py
    length     LENGTH_PER_TORSO                             app/services/sizing.py
    undertone  UNDERTONE_COOL_BELOW / UNDERTONE_WARM_ABOVE  app/services/skin_tone.py

Each measurement scales linearly with its constant, so the suggestion is
    current value x median(tape / measured)
The median keeps one odd photo from pulling it far. Take 3-5 photos, change the constants,
run the script again: the errors should shrink. CHEST_LEVEL, WAIST_LEVEL and DIP_THRESHOLD
aren't suggested: with a handful of photos, tuning them would only fit the noise.
"""

import argparse
import base64
import csv
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Run as `python scripts/calibrate.py` from the repo root: make `import app` work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.errors import AppError  # noqa: E402
from app.services import measurements, sizing, skin_tone  # noqa: E402
from app.services.debug_image import render_debug_image  # noqa: E402
from app.services.face import FaceFinder  # noqa: E402
from app.services.image_io import decode_image  # noqa: E402
from app.services.pose import PoseDetector, select_single_full_body  # noqa: E402

TAPE_COLUMNS = ("shoulder_cm", "chest_cm", "waist_cm", "shirt_length_cm")
UNDERTONES = ("warm", "neutral", "cool")

MIN_PHOTOS = 3  # fewer than this: suggestions are shown, with a warning
KEEP_WITHIN = 0.01  # a change smaller than 1% isn't worth making
SUSPICIOUS_CHANGE = 0.15  # more than 15% usually means a tape or photo problem, not a wrong constant
MAX_SPREAD = 0.08  # photos whose tape/measured ratios differ by more than 8% disagree
UNDERTONE_MARGIN = 0.5  # degrees between a labelled photo's hue and a suggested threshold


@dataclass
class Sample:
    photo: Path
    height_cm: float
    tape: dict[str, float]  # only the columns that were filled in
    undertone: str | None = None


@dataclass
class PhotoResult:
    sample: Sample
    measured: dict[str, float] = field(default_factory=dict)  # shoulder_cm, chest_cm, waist_cm, torso_cm
    estimated: set[str] = field(default_factory=set)  # chest/waist estimated from shoulder width
    hue: float | None = None
    undertone: str | None = None  # what the service says with the current thresholds
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class Suggestion:
    label: str  # "shoulder", "chest", ...
    constant: str
    where: str
    current: float
    suggested: float | None = None  # None: no tape values for it
    photos: int = 0
    bias_cm: float | None = None  # mean of (measured − tape): + = measures too big
    error_now_cm: float | None = None  # mean absolute error with the current value
    error_after_cm: float | None = None  # ... with the suggested value, on these photos
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reading samples.csv
# ---------------------------------------------------------------------------


def load_samples(csv_path: Path) -> list[Sample]:
    """Read samples.csv. Lines starting with # are comments. Raises ValueError with a readable message."""
    text = csv_path.read_text(encoding="utf-8-sig")  # -sig: Excel adds a BOM
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    missing = {"photo", "height_cm"} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{csv_path.name} needs the columns: {', '.join(sorted(missing))} (comma-separated)")

    samples = []
    for row_number, row in enumerate(reader, start=1):

        def number(column: str) -> float | None:
            value = (row.get(column) or "").strip()
            if not value:
                return None
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"row {row_number}: {column} must be a number, not {value!r}") from None

        height = number("height_cm")
        if height is None:
            raise ValueError(f"row {row_number}: height_cm is required")
        tape = {column: value for column in TAPE_COLUMNS if (value := number(column)) is not None}
        undertone = (row.get("undertone") or "").strip().lower() or None
        if undertone and undertone not in UNDERTONES:
            raise ValueError(f"row {row_number}: undertone must be one of {', '.join(UNDERTONES)}, not {undertone!r}")
        samples.append(Sample(csv_path.parent / row["photo"].strip(), height, tape, undertone))

    if not samples:
        raise ValueError(f"{csv_path.name} has no photo rows")
    return samples


# ---------------------------------------------------------------------------
# Measuring one photo (the /analyze pipeline, without the web part)
# ---------------------------------------------------------------------------


def measure_photo(sample: Sample, detector, face_finder, debug_dir: Path | None = None) -> PhotoResult:
    result = PhotoResult(sample)
    try:
        image = decode_image(sample.photo.read_bytes())
        pose = select_single_full_body(detector.detect(image))
    except FileNotFoundError:
        result.error = f"file not found: {sample.photo}"
        return result
    except AppError as err:
        result.error = f"{err.code.value}: {err.message}"
        return result

    body = measurements.measure(pose, sample.height_cm)
    m = body.measurements
    result.measured = {"shoulder_cm": m.shoulder_cm, "chest_cm": m.chest_cm, "waist_cm": m.waist_cm, "torso_cm": m.torso_cm}
    if measurements.WARN_ARMS_CHEST in body.warnings:
        result.estimated.add("chest_cm")
    if measurements.WARN_ARMS_WAIST in body.warnings:
        result.estimated.add("waist_cm")
    result.warnings = list(body.warnings)

    face = face_finder.find(image, pose)
    skin = skin_tone.analyze_skin(image, face) if face is not None else None
    circles = []
    if skin is not None:
        result.hue, result.undertone = skin.hue, skin.skin_tone.undertone
        result.warnings += skin.warnings
        circles = skin.sample_circles

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        jpeg = base64.b64decode(render_debug_image(image, pose, body, circles))
        (debug_dir / f"{sample.photo.stem}-debug.jpg").write_bytes(jpeg)
    return result


# ---------------------------------------------------------------------------
# Suggestions (pure maths, unit-tested in tests/test_calibrate.py)
# ---------------------------------------------------------------------------


def scale_suggestion(suggestion: Suggestion, pairs: list[tuple[float, float]]) -> Suggestion:
    """Fill in `suggestion` from (measured, tape) pairs, for a constant the measurement scales with."""
    suggestion.photos = len(pairs)
    if not pairs:
        suggestion.notes.append("no measured photo with a tape value for it")
        return suggestion

    ratios = [tape / measured for measured, tape in pairs]
    k = statistics.median(ratios)
    suggestion.suggested = round(suggestion.current * k, 3)
    suggestion.bias_cm = statistics.fmean(measured - tape for measured, tape in pairs)
    suggestion.error_now_cm = statistics.fmean(abs(measured - tape) for measured, tape in pairs)
    suggestion.error_after_cm = statistics.fmean(abs(measured * k - tape) for measured, tape in pairs)

    if len(pairs) < MIN_PHOTOS:
        suggestion.notes.append(f"only {len(pairs)} photo(s): take {MIN_PHOTOS}-5 before changing anything")
    spread = max(ratios) / min(ratios) - 1
    if spread > MAX_SPREAD:
        suggestion.notes.append(
            f"the photos disagree by {spread:.0%}: check pose and clothing in the debug images first"
        )
    if abs(k - 1) < KEEP_WITHIN:
        suggestion.suggested = suggestion.current
        suggestion.notes.append("already within 1%: keep it")
    elif abs(k - 1) > SUSPICIOUS_CHANGE:
        suggestion.notes.append(
            f"changing it by {abs(k - 1):.0%} is suspicious: re-check the tape value and the debug image first"
        )
    return suggestion


def suggest_measurements(results: list[PhotoResult]) -> list[Suggestion]:
    measured = [r for r in results if r.error is None]

    def pairs(key: str, tape_key: str | None = None, factor: float = 1.0) -> list[tuple[float, float]]:
        tape_key = tape_key or key
        return [
            (r.measured[key] * factor, r.sample.tape[tape_key])
            for r in measured
            if tape_key in r.sample.tape and key not in r.estimated
        ]

    suggestions = [
        scale_suggestion(
            Suggestion("shoulder", "SHOULDER_WIDTH_FACTOR", "measurements.py", measurements.SHOULDER_WIDTH_FACTOR),
            pairs("shoulder_cm"),
        ),
        scale_suggestion(
            Suggestion("chest", "CHEST_CALIBRATION", "measurements.py", measurements.CHEST_CALIBRATION),
            pairs("chest_cm"),
        ),
        scale_suggestion(
            Suggestion("waist", "WAIST_CALIBRATION", "measurements.py", measurements.WAIST_CALIBRATION),
            pairs("waist_cm"),
        ),
        # Ideal garment length = torso_cm × LENGTH_PER_TORSO, compared with a shirt that fits
        scale_suggestion(
            Suggestion("length", "LENGTH_PER_TORSO", "sizing.py", sizing.LENGTH_PER_TORSO),
            pairs("torso_cm", "shirt_length_cm", sizing.LENGTH_PER_TORSO),
        ),
    ]
    for label, key in (("chest", "chest_cm"), ("waist", "waist_cm")):
        skipped = sum(1 for r in measured if key in r.estimated and key in r.sample.tape)
        if skipped:
            note = f"{skipped} photo(s) left out: {label} was estimated from shoulder width (arms too close)"
            next(s for s in suggestions if s.label == label).notes.append(note)
    return suggestions


def undertone_thresholds(
    labelled: list[tuple[float, str]], cool_below: float, warm_above: float
) -> tuple[float, float] | str:
    """Thresholds that give every labelled photo its label, as close as possible to the current ones.

    labelled: (hue, "warm" | "neutral" | "cool"). The service says cool if hue < cool_below,
    warm if hue > warm_above, neutral in between. Returns (cool_below, warm_above), or a
    message when no thresholds can satisfy all the labels.
    """
    cool = [h for h, u in labelled if u == "cool"]
    neutral = [h for h, u in labelled if u == "neutral"]
    warm = [h for h, u in labelled if u == "warm"]

    # cool: hue < C · neutral: C <= hue <= W · warm: hue > W
    c_low = max(cool) + UNDERTONE_MARGIN if cool else -math.inf
    c_high = min(neutral) if neutral else math.inf
    w_low = max(neutral) if neutral else -math.inf
    w_high = min(warm) - UNDERTONE_MARGIN if warm else math.inf
    if c_low > c_high or w_low > w_high or c_low > w_high:
        return (
            "the labels overlap (e.g. a photo labelled cool has a higher hue than one labelled "
            "neutral or warm): the lighting differs too much between photos, or a label is off"
        )

    c = min(max(cool_below, c_low), c_high)
    w = min(max(warm_above, w_low), w_high)
    if c > w:  # e.g. cool moved up past the warm threshold: raise warm to match
        w = c
    return round(c, 1), round(w, 1)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _row(label: str, tape: float | None, measured: float, extra: str = "", names=("tape", "measured")) -> str:
    tape_name, measured_name = names
    if tape is None:
        return f"    {label:<9} {measured_name} {measured:6.1f}{extra}"
    return (
        f"    {label:<9} {tape_name} {tape:6.1f}   {measured_name} {measured:6.1f}"
        f"   ({measured - tape:+.1f} cm){extra}"
    )


def print_photo(result: PhotoResult) -> None:
    sample = result.sample
    print(f"\n  {sample.photo.name}  (height {sample.height_cm:g} cm)")
    if result.error:
        print(f"    SKIPPED: {result.error}")
        return
    tape, m = sample.tape, result.measured
    for label, key in (("shoulder", "shoulder_cm"), ("chest", "chest_cm"), ("waist", "waist_cm")):
        extra = "   [estimated from shoulders]" if key in result.estimated else ""
        print(_row(label, tape.get(key), m[key], extra))
    ideal = m["torso_cm"] * sizing.LENGTH_PER_TORSO
    extra = f"   (torso {m['torso_cm']:.1f} x {sizing.LENGTH_PER_TORSO})"
    print(_row("length", tape.get("shirt_length_cm"), ideal, extra, names=("shirt", "ideal")))
    if result.hue is None:
        print("    undertone: no face found")
    else:
        you = f"you say {sample.undertone}, " if sample.undertone else ""
        print(f"    undertone {you}service says {result.undertone} (hue {result.hue:.1f})")
    for warning in result.warnings:
        print(f"    ! {warning}")


def print_suggestions(suggestions: list[Suggestion], results: list[PhotoResult]) -> None:
    print("\nSUGGESTED VALUES  (error = average difference from your tape, on these photos)")
    for s in suggestions:
        print(f"\n  {s.label}: {s.constant}  (app/services/{s.where})")
        if s.suggested is None:
            print(f"    keep {s.current}")
            for note in s.notes:
                print(f"    note: {note}")
            continue
        if abs(s.bias_cm) < 0.05:
            bias = "no bias on average"
        else:
            bias = f"measures {abs(s.bias_cm):.1f} cm too {'big' if s.bias_cm > 0 else 'small'} on average"
        print(f"    now {s.current}  ->  suggested {s.suggested}")
        print(
            f"    {s.photos} photo(s): {bias}; "
            f"error {s.error_now_cm:.1f} cm now -> {s.error_after_cm:.1f} cm with the suggestion"
        )
        for note in s.notes:
            print(f"    note: {note}")

    labelled = [(r.hue, r.sample.undertone) for r in results if r.hue is not None and r.sample.undertone]
    current = (skin_tone.UNDERTONE_COOL_BELOW, skin_tone.UNDERTONE_WARM_ABOVE)
    print("\n  undertone: UNDERTONE_COOL_BELOW / UNDERTONE_WARM_ABOVE  (app/services/skin_tone.py)")
    if not labelled:
        print("    keep them: no photos with an undertone label and a visible face")
        return
    suggested = undertone_thresholds(labelled, *current)
    wrong = sum(1 for hue, label in labelled if skin_tone.classify_undertone(hue) != label)
    if isinstance(suggested, str):
        print(f"    keep {current[0]} / {current[1]}: {suggested}")
    elif wrong == 0:
        print(f"    keep {current[0]} / {current[1]}: all {len(labelled)} labelled photo(s) already match")
    else:
        print(f"    now {current[0]} / {current[1]}  ->  suggested {suggested[0]} / {suggested[1]}")
        print(f"    {wrong} of {len(labelled)} labelled photo(s) get a different undertone now")
        if len(labelled) < MIN_PHOTOS:
            print(f"    note: only {len(labelled)} photo(s): skin tone depends a lot on light, take {MIN_PHOTOS}+ in daylight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", type=Path, help="CSV file with one row per photo")
    parser.add_argument("--debug-dir", type=Path, help="also save a debug image per photo (what was measured) here")
    parser.add_argument("--pose-model", default="heavy", choices=["lite", "full", "heavy"])
    args = parser.parse_args()
    # The service's warnings contain characters like "—"; Windows would print them in its
    # old code page when the output is piped or redirected to a file
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        samples = load_samples(args.samples)
        detector, face_finder = PoseDetector(args.pose_model), FaceFinder()
    except (OSError, ValueError, AppError) as err:
        print(f"ERROR: {getattr(err, 'message', err)}", file=sys.stderr)
        return 1

    print(f"Measuring {len(samples)} photo(s) from {args.samples} (pose model: {args.pose_model})")
    results = [measure_photo(sample, detector, face_finder, args.debug_dir) for sample in samples]
    for result in results:
        print_photo(result)
    if all(r.error for r in results):
        print("\nNo photo could be measured: see the reasons above.", file=sys.stderr)
        return 1

    print_suggestions(suggest_measurements(results), results)
    print("\nChange the constants, then run this again: the errors should shrink.")
    if args.debug_dir:
        print(f"Debug images: {args.debug_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

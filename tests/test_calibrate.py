"""scripts/calibrate.py: reading samples.csv, the suggestion maths, and measuring a photo.

The photo tests use the fake detector (paper-doll person), so no MediaPipe model is needed.
"""

import pytest

from app.services import measurements, sizing
from scripts.calibrate import (
    PhotoResult,
    Sample,
    Suggestion,
    load_samples,
    measure_photo,
    scale_suggestion,
    suggest_measurements,
    undertone_thresholds,
)
from tests.fakes import FakeDetector, FakeFaceFinder, make_face, make_pose

# ---- samples.csv ---------------------------------------------------------------------------


def write_csv(tmp_path, text: str):
    path = tmp_path / "samples.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_rows_skipping_comments_and_empty_cells(tmp_path):
    path = write_csv(
        tmp_path,
        "﻿# a comment line\n"  # Excel's BOM
        "photo,height_cm,shoulder_cm,chest_cm,waist_cm,shirt_length_cm,undertone\n"
        "a.jpg,178,46,98,86,72,Warm\n"
        "\n"
        "b.jpg,178.5,,99,,,\n",
    )
    first, second = load_samples(path)
    assert first.photo == tmp_path / "a.jpg"
    assert first.tape == {"shoulder_cm": 46, "chest_cm": 98, "waist_cm": 86, "shirt_length_cm": 72}
    assert first.undertone == "warm"
    assert second.height_cm == 178.5
    assert second.tape == {"chest_cm": 99}
    assert second.undertone is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("photo,chest_cm\na.jpg,98\n", "needs the columns: height_cm"),
        ("photo,height_cm\na.jpg,\n", "row 1: height_cm is required"),
        ("photo,height_cm,chest_cm\na.jpg,178,ninety\n", "row 1: chest_cm must be a number"),
        ("photo,height_cm,undertone\na.jpg,178,olive\n", "undertone must be one of"),
        ("# only comments\nphoto,height_cm\n", "has no photo rows"),
    ],
)
def test_explains_what_is_wrong_with_the_csv(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_samples(write_csv(tmp_path, text))


# ---- the suggestion maths ------------------------------------------------------------------------


def test_suggests_the_factor_that_removes_a_consistent_error():
    # Every photo measures 5% too small → the factor should go up 5%, and the error to ~0
    pairs = [(40.0, 42.0), (42.0, 44.1), (44.0, 46.2)]
    s = scale_suggestion(Suggestion("shoulder", "SHOULDER_WIDTH_FACTOR", "measurements.py", 1.15), pairs)
    assert s.suggested == round(1.15 * 1.05, 3)
    assert s.photos == 3
    assert s.bias_cm == pytest.approx(-2.1)
    assert s.error_now_cm == pytest.approx(2.1)
    assert s.error_after_cm == pytest.approx(0, abs=1e-9)
    assert s.notes == []


def test_the_median_ignores_one_odd_photo():
    pairs = [(100.0, 100.0), (100.0, 101.0), (100.0, 130.0)]  # the last photo is way off
    s = scale_suggestion(Suggestion("chest", "CHEST_CALIBRATION", "measurements.py", 1.0), pairs)
    assert s.suggested == 1.01
    assert any("disagree by 30%" in note for note in s.notes)


def test_warns_with_too_few_photos_and_keeps_tiny_changes():
    s = scale_suggestion(Suggestion("waist", "WAIST_CALIBRATION", "measurements.py", 1.0), [(86.0, 86.5)])
    assert s.suggested == 1.0  # 0.6% off: not worth changing
    assert any("only 1 photo" in note for note in s.notes)
    assert any("keep it" in note for note in s.notes)


def test_no_tape_values_means_no_suggestion():
    s = scale_suggestion(Suggestion("waist", "WAIST_CALIBRATION", "measurements.py", 1.0), [])
    assert s.suggested is None
    assert s.notes == ["no measured photo with a tape value for it"]


def test_flags_a_suspiciously_large_change():
    pairs = [(37.6, 46.0), (37.0, 45.5), (38.0, 46.4)]  # ~22% off: more likely a tape or photo problem
    s = scale_suggestion(Suggestion("shoulder", "SHOULDER_WIDTH_FACTOR", "measurements.py", 1.15), pairs)
    assert any("suspicious" in note for note in s.notes)


def result(tape: dict, measured: dict, estimated=()):
    return PhotoResult(Sample(photo=None, height_cm=178, tape=tape), measured=measured, estimated=set(estimated))


def test_leaves_out_chest_estimated_from_shoulders_and_compares_length_with_the_shirt():
    measured = {"shoulder_cm": 44.0, "chest_cm": 95.0, "waist_cm": 84.0, "torso_cm": 46.0}
    tape = {"shoulder_cm": 46.0, "chest_cm": 98.0, "shirt_length_cm": 73.0}
    results = [result(tape, measured), result(tape, measured, estimated=["chest_cm"])]
    by_label = {s.label: s for s in suggest_measurements(results)}

    assert by_label["shoulder"].photos == 2
    assert by_label["chest"].photos == 1
    assert any("1 photo(s) left out" in note for note in by_label["chest"].notes)
    assert by_label["waist"].suggested is None  # no waist tape values
    # length: shirt 73 vs ideal 46 × LENGTH_PER_TORSO → LENGTH_PER_TORSO becomes 73 / 46
    assert by_label["length"].suggested == round(73 / 46, 3)
    assert by_label["length"].current == sizing.LENGTH_PER_TORSO


def test_failed_photos_are_ignored():
    ok = result({"shoulder_cm": 46.0}, {"shoulder_cm": 44.0, "chest_cm": 95.0, "waist_cm": 84.0, "torso_cm": 46.0})
    failed = PhotoResult(Sample(photo=None, height_cm=178, tape={"shoulder_cm": 46.0}), error="PARTIAL_BODY: ...")
    shoulder = suggest_measurements([ok, failed])[0]
    assert shoulder.photos == 1


# ---- undertone thresholds ------------------------------------------------------------------------


def test_undertone_thresholds_stay_when_every_label_already_matches():
    assert undertone_thresholds([(35.0, "cool"), (45.0, "neutral"), (60.0, "warm")], 40.0, 52.0) == (40.0, 52.0)


def test_undertone_thresholds_move_just_past_the_mislabelled_photos():
    # A warm photo at hue 50 (currently "neutral") and a cool one at 42 (currently "neutral")
    assert undertone_thresholds([(50.0, "warm"), (42.0, "cool")], 40.0, 52.0) == (42.5, 49.5)
    # A neutral photo at 38 (currently "cool")
    assert undertone_thresholds([(38.0, "neutral")], 40.0, 52.0) == (38.0, 52.0)


def test_undertone_labels_that_contradict_each_other_are_reported():
    message = undertone_thresholds([(55.0, "cool"), (50.0, "warm")], 40.0, 52.0)
    assert isinstance(message, str)
    assert "overlap" in message


# ---- measuring a photo (fake detector, no model files) --------------------------------------------


def test_measures_a_photo_like_analyze_does_and_writes_a_debug_image(tmp_path, jpeg_bytes):
    photo = tmp_path / "me.jpg"
    photo.write_bytes(jpeg_bytes)
    sample = Sample(photo, 175, {"shoulder_cm": 45.0})

    r = measure_photo(sample, FakeDetector([make_pose()]), FakeFaceFinder(make_face()), debug_dir=tmp_path / "debug")

    assert r.error is None
    assert set(r.measured) == {"shoulder_cm", "chest_cm", "waist_cm", "torso_cm"}
    expected = measurements.measure(make_pose(), 175).measurements
    assert r.measured["shoulder_cm"] == expected.shoulder_cm
    assert (tmp_path / "debug" / "me-debug.jpg").read_bytes()[:2] == b"\xff\xd8"  # a JPEG


def test_a_photo_that_fails_is_reported_not_fatal(tmp_path, jpeg_bytes):
    photo = tmp_path / "empty.jpg"
    photo.write_bytes(jpeg_bytes)
    no_person = measure_photo(Sample(photo, 175, {}), FakeDetector([]), FakeFaceFinder(None))
    assert no_person.error.startswith("NO_PERSON_DETECTED")

    missing = measure_photo(Sample(tmp_path / "nope.jpg", 175, {}), FakeDetector([]), FakeFaceFinder(None))
    assert missing.error.startswith("file not found")

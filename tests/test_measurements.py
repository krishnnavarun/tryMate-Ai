"""Measurement maths on the synthetic paper doll (tests/fakes.py): every number is known."""

import math

import pytest

from app.services import measurements as M
from tests.fakes import BASE_POINTS, BODY_PX, TORSO_WIDTH_PX, make_pose

HEIGHT_CM = 180.0
CM_PER_PX = HEIGHT_CM / BODY_PX  # 0.2 cm per pixel


def _dist(a, b):
    return math.dist(BASE_POINTS[a], BASE_POINTS[b])


def test_scale_comes_from_head_top_to_soles():
    result = M.measure(make_pose(), HEIGHT_CM)
    assert result.cm_per_px == pytest.approx(CM_PER_PX)


def test_all_measurements_on_a_clean_pose():
    result = M.measure(make_pose(), HEIGHT_CM)
    m = result.measurements

    assert m.shoulder_cm == pytest.approx(180 * CM_PER_PX * M.SHOULDER_WIDTH_FACTOR, abs=0.06)
    assert m.torso_cm == pytest.approx(300 * CM_PER_PX, abs=0.06)
    # Chest and waist: silhouette width × the ANSUR II circumference ratio
    assert m.chest_cm == pytest.approx(M.CHEST_CIRC_PER_BREADTH * TORSO_WIDTH_PX * CM_PER_PX, abs=0.06)
    assert m.waist_cm == pytest.approx(M.WAIST_CIRC_PER_BREADTH * TORSO_WIDTH_PX * CM_PER_PX, abs=0.06)
    # Limbs: joint-to-joint path, averaged over both sides (symmetric doll)
    assert m.arm_cm == pytest.approx((_dist(11, 13) + _dist(13, 15)) * CM_PER_PX, abs=0.06)
    assert m.leg_cm == pytest.approx((_dist(23, 25) + _dist(25, 27)) * CM_PER_PX, abs=0.06)

    assert result.warnings == []
    assert result.confidence == pytest.approx(0.99)


def test_measurements_scale_with_height():
    short = M.measure(make_pose(), 150).measurements
    tall = M.measure(make_pose(), 200).measurements
    assert tall.chest_cm / short.chest_cm == pytest.approx(200 / 150, rel=0.01)
    assert tall.leg_cm / short.leg_cm == pytest.approx(200 / 150, rel=0.01)


def test_arms_touching_body_falls_back_to_shoulder_ratio():
    from tests.fakes import ARMS_TOUCHING

    result = M.measure(make_pose(overrides=ARMS_TOUCHING), HEIGHT_CM)
    m = result.measurements

    assert M.WARN_ARMS_CHEST in result.warnings
    assert M.WARN_ARMS_WAIST in result.warnings
    assert m.chest_cm == pytest.approx(M.CHEST_CIRC_PER_BREADTH * M.CHEST_BREADTH_PER_SHOULDER * m.shoulder_cm, abs=0.2)
    assert m.waist_cm == pytest.approx(M.WAIST_CIRC_PER_BREADTH * M.WAIST_BREADTH_PER_SHOULDER * m.shoulder_cm, abs=0.2)
    assert result.confidence < 0.99 * 0.85


def test_blurred_gap_between_arm_and_torso_still_finds_the_edge():
    # Fill the gap between torso and right arm with 0.7: what MediaPipe's low-res mask does
    pose = make_pose()
    y_chest = int(230 + M.CHEST_LEVEL * 300)
    row = pose.mask[y_chest - 3 : y_chest + 4]
    gap = row[:, 370:480] < 0.5
    row[:, 370:480][gap] = 0.7

    result = M.measure(pose, HEIGHT_CM)
    assert M.WARN_ARMS_CHEST not in result.warnings
    # The edge lands inside the (former) gap, so the width is a little over the true 140 px
    measured_px = result.measurements.chest_cm / M.CHEST_CIRC_PER_BREADTH / CM_PER_PX
    assert TORSO_WIDTH_PX <= measured_px <= TORSO_WIDTH_PX + 20


def test_without_mask_uses_landmark_fallbacks():
    result = M.measure(make_pose(with_mask=False), HEIGHT_CM)
    assert M.WARN_HEAD in result.warnings
    assert M.WARN_FEET in result.warnings
    assert M.WARN_ARMS_CHEST in result.warnings
    # Fallback head top: eyes (115) − 0.62 × (eyes → shoulders = 115 px); soles: lowest foot landmark (945)
    expected_body_px = 945 - (115 - M.HEAD_TOP_ABOVE_EYES * 115)
    assert result.cm_per_px == pytest.approx(HEIGHT_CM / expected_body_px)


def test_hidden_arms_use_height_ratio():
    hidden = {i: 0.1 for i in (13, 14, 15, 16)}
    result = M.measure(make_pose(visibility=hidden), HEIGHT_CM)
    assert result.measurements.arm_cm == pytest.approx(M.ARM_PER_HEIGHT * HEIGHT_CM, abs=0.06)
    assert M.WARN_ARM_LENGTH in result.warnings


def test_one_visible_arm_is_enough():
    result = M.measure(make_pose(visibility={14: 0.1}), HEIGHT_CM)
    assert result.measurements.arm_cm == pytest.approx((_dist(11, 13) + _dist(13, 15)) * CM_PER_PX, abs=0.06)
    assert M.WARN_ARM_LENGTH not in result.warnings


def test_turned_body_warns():
    # Shoulders only 60 px apart for a 300 px torso: body turned sideways
    result = M.measure(make_pose(overrides={11: (330, 230), 12: (270, 230)}), HEIGHT_CM)
    assert M.WARN_TURNED in result.warnings


def test_debug_lines_are_produced():
    labels = [line.label for line in M.measure(make_pose(), HEIGHT_CM).lines]
    assert any(label.startswith("chest") for label in labels)
    assert any(label.startswith("waist") for label in labels)
    assert any(label.startswith("shoulder") for label in labels)

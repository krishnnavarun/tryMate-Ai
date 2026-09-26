"""Person checks (no person / several people / partial body) and the real MediaPipe model."""

import numpy as np
import pytest

from app.config import MODELS_DIR
from app.errors import AppError, ErrorCode
from app.services.pose import L, PoseDetector, select_single_full_body
from tests.fakes import make_pose


def _code(poses):
    with pytest.raises(AppError) as info:
        select_single_full_body(poses)
    return info.value.code


def test_no_detection_is_no_person():
    assert _code([]) == ErrorCode.NO_PERSON_DETECTED


def test_one_full_body_is_accepted():
    pose = make_pose()
    assert select_single_full_body([pose]) is pose


def test_second_person_is_multiple_people():
    assert _code([make_pose(), make_pose(shift_x=250)]) == ErrorCode.MULTIPLE_PEOPLE


def test_same_person_detected_twice_is_not_multiple_people():
    assert select_single_full_body([make_pose(), make_pose(shift_x=5)])


def test_faint_second_detection_is_ignored():
    faint = make_pose(shift_x=250, visibility={i: 0.2 for i in (11, 12, 23, 24)})
    assert select_single_full_body([make_pose(), faint])


@pytest.mark.parametrize("hidden", [L.LEFT_ANKLE, L.RIGHT_SHOULDER, L.NOSE, L.LEFT_HIP])
def test_hidden_key_landmark_is_partial_body(hidden):
    assert _code([make_pose(visibility={hidden: 0.2})]) == ErrorCode.PARTIAL_BODY


def test_landmark_outside_the_photo_is_partial_body():
    # Feet cut off: MediaPipe still guesses the ankle, but below the bottom edge
    assert _code([make_pose(overrides={L.LEFT_ANKLE: (330, 1100)})]) == ErrorCode.PARTIAL_BODY


def test_partial_body_message_names_the_missing_parts():
    with pytest.raises(AppError) as info:
        select_single_full_body([make_pose(visibility={L.LEFT_ANKLE: 0.1})])
    assert "left ankle" in info.value.message


# ---- the real model -------------------------------------------------------------------

model_missing = not (MODELS_DIR / "pose_landmarker_heavy.task").exists()


@pytest.mark.skipif(model_missing, reason="run scripts/download_models.py first")
def test_real_model_finds_nobody_in_an_empty_image():
    detector = PoseDetector("heavy")
    blank = np.full((800, 600, 3), 200, dtype=np.uint8)
    assert detector.detect(blank) == []


def test_missing_model_file_gives_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.pose.MODELS_DIR", tmp_path)
    with pytest.raises(AppError) as info:
        PoseDetector("heavy")
    assert "download_models.py" in info.value.message


# ---- MediaPipe mask-width workaround ---------------------------------------------------


@pytest.mark.parametrize("width", [600, 601, 602, 603])
def test_images_are_padded_to_a_width_multiple_of_4(width):
    from app.services.pose import pad_width_to_multiple_of_4

    image = np.zeros((100, width, 3), np.uint8)
    padded = pad_width_to_multiple_of_4(image)
    assert padded.shape[1] % 4 == 0
    assert 0 <= padded.shape[1] - width <= 3
    assert padded.shape[0] == 100


def test_detector_always_passes_a_width_multiple_of_4(monkeypatch):
    """Guards against a MediaPipe crash (see pad_width_to_multiple_of_4)."""
    seen = {}

    class SpyLandmarker:
        def detect(self, mp_image):
            seen["width"] = mp_image.width

            class Empty:
                pose_landmarks, segmentation_masks = [], []

            return Empty()

    detector = PoseDetector.__new__(PoseDetector)  # skip loading the real model
    detector._landmarker = SpyLandmarker()
    import threading

    detector._lock = threading.Lock()
    detector.detect(np.zeros((500, 301, 3), np.uint8))
    assert seen["width"] % 4 == 0

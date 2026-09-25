import base64

import cv2
import numpy as np
import pytest

from app.config import MAX_UPLOAD_BYTES
from app.schemas import AnalyzeResponse
from app.services import measurements as M
from tests.conftest import AUTH
from tests.fakes import ARMS_TOUCHING, make_pose


def _analyze(client, image_bytes, filename="photo.jpg", content_type="image/jpeg", **form):
    data = {"height_cm": "180", **form}
    return client.post("/analyze", headers=AUTH, data=data, files={"image": (filename, image_bytes, content_type)})


@pytest.mark.parametrize("fixture", ["jpeg_bytes", "png_bytes", "webp_bytes"])
def test_response_matches_contract_for_each_image_type(client, request, fixture):
    response = _analyze(client, request.getfixturevalue(fixture))
    assert response.status_code == 200, response.text
    body = AnalyzeResponse.model_validate(response.json())
    assert body.debug_image_base64 is None
    assert 0 <= body.confidence <= 1


def test_measurements_come_from_the_pose(client, jpeg_bytes):
    body = _analyze(client, jpeg_bytes).json()
    expected = M.measure(make_pose(), 180).measurements.model_dump()
    assert body["measurements"] == expected


def test_warnings_are_passed_through(client, fake_detector, jpeg_bytes):
    fake_detector.poses = [make_pose(overrides=ARMS_TOUCHING)]
    body = _analyze(client, jpeg_bytes).json()
    assert M.WARN_ARMS_CHEST in body["warnings"]


def test_debug_image_is_a_jpeg(client, jpeg_bytes):
    body = _analyze(client, jpeg_bytes, debug="true").json()
    raw = base64.b64decode(body["debug_image_base64"])
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape[:2] == (1000, 600)


def test_optional_weight_accepted(client, jpeg_bytes):
    assert _analyze(client, jpeg_bytes, weight_kg="72.5").status_code == 200


@pytest.mark.parametrize(
    ("poses", "code"),
    [
        ([], "NO_PERSON_DETECTED"),
        ([make_pose(), make_pose(shift_x=250)], "MULTIPLE_PEOPLE"),
        ([make_pose(visibility={27: 0.1})], "PARTIAL_BODY"),
    ],
)
def test_person_errors(client, fake_detector, jpeg_bytes, poses, code):
    fake_detector.poses = poses
    response = _analyze(client, jpeg_bytes)
    assert response.status_code == 422
    assert response.json()["error_code"] == code


# ---- input validation ---------------------------------------------------------------


@pytest.mark.parametrize("height", ["119", "231", "abc", "999"])
def test_height_out_of_range_is_invalid(client, jpeg_bytes, height):
    response = _analyze(client, jpeg_bytes, height_cm=height)
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"
    assert "height_cm" in response.json()["message"]


def test_missing_image_is_invalid(client):
    response = client.post("/analyze", headers=AUTH, data={"height_cm": "175"})
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_non_image_file_is_invalid_even_with_image_content_type(client):
    # The content type says JPEG, but the bytes are text: we check the bytes.
    response = _analyze(client, b"definitely not an image", content_type="image/jpeg")
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"
    assert "JPEG, PNG and WEBP" in response.json()["message"]


def test_gif_is_rejected(client):
    response = _analyze(client, b"GIF89a" + b"\x00" * 100, filename="a.gif", content_type="image/gif")
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_empty_file_is_invalid(client):
    response = _analyze(client, b"")
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_tiny_image_is_invalid(client):
    ok, buffer = cv2.imencode(".jpg", np.zeros((100, 80, 3), np.uint8))
    response = _analyze(client, buffer.tobytes())
    assert response.status_code == 422
    assert "too small" in response.json()["message"]


def test_file_over_10mb_is_invalid(client, jpeg_bytes):
    too_big = jpeg_bytes + b"\x00" * (MAX_UPLOAD_BYTES + 1 - len(jpeg_bytes))
    response = _analyze(client, too_big)
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"
    assert "10 MB" in response.json()["message"]

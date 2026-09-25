import pytest

from tests.conftest import AUTH

SIZE_BODY = {
    "measurements": {
        "shoulder_cm": 44.1,
        "chest_cm": 96.5,
        "waist_cm": 84.0,
        "torso_cm": 62.3,
        "arm_cm": 60.2,
        "leg_cm": 81.7,
    },
    "size_chart": {"M": {"chest": [92, 98]}},
    "category": "upper_body",
}


def _call(client, path, headers, jpeg):
    """Send a valid request to each protected endpoint."""
    if path == "/analyze":
        return client.post(path, headers=headers, data={"height_cm": "175"}, files={"image": ("p.jpg", jpeg, "image/jpeg")})
    if path == "/recommend-size":
        return client.post(path, headers=headers, json=SIZE_BODY)
    return client.post(
        path,
        headers=headers,
        data={"category": "upper_body", "garment_image_url": "https://example.com/shirt.jpg"},
        files={"person_image": ("p.jpg", jpeg, "image/jpeg")},
    )


PROTECTED = ["/analyze", "/recommend-size", "/try-on"]


@pytest.mark.parametrize("path", PROTECTED)
def test_missing_key_is_rejected(client, path, jpeg_bytes):
    response = _call(client, path, {}, jpeg_bytes)
    assert response.status_code == 401
    assert response.json()["error_code"] == "UNAUTHORIZED"
    assert response.json()["message"]


@pytest.mark.parametrize("path", PROTECTED)
def test_wrong_key_is_rejected(client, path, jpeg_bytes):
    response = _call(client, path, {"X-API-Key": "wrong"}, jpeg_bytes)
    assert response.status_code == 401
    assert response.json()["error_code"] == "UNAUTHORIZED"


@pytest.mark.parametrize("path", PROTECTED)
def test_correct_key_is_accepted(client, path, jpeg_bytes):
    response = _call(client, path, AUTH, jpeg_bytes)
    assert response.status_code == 200, response.text

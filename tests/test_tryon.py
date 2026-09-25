import pytest

from app.schemas import TryOnResponse
from tests.conftest import AUTH


def _person(jpeg):
    return ("person_image", ("person.jpg", jpeg, "image/jpeg"))


def test_stub_with_garment_url(client, jpeg_bytes):
    response = client.post(
        "/try-on",
        headers=AUTH,
        data={
            "category": "upper_body",
            "garment_image_url": "https://example.com/shirt.jpg",
            "garment_description": "navy blue cotton polo shirt",
        },
        files=[_person(jpeg_bytes)],
    )
    assert response.status_code == 200, response.text
    body = TryOnResponse.model_validate(response.json())
    # Exactly one of the two result fields is set.
    assert (body.result_image_url is None) != (body.result_image_base64 is None)


def test_stub_with_garment_file(client, jpeg_bytes, png_bytes):
    response = client.post(
        "/try-on",
        headers=AUTH,
        data={"category": "dresses"},
        files=[_person(jpeg_bytes), ("garment_image", ("shirt.png", png_bytes, "image/png"))],
    )
    assert response.status_code == 200, response.text
    TryOnResponse.model_validate(response.json())


def test_garment_is_required(client, jpeg_bytes):
    response = client.post("/try-on", headers=AUTH, data={"category": "upper_body"}, files=[_person(jpeg_bytes)])
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_both_garment_sources_is_invalid(client, jpeg_bytes, png_bytes):
    response = client.post(
        "/try-on",
        headers=AUTH,
        data={"category": "upper_body", "garment_image_url": "https://example.com/shirt.jpg"},
        files=[_person(jpeg_bytes), ("garment_image", ("shirt.png", png_bytes, "image/png"))],
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


@pytest.mark.parametrize(
    "data",
    [
        {"category": "shoes", "garment_image_url": "https://example.com/shirt.jpg"},
        {"category": "upper_body", "garment_image_url": "not a url"},
        {"category": "upper_body", "garment_image_url": "ftp://example.com/shirt.jpg"},
    ],
)
def test_invalid_fields(client, jpeg_bytes, data):
    response = client.post("/try-on", headers=AUTH, data=data, files=[_person(jpeg_bytes)])
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_bad_garment_file_type(client, jpeg_bytes):
    response = client.post(
        "/try-on",
        headers=AUTH,
        data={"category": "upper_body"},
        files=[_person(jpeg_bytes), ("garment_image", ("shirt.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 422
    assert "garment_image" in response.json()["message"]

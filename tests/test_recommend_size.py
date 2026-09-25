import copy

import pytest

from app.schemas import RecommendSizeResponse
from tests.conftest import AUTH

VALID = {
    "measurements": {
        "shoulder_cm": 44.1,
        "chest_cm": 96.5,
        "waist_cm": 84.0,
        "torso_cm": 62.3,
        "arm_cm": 60.2,
        "leg_cm": 81.7,
    },
    "size_chart": {
        "S": {"chest": [86, 92], "waist": [74, 80], "length": [68, 70], "shoulder": [41, 43]},
        "M": {"chest": [92, 98], "waist": [80, 86], "length": [70, 72], "shoulder": [43, 45]},
        "L": {"chest": [98, 104], "waist": [86, 92], "length": [72, 74], "shoulder": [45, 47]},
    },
    "category": "upper_body",
    "fit_preference": "regular",
}


def _body(**changes):
    body = copy.deepcopy(VALID)
    body.update(changes)
    return body


def test_stub_matches_contract(client):
    response = client.post("/recommend-size", headers=AUTH, json=VALID)
    assert response.status_code == 200, response.text
    body = RecommendSizeResponse.model_validate(response.json())
    # Every size in the chart gets a score + note, in the same order.
    assert list(body.per_size) == ["S", "M", "L"]
    assert body.recommended_size in body.per_size


def test_fit_preference_defaults_to_regular(client):
    body = _body()
    del body["fit_preference"]
    assert client.post("/recommend-size", headers=AUTH, json=body).status_code == 200


def test_missing_fields_in_a_size_are_allowed(client):
    chart = {"S": {"chest": [86, 92]}, "M": {"waist": [80, 86]}, "L": {}}
    response = client.post("/recommend-size", headers=AUTH, json=_body(size_chart=chart))
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "changes",
    [
        {"size_chart": {}},  # no sizes
        {"size_chart": {"M": {"chest": [98, 92]}}},  # min > max
        {"size_chart": {"M": {"chest": [92]}}},  # not a pair
        {"size_chart": {"M": {"chest": [-1, 92]}}},  # negative
        {"category": "shoes"},
        {"fit_preference": "tight"},
        {"measurements": {"chest_cm": 96.5}},  # incomplete measurements
    ],
)
def test_invalid_bodies_are_rejected(client, changes):
    response = client.post("/recommend-size", headers=AUTH, json=_body(**changes))
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_malformed_json_is_invalid_input(client):
    response = client.post(
        "/recommend-size", headers={**AUTH, "Content-Type": "application/json"}, content=b"{not json"
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"

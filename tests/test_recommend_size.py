"""Size recommendation: hand-checked cases, fit preference, notes, edge cases, validation."""

import copy

import pytest

from app.schemas import RecommendSizeResponse
from app.services import sizing
from tests.conftest import AUTH

CHART = {
    "S": {"chest": [86, 92], "waist": [74, 80], "length": [68, 70], "shoulder": [41, 43]},
    "M": {"chest": [92, 98], "waist": [80, 86], "length": [70, 72], "shoulder": [43, 45]},
    "L": {"chest": [98, 104], "waist": [86, 92], "length": [72, 74], "shoulder": [45, 47]},
    "XL": {"chest": [104, 110], "waist": [92, 98], "length": [74, 76], "shoulder": [47, 49]},
}
# A textbook medium: every measurement in the middle of M (torso 46.5 × 1.55 ≈ 72 cm length)
MEDIUM = {"shoulder_cm": 44.0, "chest_cm": 95.0, "waist_cm": 83.0, "torso_cm": 46.5, "arm_cm": 60.0, "leg_cm": 82.0}
BETWEEN_M_AND_L = {**MEDIUM, "chest_cm": 98.0, "shoulder_cm": 45.0, "waist_cm": 86.0}


def _post(client, measurements=None, chart=None, category="upper_body", fit="regular"):
    body = {
        "measurements": measurements or MEDIUM,
        "size_chart": chart or CHART,
        "category": category,
        "fit_preference": fit,
    }
    return client.post("/recommend-size", headers=AUTH, json=body)


def _result(client, **kwargs) -> RecommendSizeResponse:
    response = _post(client, **kwargs)
    assert response.status_code == 200, response.text
    return RecommendSizeResponse.model_validate(response.json())


# ---- hand-checked cases ----------------------------------------------------------------


def test_textbook_medium_gets_m_with_good_fit(client):
    result = _result(client)
    assert result.recommended_size == "M"
    assert result.per_size["M"].score > 0.95
    assert result.per_size["M"].note == "Good fit"
    assert list(result.per_size) == ["S", "M", "L", "XL"]


def test_smaller_size_is_tight_and_larger_is_loose(client):
    result = _result(client)
    assert result.per_size["S"].note.startswith("Tight at chest")
    assert result.per_size["L"].note.startswith("Loose at chest")


def test_scores_fall_off_away_from_the_best_size(client):
    s = _result(client).per_size
    assert s["M"].score > s["L"].score > s["XL"].score
    assert s["M"].score > s["S"].score


@pytest.mark.parametrize(("fit", "expected"), [("slim", "M"), ("regular", "L"), ("loose", "L")])
def test_between_sizes_fit_preference_decides(client, fit, expected):
    # Exactly on the M/L boundary: slim picks the snug M, loose the roomy L.
    # Regular is a tie, and ties go to the larger size.
    assert _result(client, measurements=BETWEEN_M_AND_L, fit=fit).recommended_size == expected


def test_slim_preference_favours_the_smaller_neighbour(client):
    slim, loose = _result(client, fit="slim").per_size, _result(client, fit="loose").per_size
    assert slim["S"].score > loose["S"].score
    assert loose["L"].score > slim["L"].score


def test_broad_shoulders_are_mentioned(client):
    result = _result(client, measurements={**MEDIUM, "shoulder_cm": 47.5})
    assert result.recommended_size == "M"
    assert "tight at shoulders" in result.per_size["M"].note.lower()


# ---- edge cases ------------------------------------------------------------------------


def test_bigger_than_every_size_gets_the_largest(client):
    big = {**MEDIUM, "chest_cm": 125.0, "waist_cm": 115.0, "shoulder_cm": 52.0}
    result = _result(client, measurements=big)
    assert result.recommended_size == "XL"
    assert "Tight" in result.per_size["XL"].note


def test_smaller_than_every_size_gets_the_smallest(client):
    small = {**MEDIUM, "chest_cm": 78.0, "waist_cm": 66.0, "shoulder_cm": 38.0}
    result = _result(client, measurements=small)
    assert result.recommended_size == "S"
    assert "Loose" in result.per_size["S"].note


def test_missing_fields_use_what_is_there(client):
    chart = {"S": {"chest": [86, 92]}, "M": {"chest": [92, 98]}, "L": {"chest": [98, 104]}}
    result = _result(client, chart=chart)
    assert result.recommended_size == "M"
    assert result.per_size["M"].score == pytest.approx(1.0, abs=0.01)


def test_a_size_without_comparable_fields_scores_zero(client):
    chart = {**CHART, "XXL": {"hip": [110, 116]}}
    result = _result(client, chart=chart)
    assert result.per_size["XXL"].score == 0
    assert "No comparable" in result.per_size["XXL"].note


def test_chart_without_any_comparable_field_is_invalid(client):
    response = _post(client, chart={"S": {"hip": [90, 95]}, "M": {}})
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_lower_body_uses_waist_and_inseam(client):
    chart = {"30": {"waist": [74, 79], "inseam": [78, 81]}, "32": {"waist": [79, 84], "inseam": [80, 83]}}
    result = _result(client, chart=chart, category="lower_body")
    assert result.recommended_size == "32"  # waist 83, leg 82


def test_length_notes(client):
    long_torso = {**MEDIUM, "torso_cm": 52.0}  # ideal length ≈ 80.6 cm, far longer than M's 70–72
    note = _result(client, measurements=long_torso).per_size["M"].note
    assert "short" in note.lower()


# ---- note wording ----------------------------------------------------------------------


def test_note_wording():
    f = sizing.FieldResult
    assert sizing.fit_note([f("chest", 0.2, 1), f("shoulder", -0.3, 1)]) == "Good fit"
    assert sizing.fit_note([f("chest", 2.0, 0), f("shoulder", 2.0, 0)]) == "Tight at chest and shoulders"
    assert sizing.fit_note([f("shoulder", -2, 0), f("length", -1.0, 0)]) == "Loose at shoulders, slightly long"
    assert sizing.fit_note([f("waist", 1.0, 0)]) == "Slightly tight at waist"


# ---- validation ------------------------------------------------------------------------


def _body(**changes):
    body = {"measurements": MEDIUM, "size_chart": copy.deepcopy(CHART), "category": "upper_body"}
    body.update(changes)
    return body


def test_fit_preference_defaults_to_regular(client):
    assert client.post("/recommend-size", headers=AUTH, json=_body()).status_code == 200


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


# ---- fit breakdown, between sizes, sleeves ----------------------------------------------


def _field(result, size, name):
    return next(f for f in result.per_size[size].fields if f.field == name)


def test_every_size_explains_its_fit_in_numbers(client):
    result = _result(client)
    chest = _field(result, "M", "chest")
    assert (chest.body_cm, chest.size_min, chest.size_max, chest.ideal_cm) == (95.0, 92, 98, 95.0)
    assert chest.difference_cm == 0.0 and chest.verdict == "good"
    # S is made for 86–92 (ideal 89): 6 cm too small for this chest
    assert (_field(result, "S", "chest").difference_cm, _field(result, "S", "chest").verdict) == (6.0, "tight")
    assert _field(result, "XL", "chest").verdict == "loose"
    length = _field(result, "M", "length")
    assert length.body_cm == round(46.5 * sizing.LENGTH_PER_TORSO, 1) and length.label == "length"


def test_verdict_words():
    assert sizing.verdict("chest", 0.5) == "good"
    assert sizing.verdict("chest", 1.0) == "slightly_tight"
    assert sizing.verdict("waist", -2.0) == "loose"
    assert sizing.verdict("length", 1.0) == "slightly_short"
    assert sizing.verdict("sleeve", -1.6) == "long"


def test_between_sizes_names_the_neighbour(client):
    result = _result(client, measurements=BETWEEN_M_AND_L)
    assert {result.recommended_size, result.alternative_size} == {"M", "L"}
    assert "between M and L" in result.alternative_note


def test_a_clear_size_has_no_alternative(client):
    result = _result(client)
    assert result.alternative_size is None and result.alternative_note is None


def test_alternative_note_says_which_size_fits_closer():
    sizes = ["S", "M", "L"]
    alt, note = sizing.between_sizes(sizes, {"S": 0.1, "M": 0.9, "L": 0.8}, "M")
    assert alt == "L"
    assert note == "You're between M and L: M is the closer match; L fits more relaxed."
    alt, note = sizing.between_sizes(sizes, {"S": 0.1, "M": 0.8, "L": 0.9}, "L")
    assert alt == "M" and note.endswith("M fits closer.")
    assert sizing.between_sizes(sizes, {"S": 0.1, "M": 0.9, "L": 0.5}, "M") == (None, None)


SLEEVES = ([61, 62.5], [62.5, 64], [64, 65.5], [65.5, 67])
SLEEVE_CHART = {size: {**ranges, "sleeve": sleeve} for (size, ranges), sleeve in zip(CHART.items(), SLEEVES)}


def test_long_sleeves_are_compared_with_arm_length(client):
    result = _result(client, chart=SLEEVE_CHART)
    sleeve = _field(result, "M", "sleeve")
    assert sleeve.body_cm == round(60.0 * sizing.SLEEVE_PER_ARM, 1)
    assert (sleeve.verdict, sleeve.label) == ("good", "sleeves")
    assert result.recommended_size == "M"


def test_short_arms_hear_about_long_sleeves(client):
    short_arms = {**MEDIUM, "arm_cm": 55.0}  # ideal sleeve ≈ 58 cm; M sleeves are 62.5–64
    result = _result(client, measurements=short_arms, chart=SLEEVE_CHART)
    assert "long in the sleeves" in result.per_size["M"].note.lower()
    assert _field(result, "M", "sleeve").verdict == "long"


def test_sleeves_only_count_for_tops():
    from app.schemas import Measurements

    assert sizing.body_value("sleeve", Measurements(**MEDIUM), "dresses") is None
    assert sizing.body_value("sleeve", Measurements(**MEDIUM), "upper_body") == 60.0 * sizing.SLEEVE_PER_ARM

"""Skin tone, undertone, colour palettes, and the face-not-found behaviour of /analyze."""

import numpy as np
import pytest

from app.config import MODELS_DIR
from app.services import skin_tone as S
from app.services.colors import _PALETTES, suggest_colors
from app.services.face import FaceFinder
from tests.conftest import AUTH
from tests.fakes import make_face, make_pose


def _image(color_rgb, size=(1000, 600)) -> np.ndarray:
    return np.full((*size, 3), color_rgb, dtype=np.uint8)


# ---- ITA tone bands (Del Bino et al. 2006) ---------------------------------------------


@pytest.mark.parametrize(
    ("ita", "tone"),
    [(60, "fair"), (50, "light"), (35, "medium"), (20, "tan"), (0, "brown"), (-45, "deep")],
)
def test_tone_bands(ita, tone):
    assert S.classify_tone(ita) == tone


def test_ita_formula():
    # L* = 70, b* = 20 → atan(20 / 20) = 45°
    assert S.ita_angle(70, 20) == pytest.approx(45.0)
    # b* = 0 must not crash (atan2)
    assert S.ita_angle(60, 0) == pytest.approx(90.0)


@pytest.mark.parametrize(("hue", "undertone"), [(30, "cool"), (45, "neutral"), (60, "warm")])
def test_undertone_from_hue(hue, undertone):
    assert S.classify_undertone(hue) == undertone


# ---- sampling and cleaning -------------------------------------------------------------


def test_skin_colour_is_measured_from_the_face():
    skin = (224, 172, 138)  # a light, warm skin colour
    result = S.analyze_skin(_image(skin), make_face())
    r, g, b = (int(result.skin_tone.hex[i : i + 2], 16) for i in (1, 3, 5))
    assert abs(r - skin[0]) <= 3 and abs(g - skin[1]) <= 3 and abs(b - skin[2]) <= 3
    assert result.skin_tone.tone in {"light", "medium"}
    assert len(result.sample_circles) == 3


def test_deep_skin_is_classified_deep():
    result = S.analyze_skin(_image((92, 58, 40)), make_face())
    assert result.skin_tone.tone in {"brown", "deep"}


def test_non_skin_pixels_are_ignored():
    image = _image((224, 172, 138))
    # Paint half of every sample circle bright blue (e.g. a background showing through)
    image[:, :300] = (40, 90, 230)
    result = S.analyze_skin(image, make_face())
    assert result.skin_tone.hex[1:3] != "28"  # not blue-dominated
    r, _, b = (int(result.skin_tone.hex[i : i + 2], 16) for i in (1, 3, 5))
    assert r > b


def test_no_skin_like_pixels_falls_back_with_warning():
    result = S.analyze_skin(_image((40, 90, 230)), make_face())
    assert S.WARN_FEW_PIXELS in result.warnings


def test_small_face_warns():
    face = make_face()
    face.width_px = 20
    assert S.WARN_SMALL_FACE in S.analyze_skin(_image((224, 172, 138)), face).warnings


def test_white_balance_brightens_a_dim_photo():
    image = _image((150, 110, 90))
    image[:100, :] = (190, 190, 190)  # a dim "white" wall at the top
    gains = S.white_balance_gains(image)
    assert gains == pytest.approx(np.full(3, S.WB_TARGET / 190), rel=0.02)


def test_white_balance_skips_photos_without_grey_areas():
    assert S.white_balance_gains(_image((224, 172, 138))) == pytest.approx(np.ones(3))


def test_blown_out_white_is_not_used_as_reference():
    image = _image((150, 110, 90))
    image[:100, :] = (255, 255, 255)
    assert S.white_balance_gains(image) == pytest.approx(np.ones(3))


def test_color_cast_warning():
    orange_light = _image((230, 150, 60))
    assert S.WARN_COLOR_CAST in S.lighting_warnings(orange_light)


def test_dark_photo_warning():
    assert S.WARN_DARK_PHOTO in S.lighting_warnings(_image((30, 25, 20)))


# ---- colour palettes -------------------------------------------------------------------


@pytest.mark.parametrize("undertone", ["warm", "cool", "neutral"])
@pytest.mark.parametrize("tone", ["fair", "light", "medium", "tan", "brown", "deep"])
def test_every_combination_has_eight_colours(tone, undertone):
    colors = suggest_colors(tone, undertone)
    assert len(colors) == 8
    assert len({c.name for c in colors}) == 8


def test_warm_and_cool_palettes_differ():
    warm = {c.name for c in suggest_colors("medium", "warm")}
    cool = {c.name for c in suggest_colors("medium", "cool")}
    assert "Mustard" in warm and "Mustard" not in cool


def test_all_palettes_are_valid_hex():
    for palette in _PALETTES.values():
        for _, hex_code in palette:
            assert len(hex_code) == 7 and hex_code.startswith("#")
            int(hex_code[1:], 16)


# ---- /analyze with and without a face --------------------------------------------------


def _analyze(client, jpeg):
    return client.post(
        "/analyze", headers=AUTH, data={"height_cm": "180"}, files={"image": ("p.jpg", jpeg, "image/jpeg")}
    )


def test_analyze_returns_skin_tone_and_colours(client, jpeg_bytes):
    body = _analyze(client, jpeg_bytes).json()
    assert body["skin_tone"]["undertone"] in {"warm", "cool", "neutral"}
    assert len(body["color_suggestions"]) == 8


def test_no_face_still_returns_measurements(client, fake_face_finder, jpeg_bytes):
    fake_face_finder.face = None
    response = _analyze(client, jpeg_bytes)
    assert response.status_code == 200
    body = response.json()
    assert body["skin_tone"] is None
    assert body["color_suggestions"] == []
    assert body["measurements"]["chest_cm"] > 0
    assert any("face" in w for w in body["warnings"])


# ---- the real face model ---------------------------------------------------------------


@pytest.mark.skipif(not (MODELS_DIR / "face_landmarker.task").exists(), reason="run scripts/download_models.py")
def test_real_face_model_finds_no_face_in_a_plain_image():
    assert FaceFinder().find(_image((200, 200, 200)), make_pose()) is None

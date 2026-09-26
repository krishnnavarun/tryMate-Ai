"""Shared test fixtures.

`client` is a FastAPI TestClient whose settings are fixed in code, so tests never
depend on whatever is in your local .env file. Its pose detector is a fake that returns
the synthetic paper-doll person from tests/fakes.py, and its face finder a fake face on
the doll's head. Change what they return with the `fake_detector` / `fake_face_finder`
fixtures (e.g. `fake_detector.poses = []`, `fake_face_finder.face = None`).
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services.face import get_face_finder
from app.services.pose import get_pose_detector
from tests.fakes import FakeDetector, FakeFaceFinder, make_face, make_pose

TEST_API_KEY = "test-key"
AUTH = {"X-API-Key": TEST_API_KEY}
# Mock try-on (never call Replicate from tests) and no try-on rate limit (tested separately)
TEST_SETTINGS = Settings(_env_file=None, service_api_key=TEST_API_KEY, tryon_mock=True, tryon_rate_limit_per_minute=0)


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector([make_pose()])


@pytest.fixture
def fake_face_finder() -> FakeFaceFinder:
    return FakeFaceFinder(make_face())


@pytest.fixture
def client(fake_detector, fake_face_finder):
    app.dependency_overrides[get_settings] = lambda: TEST_SETTINGS
    app.dependency_overrides[get_pose_detector] = lambda: fake_detector
    app.dependency_overrides[get_face_finder] = lambda: fake_face_finder
    # raise_server_exceptions=False: let unexpected errors come back as a 500 response
    # (like in production) instead of crashing the test.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _encode(ext: str) -> bytes:
    """A real 600 × 1000 image generated in memory (no files on disk)."""
    pixels = np.full((1000, 600, 3), (90, 140, 200), dtype=np.uint8)
    ok, buffer = cv2.imencode(ext, pixels)
    assert ok
    return buffer.tobytes()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return _encode(".jpg")


@pytest.fixture
def png_bytes() -> bytes:
    return _encode(".png")


@pytest.fixture
def webp_bytes() -> bytes:
    return _encode(".webp")

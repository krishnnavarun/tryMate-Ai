from fastapi.testclient import TestClient

from app import __version__
from app.config import get_settings
from app.main import app
from app.services.face import get_face_finder
from app.services.pose import get_pose_detector
from tests.conftest import TEST_SETTINGS


def test_health_needs_no_api_key(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_version_is_contract_version():
    assert __version__ == "0.1.0"


def test_health_works_even_if_the_models_cannot_load(caplog):
    # What happened on Linux without libgles2: loading MediaPipe raised an OSError at
    # startup, the app never started, and even /health was unreachable.
    calls = []

    def broken_loader():
        calls.append(1)
        raise OSError("libGLESv2.so.2: cannot open shared object file: No such file or directory")

    app.dependency_overrides[get_settings] = lambda: TEST_SETTINGS
    app.dependency_overrides[get_pose_detector] = broken_loader
    app.dependency_overrides[get_face_finder] = broken_loader
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
    finally:
        app.dependency_overrides.clear()

    assert len(calls) == 2, "startup uses the overrides (tests never load the real models)"
    assert "Could not load a model at startup (get_pose_detector)" in caplog.text
    assert "libGLESv2.so.2" in caplog.text

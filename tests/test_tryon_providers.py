"""Try-on providers: mock, Replicate IDM-VTON (with a fake client), provider selection, rate limit."""

import asyncio
import base64

import cv2
import httpx
import numpy as np
import pytest
from pydantic import ValidationError
from replicate.exceptions import ReplicateError

from app.config import Settings
from app.errors import AppError, ErrorCode
from app.main import app
from app.rate_limit import SlidingWindowLimiter, tryon_limiter
from app.config import get_settings
from app.tryon import get_tryon_provider
from app.tryon import replicate_idm as R
from app.tryon.base import TryOnRequest
from app.tryon.mock import MockTryOnProvider
from tests.conftest import AUTH, TEST_API_KEY


def _jpeg(width=300, height=400) -> bytes:
    return cv2.imencode(".jpg", np.full((height, width, 3), 180, np.uint8))[1].tobytes()


def _request(**kw) -> TryOnRequest:
    return TryOnRequest(person_image=_jpeg(), category="upper_body", garment_image_url="https://x.test/shirt.jpg", **kw)


# ---- mock ------------------------------------------------------------------------------


def test_mock_returns_a_decodable_image():
    result = asyncio.run(MockTryOnProvider().run(_request()))
    assert result.image_url is None
    image = cv2.imdecode(np.frombuffer(base64.b64decode(result.image_base64), np.uint8), cv2.IMREAD_COLOR)
    assert image is not None


def test_mock_pastes_an_uploaded_garment():
    request = TryOnRequest(person_image=_jpeg(), category="upper_body", garment_image=_jpeg(100, 120))
    assert asyncio.run(MockTryOnProvider().run(request)).image_base64


# ---- Replicate, with a fake client ------------------------------------------------------


class FakePrediction:
    def __init__(self, statuses, output="https://replicate.delivery/out.jpg", error=None):
        self._statuses = list(statuses)
        self.status = self._statuses.pop(0)
        self.output, self.error, self.id = output, error, "pred-1"
        self.cancelled = False

    async def async_reload(self):
        if self._statuses:
            self.status = self._statuses.pop(0)

    async def async_cancel(self):
        self.cancelled = True


class FakePredictions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # each: a FakePrediction or an exception to raise
        self.inputs = []

    async def async_create(self, version, input):  # noqa: A002 — Replicate's parameter name
        self.inputs.append(input)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _provider(outcomes) -> R.ReplicateIdmVtonProvider:
    provider = R.ReplicateIdmVtonProvider(api_token="r8_test", version="v1")
    provider._client = type("FakeClient", (), {"predictions": FakePredictions(outcomes)})()
    return provider


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setattr(R, "POLL_INTERVAL_SECONDS", 0.001)


def test_success_returns_the_output_url():
    provider = _provider([FakePrediction(["starting", "processing", "succeeded"])])
    result = asyncio.run(provider.run(_request(garment_description="navy polo")))
    assert result.image_url == "https://replicate.delivery/out.jpg"
    sent = provider._client.predictions.inputs[0]
    assert sent["garm_img"] == "https://x.test/shirt.jpg"  # URLs are passed straight through
    assert sent["garment_des"] == "navy polo"
    assert sent["category"] == "upper_body"


def test_output_list_is_accepted():
    provider = _provider([FakePrediction(["succeeded"], output=["https://r.test/a.jpg"])])
    assert asyncio.run(provider.run(_request())).image_url == "https://r.test/a.jpg"


def test_model_failure_is_tryon_failed_without_retry():
    provider = _provider([FakePrediction(["processing", "failed"], error="CUDA out of memory")])
    with pytest.raises(AppError) as info:
        asyncio.run(provider.run(_request()))
    assert info.value.code == ErrorCode.TRYON_FAILED
    assert len(provider._client.predictions.inputs) == 1


def test_timeout_cancels_the_prediction(monkeypatch):
    monkeypatch.setattr(R, "TRYON_TIMEOUT_SECONDS", 0.05)
    prediction = FakePrediction(["processing"] * 10_000)
    with pytest.raises(AppError) as info:
        asyncio.run(_provider([prediction]).run(_request()))
    assert info.value.code == ErrorCode.TRYON_TIMEOUT
    assert prediction.cancelled


def test_transient_error_is_retried_once():
    provider = _provider([httpx.ConnectError("boom"), FakePrediction(["succeeded"])])
    assert asyncio.run(provider.run(_request())).image_url
    assert len(provider._client.predictions.inputs) == 2


def test_two_transient_errors_give_up():
    provider = _provider([ReplicateError(status=503), ReplicateError(status=502)])
    with pytest.raises(AppError) as info:
        asyncio.run(provider.run(_request()))
    assert info.value.code == ErrorCode.TRYON_FAILED


def test_client_errors_are_not_retried():
    provider = _provider([ReplicateError(status=422, detail="bad input"), FakePrediction(["succeeded"])])
    with pytest.raises(AppError):
        asyncio.run(provider.run(_request()))
    assert len(provider._client.predictions.inputs) == 1


def test_missing_token_is_a_clear_error():
    with pytest.raises(AppError) as info:
        R.ReplicateIdmVtonProvider(api_token="")
    assert "REPLICATE_API_TOKEN" in info.value.message


# ---- provider selection ------------------------------------------------------------------


def test_mock_flag_wins():
    settings = Settings(_env_file=None, tryon_provider="replicate_idm", tryon_mock=True)
    assert get_tryon_provider(settings).name == "mock"


def test_replicate_is_selected_with_a_token():
    settings = Settings(_env_file=None, tryon_provider="replicate_idm", replicate_api_token="r8_x")
    assert get_tryon_provider(settings).name == "replicate_idm"


def test_unknown_provider_is_rejected_at_startup():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tryon_provider="catvton")


# ---- rate limit --------------------------------------------------------------------------


def test_sliding_window_limiter():
    limiter = SlidingWindowLimiter(window_seconds=60)
    assert [limiter.allow(2) for _ in range(3)] == [True, True, False]


def test_tryon_rate_limit_returns_429(client):
    limited = Settings(_env_file=None, service_api_key=TEST_API_KEY, tryon_mock=True, tryon_rate_limit_per_minute=2)
    app.dependency_overrides[get_settings] = lambda: limited
    tryon_limiter.reset()
    try:
        codes = [
            client.post(
                "/try-on",
                headers=AUTH,
                data={"category": "upper_body", "garment_image_url": "https://x.test/s.jpg"},
                files={"person_image": ("p.jpg", _jpeg(), "image/jpeg")},
            ).status_code
            for _ in range(3)
        ]
    finally:
        tryon_limiter.reset()
    assert codes == [200, 200, 429]

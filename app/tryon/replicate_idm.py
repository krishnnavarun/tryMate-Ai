"""IDM-VTON on Replicate (https://replicate.com/cuuupid/idm-vton).

- Inputs: human_img, garm_img (file or URL), garment_des, category, crop, steps, seed
- Output: one image URL (hosted by Replicate, expires after about an hour)
- Typical run: ~17 s on an A100 (up to a minute+ on a cold start), ≈ $0.023 per run
- LICENSE: CC BY-NC-SA 4.0 — NON-COMMERCIAL use only. Fine for a demo/portfolio; a
  commercial store needs a commercially licensed model (e.g. a hosted API such as FASHN,
  or CatVTON self-hosted after checking its licence). Swap it in as another TryOnProvider.

How a run works here:
1. Look up the model's latest version once (or use REPLICATE_IDM_VERSION to pin one).
2. Create a prediction, then poll it every second.
3. After TRYON_TIMEOUT_SECONDS: cancel it (so we stop paying) → TRYON_TIMEOUT.
   Model error → TRYON_FAILED. Network hiccup / Replicate 5xx / 429 → retry ONCE.
"""

import asyncio
import io
import logging
import time

import httpx
from replicate.client import Client
from replicate.exceptions import ReplicateError

from app.errors import AppError, ErrorCode
from app.tryon.base import TryOnProvider, TryOnRequest, TryOnResult

logger = logging.getLogger(__name__)

MODEL = "cuuupid/idm-vton"
TRYON_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 1.0
MAX_ATTEMPTS = 2  # the first try + one retry on transient errors
DIFFUSION_STEPS = 30
SEED = 42  # fixed, so the same inputs give the same image


class TransientError(Exception):
    """A failure worth retrying once (network error, Replicate 5xx or 429)."""


class ReplicateIdmVtonProvider(TryOnProvider):
    name = "replicate_idm"

    def __init__(self, api_token: str, version: str | None = None) -> None:
        if not api_token:
            raise AppError(ErrorCode.TRYON_FAILED, "Try-on is not configured (REPLICATE_API_TOKEN is empty).")
        self._client = Client(api_token=api_token, timeout=httpx.Timeout(30.0))
        self._version = version or None

    async def run(self, request: TryOnRequest) -> TryOnResult:
        deadline = time.monotonic() + TRYON_TIMEOUT_SECONDS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self._run_once(request, deadline)
            except TransientError as err:
                logger.warning("Try-on attempt %d failed with a transient error: %s", attempt, err)
                if attempt == MAX_ATTEMPTS or time.monotonic() >= deadline:
                    raise AppError(ErrorCode.TRYON_FAILED) from err
        raise AppError(ErrorCode.TRYON_FAILED)  # not reached

    async def _latest_version(self) -> str:
        if self._version is None:
            try:
                model = await self._client.models.async_get(MODEL)
            except (httpx.TransportError, ReplicateError) as err:
                raise _classify(err) from err
            if model.latest_version is None:
                raise AppError(ErrorCode.TRYON_FAILED, f"{MODEL} has no published version.")
            self._version = model.latest_version.id
        return self._version

    def _inputs(self, request: TryOnRequest) -> dict:
        # File-like objects are uploaded by the Replicate client; a URL string is fetched by Replicate
        garment = request.garment_image_url or io.BytesIO(request.garment_image or b"")
        return {
            "human_img": io.BytesIO(request.person_image),
            "garm_img": garment,
            "garment_des": request.garment_description or _default_description(request.category),
            "category": request.category,
            "crop": True,  # lets the model handle photos that aren't 3:4
            "steps": DIFFUSION_STEPS,
            "seed": SEED,
        }

    async def _run_once(self, request: TryOnRequest, deadline: float) -> TryOnResult:
        version = await self._latest_version()
        try:
            prediction = await self._client.predictions.async_create(version=version, input=self._inputs(request))
            while prediction.status not in ("succeeded", "failed", "canceled"):
                if time.monotonic() >= deadline:
                    await _cancel_quietly(prediction)
                    raise AppError(ErrorCode.TRYON_TIMEOUT)
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                await prediction.async_reload()
        except (httpx.TransportError, ReplicateError) as err:
            raise _classify(err) from err

        if prediction.status != "succeeded":
            # The model itself failed (bad input, out of memory...): retrying rarely helps
            logger.warning("Try-on prediction %s ended as %s: %s", prediction.id, prediction.status, prediction.error)
            raise AppError(ErrorCode.TRYON_FAILED)

        url = _output_url(prediction.output)
        if not url:
            raise AppError(ErrorCode.TRYON_FAILED, "The try-on provider returned no image.")
        return TryOnResult(image_url=url)


def _default_description(category: str) -> str:
    return {"upper_body": "a shirt", "lower_body": "trousers", "dresses": "a dress"}.get(category, "a garment")


def _output_url(output) -> str | None:
    """The model returns one URL; accept a list or a FileOutput-like object too."""
    if isinstance(output, list):
        output = output[0] if output else None
    if output is None:
        return None
    return str(getattr(output, "url", output))


def _classify(err: Exception) -> Exception:
    """Network errors, 5xx and 429 are transient (retry once); everything else is a failure."""
    if isinstance(err, httpx.TransportError):
        return TransientError(str(err))
    status = getattr(err, "status", None)
    if status is not None and (status >= 500 or status == 429):
        return TransientError(f"HTTP {status}")
    logger.warning("Try-on provider error: %s", err)
    return AppError(ErrorCode.TRYON_FAILED)


async def _cancel_quietly(prediction) -> None:
    try:
        await prediction.async_cancel()
    except Exception:  # noqa: BLE001 — best effort, we're already reporting a timeout
        logger.warning("Could not cancel timed-out prediction %s", getattr(prediction, "id", "?"))

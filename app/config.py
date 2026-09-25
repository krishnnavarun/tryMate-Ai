"""App settings, loaded from environment variables / the .env file.

pydantic-settings matches env vars to fields case-insensitively,
so SERVICE_API_KEY in .env fills `service_api_key` below.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---- Fixed limits (part of the contract, so not configurable via .env) ----

MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# /try-on can carry two images, plus a little room for the text fields and
# multipart boundaries. Anything bigger is rejected before it is parsed.
MAX_REQUEST_BYTES = 2 * MAX_UPLOAD_BYTES + 1024 * 1024

ALLOWED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")

# Decoded photos are shrunk so the longest side is at most this many pixels.
# Pose models work on ~256 px inputs anyway; this keeps memory and time low
# while leaving enough detail for measuring silhouette widths.
MAX_IMAGE_SIDE = 1280
# Smaller photos don't have enough detail to measure a body.
MIN_IMAGE_SIDE = 256

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_api_key: str = "change-me"
    replicate_api_token: str = ""
    tryon_provider: Literal["replicate_idm", "catvton", "mock"] = "replicate_idm"
    tryon_mock: bool = False
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    port: int = 8000
    # MediaPipe pose model: heavy is the most accurate (we analyse one photo, so its
    # extra ~0.5 s doesn't matter); full/lite use less CPU and memory.
    pose_model: Literal["lite", "full", "heavy"] = "heavy"

    @field_validator("log_level", mode="before")
    @classmethod
    def _lowercase_log_level(cls, value: str) -> str:
        # Accept LOG_LEVEL=INFO as well as LOG_LEVEL=info
        return value.lower() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    """Read settings once and reuse them. Used as a FastAPI dependency so tests can override it."""
    return Settings()

"""Virtual try-on providers.

base.py defines the TryOnProvider interface; replicate_idm.py (IDM-VTON on Replicate) and
mock.py (free local placeholder) implement it. get_tryon_provider() picks one from settings:

    TRYON_MOCK=true           → mock (whatever TRYON_PROVIDER says)
    TRYON_PROVIDER=mock       → mock
    TRYON_PROVIDER=replicate_idm → Replicate IDM-VTON (needs REPLICATE_API_TOKEN)
    TRYON_PROVIDER=catvton    → not implemented yet (TRYON_FAILED with a clear message)
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.errors import AppError, ErrorCode
from app.tryon.base import TryOnProvider
from app.tryon.mock import MockTryOnProvider
from app.tryon.replicate_idm import ReplicateIdmVtonProvider


@lru_cache
def _replicate_provider(api_token: str, version: str) -> ReplicateIdmVtonProvider:
    # Cached so the model's version is looked up only once per process
    return ReplicateIdmVtonProvider(api_token=api_token, version=version or None)


def get_tryon_provider(settings: Annotated[Settings, Depends(get_settings)]) -> TryOnProvider:
    if settings.tryon_mock or settings.tryon_provider == "mock":
        return MockTryOnProvider()
    if settings.tryon_provider == "replicate_idm":
        return _replicate_provider(settings.replicate_api_token, settings.replicate_idm_version)
    raise AppError(ErrorCode.TRYON_FAILED, f"Try-on provider '{settings.tryon_provider}' is not implemented yet.")

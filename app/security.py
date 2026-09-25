"""X-API-Key check. Every endpoint except /health depends on `require_api_key`."""

import secrets
from typing import Annotated

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings
from app.errors import AppError, ErrorCode

# auto_error=False: we raise our own AppError so the response uses the contract's error shape.
# Declaring it with APIKeyHeader also adds an "Authorize" button to /docs.
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Shared secret between the Express server and this service (SERVICE_API_KEY).",
)


def require_api_key(
    api_key: Annotated[str | None, Security(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    # compare_digest takes the same time whether the first or last character differs,
    # so an attacker can't guess the key one character at a time from response timings.
    if not api_key or not secrets.compare_digest(api_key.encode(), settings.service_api_key.encode()):
        raise AppError(ErrorCode.UNAUTHORIZED)

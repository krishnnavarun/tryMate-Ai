"""HTTP routes. Each module defines an APIRouter that app/main.py includes."""

from typing import Any

from app.schemas import ErrorResponse

# Documents our error shape in /docs for the status codes every protected endpoint can return.
# (Replaces FastAPI's default 422 schema, since our handler reshapes validation errors.)
COMMON_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "UNAUTHORIZED: missing or wrong X-API-Key"},
    422: {"model": ErrorResponse, "description": "INVALID_INPUT or an image-analysis error"},
    500: {"model": ErrorResponse, "description": "INTERNAL_ERROR"},
}

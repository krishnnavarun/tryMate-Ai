"""Error codes, the AppError exception, and handlers that turn every error into

    { "error_code": "...", "message": "..." }

so the Express server always gets the same shape back (see PROJECT_SPEC.md §5).
"""

import logging
from enum import StrEnum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    UNAUTHORIZED = "UNAUTHORIZED"
    INVALID_INPUT = "INVALID_INPUT"
    NO_PERSON_DETECTED = "NO_PERSON_DETECTED"
    MULTIPLE_PEOPLE = "MULTIPLE_PEOPLE"
    PARTIAL_BODY = "PARTIAL_BODY"
    FACE_NOT_FOUND = "FACE_NOT_FOUND"
    TRYON_FAILED = "TRYON_FAILED"
    TRYON_TIMEOUT = "TRYON_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# HTTP status for each code (from the error table in the spec)
STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.INVALID_INPUT: 422,
    ErrorCode.NO_PERSON_DETECTED: 422,
    ErrorCode.MULTIPLE_PEOPLE: 422,
    ErrorCode.PARTIAL_BODY: 422,
    ErrorCode.FACE_NOT_FOUND: 422,
    ErrorCode.TRYON_FAILED: 502,
    ErrorCode.TRYON_TIMEOUT: 504,
    ErrorCode.INTERNAL_ERROR: 500,
}

DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.UNAUTHORIZED: "Missing or invalid X-API-Key header.",
    ErrorCode.INVALID_INPUT: "The request is invalid.",
    ErrorCode.NO_PERSON_DETECTED: "No person found in the photo. Use a clear full-body photo.",
    ErrorCode.MULTIPLE_PEOPLE: "More than one person found in the photo. Use a photo with only one person.",
    ErrorCode.PARTIAL_BODY: "The full body (head to feet) must be visible in the photo.",
    ErrorCode.FACE_NOT_FOUND: "No face found in the photo.",
    ErrorCode.TRYON_FAILED: "The try-on provider returned an error.",
    ErrorCode.TRYON_TIMEOUT: "The try-on provider took too long to respond.",
    ErrorCode.INTERNAL_ERROR: "Something went wrong on our side.",
}


class AppError(Exception):
    """Raise this anywhere in the app to return a contract-shaped error.

    Example:
        raise AppError(ErrorCode.PARTIAL_BODY)
        raise AppError(ErrorCode.INVALID_INPUT, "height_cm must be between 120 and 230")
    """

    def __init__(self, code: ErrorCode, message: str | None = None) -> None:
        self.code = code
        self.message = message or DEFAULT_MESSAGES[code]
        self.status_code = STATUS_BY_CODE[code]
        super().__init__(self.message)


def error_response(code: ErrorCode, message: str, status_code: int | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code or STATUS_BY_CODE[code],
        content={"error_code": code.value, "message": message},
    )


def _format_validation_errors(exc: RequestValidationError) -> str:
    """Turn Pydantic's error list into one readable line.

    e.g. "height_cm: Input should be greater than or equal to 120; image: Field required"
    """
    parts = []
    for err in exc.errors():
        # loc looks like ("body", "height_cm") or ("body", "size_chart", "S", "chest").
        # Drop the "body"/"query"/"header" prefix, it only adds noise.
        loc = [str(p) for p in err.get("loc", ()) if p not in ("body", "query", "header", "path")]
        field = ".".join(loc) or "request"
        parts.append(f"{field}: {err.get('msg', 'invalid value')}")
    return "; ".join(parts) or DEFAULT_MESSAGES[ErrorCode.INVALID_INPUT]


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return error_response(exc.code, exc.message, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(ErrorCode.INVALID_INPUT, _format_validation_errors(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Errors raised by FastAPI/Starlette itself (404 unknown route, 405, body parse errors...)
        if exc.status_code == 401:
            return error_response(ErrorCode.UNAUTHORIZED, str(exc.detail))
        if exc.status_code >= 500:
            return error_response(ErrorCode.INTERNAL_ERROR, DEFAULT_MESSAGES[ErrorCode.INTERNAL_ERROR])
        # 400 (e.g. malformed body) and 413 become the contract's 422 INVALID_INPUT;
        # other 4xx (404, 405) keep their status so they still make sense.
        status = 422 if exc.status_code in (400, 413) else exc.status_code
        return error_response(ErrorCode.INVALID_INPUT, str(exc.detail), status)

    @app.exception_handler(Exception)
    async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        # Log the traceback (never request bodies / image bytes), return a generic message.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return error_response(ErrorCode.INTERNAL_ERROR, DEFAULT_MESSAGES[ErrorCode.INTERNAL_ERROR])

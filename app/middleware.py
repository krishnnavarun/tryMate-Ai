"""ASGI middleware."""

import json

from fastapi import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import ErrorCode


class BodySizeLimitMiddleware:
    """Reject request bodies larger than `max_bytes` with 422 INVALID_INPUT.

    Two checks:
    1. If the client sends a Content-Length header that is too big, reply right away
       without reading the body at all.
    2. If there is no Content-Length (chunked upload), count bytes as they arrive and stop
       once the limit is passed. We raise FastAPI's HTTPException(413) from inside
       `receive`; FastAPI re-raises HTTPExceptions that happen while it reads the body,
       and our error handler (app/errors.py) turns 413 into 422 INVALID_INPUT.

    This keeps memory use bounded, which matters because uploads are held in RAM
    (we never spill photos to disk).
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.message = f"Request body is larger than {max_bytes // (1024 * 1024)} MB."

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = dict(scope["headers"]).get(b"content-length")
        if content_length is not None and content_length.isdigit() and int(content_length) > self.max_bytes:
            await self._reject(send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(status_code=413, detail=self.message)
            return message

        await self.app(scope, limited_receive, send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps({"error_code": ErrorCode.INVALID_INPUT.value, "message": self.message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 422,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})

"""A small in-memory rate limiter for /try-on (each run costs money on Replicate).

Only the store's server calls this service, and the store already limits try-ons per user,
so this is a safety net: a global cap on try-ons per minute (TRYON_RATE_LIMIT_PER_MINUTE).

"Sliding window": we remember the time of each recent request and count how many fall in the
last 60 s. Memory is per process — with several server processes each has its own count.
"""

import threading
import time
from collections import deque
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.errors import AppError, ErrorCode


class SlidingWindowLimiter:
    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window = window_seconds
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self, limit: int) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._hits and now - self._hits[0] >= self.window:
                self._hits.popleft()
            if len(self._hits) >= limit:
                return False
            self._hits.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


tryon_limiter = SlidingWindowLimiter()


def limit_tryon(settings: Annotated[Settings, Depends(get_settings)]) -> None:
    if settings.tryon_rate_limit_per_minute > 0 and not tryon_limiter.allow(settings.tryon_rate_limit_per_minute):
        raise AppError(ErrorCode.RATE_LIMITED)

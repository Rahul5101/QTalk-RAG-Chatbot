"""
Rate Limiter Module for FastAPI and standalone execution.
Implements in-memory sliding window rate limiting with configurable window and request limits.
"""

import time
import os
from collections import defaultdict
from typing import Tuple, Dict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class SlidingWindowRateLimiter:
    def __init__(self, requests_per_minute: int = 60, window_seconds: int = 60):
        self.requests_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", requests_per_minute))
        self.window_seconds = window_seconds
        self.history: Dict[str, list] = defaultdict(list)

    def _clean_old_requests(self, client_id: str, current_time: float):
        cutoff = current_time - self.window_seconds
        self.history[client_id] = [t for t in self.history[client_id] if t > cutoff]

    def is_allowed(self, client_id: str) -> Tuple[bool, int, float]:
        """
        Check if request is allowed.
        Returns: (is_allowed, remaining_requests, reset_time_seconds)
        """
        now = time.time()
        self._clean_old_requests(client_id, now)

        recent_requests = self.history[client_id]
        if len(recent_requests) >= self.requests_per_minute:
            oldest_request = recent_requests[0]
            retry_after = round(self.window_seconds - (now - oldest_request), 2)
            return False, 0, max(0.1, retry_after)

        self.history[client_id].append(now)
        remaining = self.requests_per_minute - len(self.history[client_id])
        return True, remaining, 0.0

rate_limiter = SlidingWindowRateLimiter()

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Exclude health check and OpenAPI docs from rate limiting
        path = request.url.path
        if path in ["/health", "/docs", "/openapi.json", "/metrics"]:
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        session_id = request.headers.get("x-session-id", client_ip)
        client_key = f"{client_ip}:{session_id}"

        allowed, remaining, retry_after = rate_limiter.is_allowed(client_key)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded. Please try again later.",
                    "retry_after_seconds": retry_after
                },
                headers={"Retry-After": str(int(retry_after))}
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

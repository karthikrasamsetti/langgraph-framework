"""
Authentication and rate limiting for the LangGraph API.

Auth    — X-API-Key header validation
Rate    — sliding window counter per API key
Pattern — FastAPI Depends() injection on protected endpoints
"""
import time
import threading
from collections import defaultdict, deque
from typing import Optional

from fastapi import Header, HTTPException, Request
from fastapi.security import APIKeyHeader

from config.settings import get_settings, parse_api_keys
from observability.logger import log

settings   = get_settings()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── Rate limit store ───────────────────────────────────────────────────────────
# In-memory sliding window per API key.
# Each key maps to a deque of request timestamps.
# Thread-safe via lock — FastAPI uses a thread pool.
_rate_store: dict[str, deque] = defaultdict(deque)
_rate_lock  = threading.Lock()


def _is_rate_limited(api_key: str, rpm: int) -> tuple[bool, int]:
    """
    Sliding window rate limiter.
    Returns (is_limited, seconds_until_reset).
    """
    if not settings.RATE_LIMIT_ENABLED:
        return False, 0

    now    = time.time()
    window = settings.RATE_LIMIT_WINDOW_SECONDS

    with _rate_lock:
        timestamps = _rate_store[api_key]

        # Remove timestamps outside the current window
        while timestamps and timestamps[0] < now - window:
            timestamps.popleft()

        count = len(timestamps)

        if count >= rpm:
            # How long until the oldest request falls outside the window
            oldest        = timestamps[0]
            reset_in      = int((oldest + window) - now) + 1
            return True, reset_in

        # Record this request
        timestamps.append(now)
        return False, 0


def _get_remaining(api_key: str, rpm: int) -> int:
    """Returns how many requests are remaining in the current window."""
    now    = time.time()
    window = settings.RATE_LIMIT_WINDOW_SECONDS

    with _rate_lock:
        timestamps = _rate_store[api_key]
        recent     = sum(1 for t in timestamps if t >= now - window)
        return max(0, rpm - recent)


# ── FastAPI dependency ─────────────────────────────────────────────────────────
async def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    """
    FastAPI dependency — validates API key and checks rate limit.
    Inject with: Depends(require_api_key)

    Returns the key metadata dict: {"name": str, "rpm": int}
    Raises 401 if key missing or invalid.
    Raises 429 if rate limit exceeded.
    Adds rate limit headers to every response.
    """
    # ── Auth disabled — skip all checks ───────────────────────────────────────
    if not settings.AUTH_ENABLED:
        return {"name": "anonymous", "rpm": 9999}

    # ── Key missing ───────────────────────────────────────────────────────────
    if not x_api_key:
        log.warning("auth_missing_key",
                    path=request.url.path,
                    client=request.client.host if request.client else "unknown")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Add X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # ── Key invalid ───────────────────────────────────────────────────────────
    valid_keys = parse_api_keys()
    if x_api_key not in valid_keys:
        log.warning("auth_invalid_key",
                    path=request.url.path,
                    key_prefix=x_api_key[:8]+"..." if len(x_api_key) > 8 else "short")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_meta = valid_keys[x_api_key]
    rpm      = key_meta["rpm"]

    # ── Rate limit check ──────────────────────────────────────────────────────
    is_limited, reset_in = _is_rate_limited(x_api_key, rpm)
    remaining            = _get_remaining(x_api_key, rpm)

    # Add rate limit headers to response
    request.state.rate_limit_headers = {
        "X-RateLimit-Limit":     str(rpm),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset":     str(reset_in),
        "X-RateLimit-Window":    str(settings.RATE_LIMIT_WINDOW_SECONDS),
    }

    if is_limited:
        log.warning("rate_limit_exceeded",
                    key_name=key_meta["name"],
                    rpm=rpm,
                    reset_in=reset_in)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. {rpm} requests per "
                   f"{settings.RATE_LIMIT_WINDOW_SECONDS} seconds. "
                   f"Try again in {reset_in} seconds.",
            headers={
                "Retry-After":           str(reset_in),
                "X-RateLimit-Limit":     str(rpm),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset":     str(reset_in),
            },
        )

    log.info("auth_ok",
             key_name=key_meta["name"],
             path=request.url.path,
             remaining=remaining)

    return key_meta


# ── Optional dependency — admin only ──────────────────────────────────────────
async def require_admin_key(
    key_meta: dict = Header(default=None),
) -> dict:
    """
    Stricter dependency for admin endpoints.
    Only keys with name='admin' are allowed.
    """
    if not settings.AUTH_ENABLED:
        return {"name": "admin", "rpm": 9999}

    if key_meta.get("name") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required.",
        )
    return key_meta
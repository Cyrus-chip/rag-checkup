"""Small, bounded retry helper shared by HTTP connectors."""
from __future__ import annotations

import time
from collections.abc import Callable

import httpx


# These statuses are normally safe to retry.  In particular, 401/403 and the
# rest of the 4xx range are returned immediately so a bad credential cannot
# turn one query into a burst of rejected requests.
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int = 2,
    retry_backoff: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
) -> httpx.Response:
    """Perform one request and retry only bounded transient failures.

    ``max_retries`` counts retries after the first attempt.  A final response
    is returned to the caller even when it is retryable so its normal
    ``raise_for_status`` path and exception type remain visible to the caller.
    Transport errors are re-raised after the final attempt.
    """
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if retry_backoff < 0:
        raise ValueError("retry_backoff must be non-negative")

    send = getattr(client, method.lower())
    for attempt in range(max_retries + 1):
        try:
            response = send(url, **kwargs)
        except TRANSIENT_ERRORS:
            if attempt >= max_retries:
                raise
            if retry_backoff:
                sleep(retry_backoff * (2**attempt))
            continue
        # Lightweight fake clients used by offline callers may only implement
        # ``raise_for_status``.  Real httpx responses always have a status.
        status_code = getattr(response, "status_code", None)
        if status_code not in RETRYABLE_STATUS_CODES or attempt >= max_retries:
            return response
        if retry_backoff:
            sleep(retry_backoff * (2**attempt))
    raise AssertionError("request retry loop did not return or raise")

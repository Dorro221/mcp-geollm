"""Telling apart the ways an upstream provider can say no.

The routing service and the geocoder are the same host with the same key and
three separate daily quotas, and when one of them runs out the answer is a 403
whose body says `Quota exceeded`. Collapsed into "the service could not be
reached", that reads as a network fault: the natural response is to retry, which
consumes nothing but time, and to keep retrying for the rest of the day.

The distinction matters to two different readers. A model is told the same thing
either way — the computation did not happen, do not substitute an estimate —
but the researcher watching the log needs to know whether to restart the sweep
in a minute or tomorrow.
"""
from __future__ import annotations

import httpx

# ORS answers an exhausted quota with 403 and a body naming it; other services
# use 429. Both are read, because the body is the only thing that separates an
# exhausted quota from a key that was never authorised.
_QUOTA_MARKERS = ("quota", "rate limit", "rate-limit", "too many requests")


def quota_exhausted(exc: BaseException) -> bool:
    """True when the provider refused because the allowance is spent."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    status = exc.response.status_code
    if status == 429:
        return True
    if status != 403:
        return False
    try:
        body = exc.response.text.casefold()
    except Exception:
        return False
    return any(marker in body for marker in _QUOTA_MARKERS)


def describe(exc: BaseException) -> str:
    """A short phrase naming what went wrong, for the log and for the message.

    Deliberately free of advice: what to do about it differs between the
    geocoder and the router, and belongs to their own failure messages.
    """
    if quota_exhausted(exc):
        return "the daily quota for this endpoint is exhausted"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return f"the API key was refused ({status})"
        return f"the service answered {status}"
    if isinstance(exc, httpx.TimeoutException):
        return "the service did not answer in time"
    return "the service could not be reached"

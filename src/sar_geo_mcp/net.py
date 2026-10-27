"""One retry policy, shared by every public raster service this package reads.

WHY
---
`fetch_dem` fans out over up to sixteen tiles and `write_evt_clip` over as
many again. A single transient 5xx therefore discards a whole staging call,
and that is not hypothetical: a 502 from 3DEP killed a five-area run
mid-flight, and repeating the identical request a few seconds later
succeeded for all five. The failure was the service, not the request.

WHAT IS AND IS NOT RETRIED
--------------------------
Retried: 429, the 5xx family, and transport errors (timeouts, connection
resets). These say "ask again".

Not retried: every other 4xx. A 400 from a malformed bbox or a 404 from a
renamed service can never succeed, and retrying it spends a minute of the
operator's time to arrive at the same answer with the cause buried under
five identical failures.

The budget is small on purpose. This runs in front of an interactive tool
call, so the right behaviour on a service that is genuinely down is to fail
in seconds with the real status, not to hold the call open for minutes.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

#: Response statuses worth asking again about. Declared as data so the policy
#: is readable without tracing the retry loop.
RETRY_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504, 507, 509})

#: Transport-level failures worth asking again about: timeouts, connection
#: resets, DNS blips. `httpx.TransportError` is the common base for all of
#: them, and it deliberately does NOT cover `HTTPStatusError`.
RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (httpx.TransportError,)

DEFAULT_ATTEMPTS = 5
DEFAULT_BASE_DELAY_S = 2.0
DEFAULT_TIMEOUT_S = 180.0


def get_with_retry(
    url: str,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_S,
    timeout: float = DEFAULT_TIMEOUT_S,
    get: Callable[..., httpx.Response] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """GET `url`, retrying only what is worth retrying, and return the response.

    Raises the LAST failure once the budget is spent, so the caller sees the
    real status rather than a wrapper hiding it. `get` and `sleep` are
    injected for tests; production passes neither.

    Delay doubles per attempt from `base_delay`, which puts a five-attempt
    budget at 2 + 4 + 8 + 16 seconds of waiting in the worst case.
    """
    fetch = get if get is not None else httpx.get
    delay = base_delay
    last: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = fetch(url, timeout=timeout, follow_redirects=True)
        except RETRY_EXCEPTIONS as error:
            last = error
        else:
            if response.status_code not in RETRY_STATUS:
                # Includes the success path AND every non-retryable error:
                # raise_for_status turns a 400 into an exception here, on the
                # first attempt, which is the point.
                response.raise_for_status()
                return response
            last = httpx.HTTPStatusError(
                f"{response.status_code} from {url}",
                request=response.request, response=response)

        if attempt < attempts:
            sleep(delay)
            delay *= 2

    assert last is not None    # unreachable: attempts >= 1 always sets it
    raise last


def bytes_with_retry(url: str, **kwargs) -> bytes:
    """`get_with_retry` for the callers that only want the body.

    This is the shape `fetch_dem` and `write_evt_clip` inject as `get_bytes`.
    """
    return get_with_retry(url, **kwargs).content

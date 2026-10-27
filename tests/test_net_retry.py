"""Transient upstream failures are retried; permanent ones are not.

WHY THIS EXISTS
---------------
A 502 from 3DEP aborted a five-area staging run mid-flight: `fetch_dem` fans
out over up to 16 tiles and one bad response discarded the whole call. A
caller-side retry loop made all five succeed unchanged, so the fault was
transient. The policy has to discriminate, though - a 400 from a malformed
bbox must still fail immediately and loudly, because retrying it just wastes
the operator's time on a request that can never succeed.
"""

import httpx
import pytest

from sar_geo_mcp.net import RETRY_STATUS, get_with_retry


def response(status: int, url: str = "https://example.invalid/x") -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", url), content=b"body")


class Recorder:
    """A `get` that replays a scripted sequence and counts its calls."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, url, **kwargs):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_transient_status_is_retried_until_it_succeeds():
    get = Recorder(response(502), response(503), response(200))
    result = get_with_retry("https://example.invalid/x", get=get, sleep=lambda _: None)
    assert result.status_code == 200
    assert get.calls == 3


def test_rate_limit_is_retried():
    get = Recorder(response(429), response(200))
    assert get_with_retry("u", get=get, sleep=lambda _: None).status_code == 200
    assert get.calls == 2


def test_transport_error_is_retried():
    get = Recorder(httpx.ConnectTimeout("slow"), response(200))
    assert get_with_retry("u", get=get, sleep=lambda _: None).status_code == 200
    assert get.calls == 2


def test_client_error_is_not_retried():
    """A 400 is the caller's fault. Fail on the first response, not the sixth."""
    get = Recorder(response(400))
    with pytest.raises(httpx.HTTPStatusError):
        get_with_retry("u", get=get, sleep=lambda _: None)
    assert get.calls == 1


def test_404_is_not_retried():
    get = Recorder(response(404))
    with pytest.raises(httpx.HTTPStatusError):
        get_with_retry("u", get=get, sleep=lambda _: None)
    assert get.calls == 1


def test_gives_up_after_the_attempt_budget_and_raises_the_last_error():
    get = Recorder(response(502))
    with pytest.raises(httpx.HTTPStatusError) as caught:
        get_with_retry("u", get=get, sleep=lambda _: None, attempts=4)
    assert caught.value.response.status_code == 502
    assert get.calls == 4


def test_backoff_grows_and_is_reported_to_the_sleeper():
    waits = []
    get = Recorder(response(502), response(502), response(200))
    get_with_retry("u", get=get, sleep=waits.append, base_delay=2.0)
    assert waits == [2.0, 4.0]


def test_every_retryable_status_is_declared_not_inferred():
    """The set is data, so a reader can see the policy without tracing code."""
    assert 429 in RETRY_STATUS
    assert {500, 502, 503, 504} <= RETRY_STATUS
    assert 400 not in RETRY_STATUS and 404 not in RETRY_STATUS

"""A panel or chip fetch failed: say why, without saying where it was asked.

`fetch_panel`'s own `RuntimeError`/`ValueError` and httpx's `HTTPStatusError`
carry the full CalTopo GetMap URL in their message - and that URL's path
holds the account token, the same way a Mapbox or Google static-image URL
carries its key in the query string. A tool that raises or warns with that
message verbatim publishes the credential wherever the warning is logged,
displayed or returned to an agent. This is the one place that turns such an
error into something safe to surface, shared by every module that fetches a
panel or a chip rather than reimplemented per caller.
"""

from __future__ import annotations


def safe_reason(error: BaseException) -> str:
    """Why a fetch failed, fit for a warning or a refusal: the exception's
    type, and for an HTTP status error its status code ("HTTPStatusError
    401"). Never the message - see the module docstring for why."""
    import httpx

    if isinstance(error, httpx.HTTPStatusError):
        return f"{type(error).__name__} {error.response.status_code}"
    return type(error).__name__

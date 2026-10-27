"""No request URL - and so no token or key in one - reaches a log."""

import logging


def test_importing_the_hub_quiets_the_http_client_loggers():
    # On this machine both loggers already inherit WARNING from the root
    # logger, so the plain assertion below would pass even without the fix
    # in sar_geo_mcp/__init__.py. Forcing INFO on each here, before the
    # (re)import, makes the test exercise that fix for both rather than
    # the environment's own default.
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)

    import sar_geo_mcp  # noqa: F401
    import importlib

    importlib.reload(sar_geo_mcp)

    for name in ("httpx", "httpcore"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING, name

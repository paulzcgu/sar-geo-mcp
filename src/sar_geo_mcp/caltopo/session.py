"""CalTopo transport: opening a session, reading a public map, and finding
the feature list in whatever envelope a reply came in.

Nothing here knows what an assignment is. Finding one by name is parsing,
and lives with the model in `assignment.find_assignment`; until 0.27.1 it
lived here, which made this module (infra) import domain
(AGENTS.md). There is deliberately no re-export: that would be
the same upward import under another name.
"""

from pathlib import Path

from caltopo_python import CaltopoSession

from ..config import CaltopoCredentials


def extract_features(map_data: dict) -> list[dict]:
    """Find the FeatureCollection wherever CalTopo nested it.

    Observed shapes: {'state': {'features': [...]}} from an authenticated
    session, {'result': {'state': {'features': [...]}}} from the public
    since/0 endpoint.
    """
    if isinstance(map_data, dict):
        features = map_data.get("features")
        if isinstance(features, list):
            return features
        for value in map_data.values():
            found = extract_features(value)
            if found:
                return found
    return []


def fetch_public_map(map_id: str, host: str = "caltopo.com",
                    since: int = 0) -> dict:
    """Read a publicly-shared map without credentials.

    `since` is CalTopo's change token (`result.timestamp`, milliseconds
    since epoch). Zero asks for the whole map; a previously-returned token
    asks only for what changed after it, which on an unchanged map is 84
    bytes against 2.75 MB. See `state.py` for why that reply is used to
    DETECT change rather than to merge it.

    Raises httpx.HTTPError on failure, including when a private map returns
    an error for an unauthenticated caller.
    """
    import httpx

    url = f"https://{host}/api/v1/map/{map_id}/since/{int(since)}"
    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def sole_account(config_path: Path) -> str | None:
    """The only account section in a cts.ini, if there is exactly one.

    `caltopo_python` matches the section name against a configured account
    and refuses an empty one, so a perfectly good credentials file fails
    just because nobody set the environment variable to the section header
    they had already written. When the file describes one account there is
    nothing to choose between, so choosing it is not a guess.

    Two or more sections IS a choice, and this returns None rather than
    picking - the wrong CalTopo account is the kind of mistake that reads
    somebody else's incident.
    """
    import configparser

    parser = configparser.ConfigParser()
    try:
        parser.read(config_path)
    except (configparser.Error, OSError):
        return None
    sections = parser.sections()
    return sections[0] if len(sections) == 1 else None


def open_map(map_id: str, settings: CaltopoCredentials, *,
             sync: bool = False, sync_interval: int = 10,
             sync_timeout: int = 10,
             **session_kwargs) -> CaltopoSession:
    """Open a CalTopo map. Raises with actionable text when credentials are absent.

    `sync=False` (the default) is a one-shot read: the session does its
    single full fetch and starts no background polling. Note that even a
    one-shot session leaves the library's `requestThread` behind for the
    life of the process, so a caller that reads the same map repeatedly
    must hold ONE session rather than open one per read - `reader.MapReader`
    does exactly that and is what the server uses.

    `sync=True` is for a caller that will HOLD the session for the life of
    its process and watch one map all day: the library's own loop then
    polls `/since/<token>` every `sync_interval` seconds (84 bytes when
    nothing changed), merges what moved, handles deletions, and reconnects
    by itself after a failed poll. Such a caller owns the session's
    lifetime and must call `closeMap()` when it is done.

    `sync_timeout` is the per-request timeout in seconds, including the
    initial full fetch. `session_kwargs` go straight to the library - the
    callbacks a watcher needs (`disconnectedCallback`, `mapClosedCallback`,
    the feature-change callbacks).

    A credential given directly (`caltopo_id` and `caltopo_key`, as the
    Claude Desktop extension passes them) is used in place of the file.
    Read with `getattr`: the protocol only promises the three file
    attributes, and a caller's own settings object may not carry these.
    """
    cred_id = getattr(settings, "caltopo_id", "") or ""
    cred_key = getattr(settings, "caltopo_key", "") or ""
    if cred_id and cred_key:
        # Same sync=False reasoning as below.
        return CaltopoSession(
            settings.caltopo_host,
            map_id,
            id=cred_id,
            key=cred_key,
            accountId=getattr(settings, "caltopo_account_id", "") or None,
            sync=sync,
            syncInterval=sync_interval,
            syncTimeout=sync_timeout,
            **session_kwargs,
        )

    if not settings.caltopo_config_path.exists():
        raise FileNotFoundError(
            f"CalTopo credentials not found at {settings.caltopo_config_path}. "
            "Either set the CalTopo credential ID and key in the extension's "
            "settings (ZA_CALTOPO_ID and ZA_CALTOPO_KEY), or copy "
            "cts.ini.example to cts.ini and fill in id, key, and accountId, "
            "then point caltopo_account at the matching section name."
        )

    account = settings.caltopo_account or sole_account(settings.caltopo_config_path)
    if not account:
        import configparser

        parser = configparser.ConfigParser()
        parser.read(settings.caltopo_config_path)
        available = ", ".join(parser.sections()) or "(none)"
        raise ValueError(
            f"which CalTopo account? {settings.caltopo_config_path} defines: "
            f"{available}. Set ZA_CALTOPO_ACCOUNT to one of them."
        )

    # The default sync=False is load-bearing. The library's own default
    # (True) starts a daemon thread that polls `/since/` every ten seconds
    # for the life of the process, and nothing anyone holding this session
    # can call stops it - `__del__` never runs because the thread owns a
    # reference to the session. A one-shot reader that dropped such a
    # session leaked one poller for good: a debriefer server re-reading
    # its map once a minute all day was polling CalTopo forty times a
    # second by evening, and drawing 502s for it (2026-09-06). `openMap`
    # still does its one initial `/since/0` fetch with sync off, which is
    # the whole read a one-shot caller wants.
    return CaltopoSession(
        settings.caltopo_host,
        map_id,
        configpath=str(settings.caltopo_config_path),
        account=account,
        sync=sync,
        syncInterval=sync_interval,
        syncTimeout=sync_timeout,
        **session_kwargs,
    )

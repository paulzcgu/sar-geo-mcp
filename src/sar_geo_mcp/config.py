"""Credentials this package needs, as a plain structural contract.

The one dependency that had to be broken when this core was lifted out of
the SAR debrief app: `caltopo/session.py` imported that app's concrete
`Settings` class, which would have dragged the whole application behind any
consumer of this package - the opposite of the point.

`CaltopoCredentials` is what `open_map` actually uses, and nothing more.
Deliberately a protocol-shaped duck type rather than a required base class:
the debrief app's own `Settings` object already carries these three
attributes with these exact names, so it keeps working as a caller with no
adapter and no change at any call site. A different consumer can pass this
dataclass, a pydantic-settings object, or anything else with the attributes.

The values themselves come from a `cts.ini` written by hand - never from a
network request, and never from anything a model said. See the hub's write
governance for why that boundary matters once writes exist.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


def _system_cache_dir() -> Path:
    """The platform's place for re-downloadable data, out of the user's way
    and not synced: ~/Library/Caches on macOS, %LOCALAPPDATA% on Windows."""
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "sar_geo" / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "sar_geo"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "sar_geo"


#: Where the Claude Desktop extension caches terrain when its Cache folder
#: setting is blank or arrives unexpanded. Gigabytes it can download again,
#: so not in Documents, which many people sync to the cloud.
EXTENSION_CACHE_DIR = _system_cache_dir()
#: Where files meant for the person go (`save_export`), one folder per chat.
DEFAULT_EXPORTS_DIR = Path.home() / "Documents" / "sar_geo"

#: The folder placeholders Claude Desktop documents for extension settings.
#: It has been seen passing them through unexpanded, so they are filled here.
def _placeholders() -> dict[str, Path]:
    home = Path.home()
    return {"${HOME}": home, "${DOCUMENTS}": home / "Documents",
            "${DOWNLOADS}": home / "Downloads", "${DESKTOP}": home / "Desktop"}


def expand_folder(value: str | None) -> Path | None:
    """A folder setting as an absolute path, or None when it is unusable:
    blank, or still holding a `${...}` nothing could fill."""
    raw = (value or "").strip()
    for placeholder, folder in _placeholders().items():
        raw = raw.replace(placeholder, str(folder))
    raw = os.path.expanduser(os.path.expandvars(raw))
    if not raw or "${" in raw:
        return None
    return Path(raw).absolute()


def cache_dir_from_env() -> Path:
    """`ZA_CACHE_DIR` as an absolute path.

    Absolute because every file a tool writes is named by a path under it,
    and a relative path (`cache/staged/.../medevac.geojson`) means nothing
    to the person who has to find the file. Unusable under the extension
    (`SAR_GEO_LAUNCH_MODE` is set) means the system cache folder; elsewhere
    it keeps the old `./cache`, resolved against the working directory.
    """
    folder = expand_folder(os.environ.get("ZA_CACHE_DIR"))
    if folder is not None:
        return folder
    if os.environ.get("SAR_GEO_LAUNCH_MODE"):
        return EXTENSION_CACHE_DIR
    return Path("cache").absolute()


def exports_dir_from_env() -> Path:
    """`SAR_GEO_EXPORTS_DIR`, or ~/Documents/sar_geo."""
    return expand_folder(os.environ.get("SAR_GEO_EXPORTS_DIR")) or DEFAULT_EXPORTS_DIR


@runtime_checkable
class CaltopoCredentials(Protocol):
    """What `open_map` needs. Any object with these attributes will do."""

    caltopo_host: str
    caltopo_config_path: Path
    caltopo_account: str


@dataclass(frozen=True)
class CaltopoConfig:
    """A standalone implementation, for consumers with no settings object.

    `from_env` reads the same variable names the debrief app uses, so a
    machine already configured for that app needs nothing new to run this
    package on its own.
    """

    caltopo_host: str = "caltopo.com"
    caltopo_config_path: Path = Path("cts.ini")
    caltopo_account: str = ""
    #: Where downloads and map state are cached. None disables caching
    #: entirely, which is right for a test and wasteful for anything else.
    cache_dir: Path | None = None
    #: repr=False is not cosmetic: this value lands in a URL path, and a
    #: dataclass repr in a traceback or a log line would publish it.
    wms_token: str = field(default="", repr=False)
    #: A replacement for the shipped assignment norms (planning/norms.py),
    #: for another agency's corpus. None means the shipped file.
    norms_path: Path | None = None
    #: Keys for the static satellite chips a landing zone is cut from
    #: (sources/static_imagery.py); empty means that source is not asked. repr=False
    #: for the reason `wms_token` has it: each lands in a request URL.
    mapbox_token: str = field(default="", repr=False)
    google_maps_key: str = field(default="", repr=False)
    #: A CalTopo credential given directly, for a host that has no
    #: `cts.ini` to point at - the Claude Desktop extension's settings form
    #: passes these as environment variables. When the id and key are both
    #: set they are used instead of the file. repr=False: the key signs
    #: every request, and the id names whose account it signs for.
    caltopo_id: str = field(default="", repr=False)
    caltopo_key: str = field(default="", repr=False)
    caltopo_account_id: str = ""
    #: Where `save_export` puts files for the person: one subfolder per
    #: chat. Not the cache - that is the server's, this is theirs.
    exports_dir: Path = DEFAULT_EXPORTS_DIR

    def __post_init__(self) -> None:
        # Absolute however the config was built: every file a tool reports
        # is a path under the cache, and the user never sees the server's
        # working directory (review B22).
        if self.cache_dir is not None and not Path(self.cache_dir).is_absolute():
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).absolute())

    def wms_base_url(self) -> str | None:
        """The panel endpoint, or None when no token is configured.

        None rather than an exception: elevation and landcover need no
        credentials, so a caller without a token should still get
        everything it can have.
        """
        if not self.wms_token:
            return None
        return f"https://{self.caltopo_host}/api/{self.wms_token}/wms"

    @classmethod
    def from_env(cls) -> "CaltopoConfig":
        return cls(
            caltopo_host=os.environ.get("ZA_CALTOPO_HOST", "caltopo.com"),
            caltopo_config_path=Path(os.environ.get("ZA_CALTOPO_CONFIG_PATH", "cts.ini")),
            caltopo_account=os.environ.get("ZA_CALTOPO_ACCOUNT", ""),
            cache_dir=cache_dir_from_env(),
            wms_token=os.environ.get("ZA_WMS_TOKEN", "").strip(),
            norms_path=(Path(os.environ["ZA_ASSIGNMENT_NORMS"])
                        if os.environ.get("ZA_ASSIGNMENT_NORMS") else None),
            mapbox_token=os.environ.get("ZA_MAPBOX_TOKEN", ""),
            google_maps_key=os.environ.get("ZA_GOOGLE_MAPS_KEY", ""),
            caltopo_id=os.environ.get("ZA_CALTOPO_ID", "").strip(),
            caltopo_key=os.environ.get("ZA_CALTOPO_KEY", "").strip(),
            caltopo_account_id=os.environ.get("ZA_CALTOPO_ACCOUNT_ID", "").strip(),
            exports_dir=exports_dir_from_env(),
        )

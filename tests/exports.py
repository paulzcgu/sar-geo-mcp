"""Reading a GeoJSON file written for CalTopo, as the tests do.

Every such file holds one CalTopo Folder and puts every other feature in it
(`planning.export.in_one_folder`). `read_export` checks exactly that, then
hands back the collection without the folder, so a test reads the features
it is about.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_export(path) -> dict:
    collection = json.loads(Path(path).read_text(encoding="utf-8"))
    features = collection["features"]
    folders = [f for f in features if f["properties"].get("class") == "Folder"]
    assert len(folders) == 1, f"{path} holds {len(folders)} folders, not one"
    folder = folders[0]
    assert features[0] is folder and folder["geometry"] is None
    assert folder["properties"]["title"].startswith("SAR Geo ")
    members = features[1:]
    assert all(f["properties"].get("folderId") == folder["id"] for f in members), (
        "a feature is outside the file's folder")
    return {**collection, "features": members}

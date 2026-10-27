"""Components: reading a CalTopo map."""

from __future__ import annotations

from typing import Any

from ..caltopo.assignment import find_assignment
from ..caltopo.state import DEFAULT_MAX_AGE_SECONDS
from ..context import ServerContext, mcp_tool
from ..geo import bbox_center_wgs84, geometry_bounds_wgs84, to_web_mercator


def list_assignments(ctx: ServerContext, map_id: str, public: bool = True,
                     max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
                     ) -> dict[str, Any]:
    """The `list_assignments` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    state = ctx.state(map_id, public, max_age_seconds)
    assignments = []
    for feature in state.features:
        properties = feature.get("properties") or {}
        if properties.get("class") != "Assignment":
            continue
        geometry = feature.get("geometry") or {}
        label = (f"{properties.get('letter', '') or ''}"
                 f"{properties.get('number', '') or ''}").strip()
        entry: dict[str, Any] = {
            "label": label,
            "geometry_kind": geometry.get("type", "unknown"),
            "title": properties.get("title", ""),
        }
        try:
            latitude, longitude = bbox_center_wgs84(
                to_web_mercator(geometry_bounds_wgs84(geometry)))
            entry["center"] = {"latitude": latitude, "longitude": longitude}
        except Exception:
            # A feature with no usable geometry still belongs in the
            # list - its label is what a caller looks it up by.
            entry["center"] = None
        assignments.append(entry)
    return {"assignments": assignments, "provenance": state.provenance()}


def get_assignment(ctx: ServerContext, map_id: str, label: str, public: bool = True,
                   max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
                   ) -> dict[str, Any]:
    """The `get_assignment` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    state = ctx.state(map_id, public, max_age_seconds)
    assignment = find_assignment(state.features, label)
    payload = assignment.model_dump()
    try:
        latitude, longitude = bbox_center_wgs84(
            to_web_mercator(geometry_bounds_wgs84(assignment.geometry)))
        payload["center"] = {"latitude": latitude, "longitude": longitude}
    except Exception:
        payload["center"] = None
    payload["provenance"] = state.provenance()
    return payload


def list_map_features(ctx: ServerContext, map_id: str, public: bool = True,
                      feature_class: str = "",
                      max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
                      ) -> dict[str, Any]:
    """The `list_map_features` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    state = ctx.state(map_id, public, max_age_seconds)
    results = []
    for feature in state.features:
        properties = feature.get("properties") or {}
        if feature_class and properties.get("class") != feature_class:
            continue
        geometry = feature.get("geometry") or {}
        entry = {
            "id": feature.get("id"),
            "class": properties.get("class"),
            "title": properties.get("title", ""),
            "folder": properties.get("folderId"),
            "geometry_kind": geometry.get("type"),
            # A marker's symbol is what it MEANS - a command post, a clue,
            # a gate - and a caller filtering for staging areas cannot do
            # it from the title, which is free text somebody typed at
            # 02:00.
            "marker_symbol": properties.get("marker-symbol"),
            "point": None,
        }
        if geometry.get("type") == "Point":
            coordinates = geometry.get("coordinates") or []
            if len(coordinates) >= 2:
                # CalTopo writes four ordinates - lon, lat, and two it
                # fills with zero. Two numbers is not the payload this
                # tool summarises away; a track's thousands are.
                entry["point"] = {"latitude": coordinates[1],
                                  "longitude": coordinates[0]}
        results.append(entry)
    return {"features": results, "provenance": state.provenance()}


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="list_assignments", tags={"component"})
    def _list_assignments_tool(map_id: str, public: bool = True,
                               max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
                               ) -> dict[str, Any]:
        """List the search assignments on a CalTopo map.

        Returns each assignment's label, geometry kind (line or area), and
        centre point, alongside a `provenance` block saying how old the map
        data is. Use the label with `get_assignment` for full detail.

        `max_age_seconds` is how stale a cached copy may be before it is
        revalidated; pass 0 to force a check. Read the `provenance` before
        acting on anything time-sensitive - on a moving incident an
        assignment drawn five minutes ago may not be the one on screen now.
        """
        return list_assignments(ctx, map_id=map_id, public=public,
                                max_age_seconds=max_age_seconds)

    @mcp_tool(mcp, name="get_assignment", tags={"component"})
    def _get_assignment_tool(map_id: str, label: str, public: bool = True,
                             max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
                             ) -> dict[str, Any]:
        """One assignment in full: its geometry, description, and centre.

        `label` matches the way CalTopo shows it (QA103, QB102) and is
        case-insensitive. The `provenance` block says how old the geometry
        is; pass `max_age_seconds=0` to force a freshness check before
        acting on the boundary.
        """
        return get_assignment(ctx, map_id=map_id, label=label, public=public,
                              max_age_seconds=max_age_seconds)

    @mcp_tool(mcp, name="list_map_features", tags={"component"})
    def _list_map_features_tool(map_id: str, public: bool = True,
                                feature_class: str = "",
                                max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
                                ) -> dict[str, Any]:
        """Every feature on a map - markers, lines, polygons, tracks.

        `feature_class` filters by CalTopo's own class name (Marker, Shape,
        Assignment, ...). Geometry is summarised to its type rather than
        returned in full: a single track can carry thousands of coordinates,
        which would swamp a model's context to no purpose. Use
        `get_assignment` when the actual geometry is needed. A Marker also
        carries `marker_symbol` (CP, clue, gate, ...) and its `point` - two
        numbers, not the geometry this tool otherwise summarises away.
        """
        return list_map_features(ctx, map_id=map_id, public=public,
                                 feature_class=feature_class,
                                 max_age_seconds=max_age_seconds)

"""Scenario: a team back from an assignment - what the ground was, how much
of it their GPS tracks covered and where the gaps are, and the terrain
features worth noting, in one call."""

from __future__ import annotations

from typing import Any, Literal

from ..context import ServerContext, mcp_tool
from ._sections import (
    check_assignment,
    check_feature_kinds,
    check_overrides,
    check_track_names,
    coverage_facts,
    coverage_section,
    features_facts,
    features_section,
    ground_handle,
    guard,
    metrics_facts,
    metrics_section,
    overall_status,
    requested,
    skipped,
    stage_assignment,
    staleness_warnings,
    summarise,
    unused_overrides,
)

#: The sections, in the order they run and are reported: the metrics first,
#: because they save the sweep width an area assignment's coverage reads.
#: A type, so the three names reach the caller as an enum in the schema.
DebriefSection = Literal["metrics", "coverage", "features"]
SECTIONS: tuple[str, ...] = ("metrics", "coverage", "features")
#: The kinds `identify_features` knows, in the registry's order, so the
#: caller sees them as an enum (`list_feature_kinds` describes each). A test
#: holds this to `terrain.features.REGISTRY`, so a new detector cannot be
#: left out of it unnoticed.
FeatureKind = Literal["steep_band", "clearing", "drainage", "cliff_band", "route", "ingress",
                      "ridge", "cliff_pitch"]
#: The kinds a debrief asks about unless told otherwise: the ground that
#: slowed or stopped the team (steep and cliff bands) and the channels a
#: subject follows and a team crosses (drainages). Not clearings: they are
#: read off a vegetation panel, which ground `prepare_terrain` stages has
#: not got, so on it they always come back unknown. Not routes or ingress -
#: the metrics section already counts the routes by type and the ways in -
#: nor ridges, the crests that bound a planned block, which the metrics
#: already count by kind (and gaps cite by id) and which identify_features
#: finds with a second flow-routing pass over the ground, nor cliff
#: pitches, a rope-rescue question.
#: One list both the function and the wrapper default to, never changed in
#: place.
DEFAULT_FEATURE_KINDS: list = ["steep_band", "drainage", "cliff_band"]

#: What debrief_assignment sets in each feature call from its own arguments,
#: so an override may not: {section: {keyword: the argument to use instead}}.
SCENARIO_SETS = {"coverage": {"track_names": "track_names"}, "features": {"kinds": "kinds"}}

#: The one ground, as `staging`, `handles` and each section's `ground` name it.
GROUND = "assignment"

NOTE = ("No verdict: this lays out what terrain_metrics, coverage_for and identify_features "
        "measured, and judges neither the search nor the team. The covered fraction is the "
        "ground that came within half a modelled sweep width of a track chosen by its name "
        "for the team - where the GPS was, not where eyes went - and is not a probability of "
        "detection; set it beside the searchers' own account, and check the coverage "
        "warnings, which name any line chosen or set aside on a doubtful name. Which gaps to "
        "search again, and how, is the planner's call.")


def debrief_assignment(ctx: ServerContext, map_id: str, label: str, track_names: list[str],
                       public: bool = True,
                       kinds: list[FeatureKind] = DEFAULT_FEATURE_KINDS,
                       include: list[DebriefSection] | None = None,
                       overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The `debrief_assignment` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    # Every refusal first, before the map is read or anything staged: a
    # caller's mistake is theirs to hear at once, not three failed sections.
    # Every argument is checked whether or not its section runs.
    check_assignment(map_id, label)
    asked = requested(include, SECTIONS)
    tracks = check_track_names(track_names, needed="coverage" in asked)
    feature_kinds = check_feature_kinds(kinds, needed="features" in asked)
    keywords = check_overrides("debrief_assignment", overrides, SECTIONS, SCENARIO_SETS)

    staging = {GROUND: stage_assignment(ctx, map_id=map_id, label=label, public=public)}

    outcomes = {
        "metrics": guard(ctx, lambda: metrics_section(
            ctx, ground_handle(staging, GROUND),
            overrides=keywords["metrics"])) if "metrics" in asked else skipped(),
    }
    # After the metrics, which save the sweep width an area assignment's
    # coverage is computed with: a metrics section that failed fails an
    # area's coverage, naming why, rather than leave it to read a width from
    # a file this call did not write.
    metrics = outcomes["metrics"].section if "metrics" in asked else None
    outcomes["coverage"] = guard(ctx, lambda: coverage_section(
        ctx, ground_handle(staging, GROUND), track_names=tracks,
        overrides=keywords["coverage"], metrics=metrics)) if "coverage" in asked else skipped()
    outcomes["features"] = guard(ctx, lambda: features_section(
        ctx, ground_handle(staging, GROUND), kinds=feature_kinds,
        overrides=keywords["features"])) if "features" in asked else skipped()
    sections = {name: outcome.section for name, outcome in outcomes.items()}
    for name in asked:
        sections[name]["ground"] = GROUND

    # Each call revalidated the boundary and said so; the reader gets it
    # once - the latest check, the one nearest the moment of reading - and,
    # when the boundary moved or could not be checked, one warning naming
    # every section that answered: they all stand on the same staged
    # ground, so what a check says of it holds for each of them.
    checks = {name: outcome.staleness for name, outcome in outcomes.items()
              if outcome.staleness is not None}
    warnings = unused_overrides(keywords, asked)
    warnings.extend(staleness_warnings(checks))

    return {
        "status": overall_status(sections),
        "assignment": {"map_id": map_id, "label": label},
        "summary": summarise(sections, asked, FACTS),
        "sections": sections,
        "staging": staging,
        "handles": {name: record["handle"] for name, record in staging.items()},
        "staleness": list(checks.values())[-1] if checks else None,
        "exports": [],
        "warnings": warnings,
        "note": NOTE,
    }


#: Each section's top-line facts, read off its answered section (`summarise`).
FACTS = {"metrics": metrics_facts, "coverage": coverage_facts, "features": features_facts}


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="debrief_assignment", tags={"scenario"})
    def _debrief_assignment_tool(map_id: str, label: str, track_names: list[str],
                                 public: bool = True,
                                 kinds: list[FeatureKind] = DEFAULT_FEATURE_KINDS,
                                 include: list[DebriefSection] | None = None,
                                 overrides: dict[str, dict[str, Any]] | None = None
                                 ) -> dict[str, Any]:
        """A team back from an assignment, in one call: what the ground was, how
        much of it their GPS tracks covered and where the gaps are, and the
        terrain features worth asking them about. It stages the CalTopo
        assignment with `prepare_terrain`, then runs `terrain_metrics`,
        `coverage_for` with the team's tracks and `identify_features` on it,
        and returns their facts summarised, one section each. Nothing is
        judged.

        THE ASSIGNMENT is `label` on map `map_id`, as CalTopo shows it or as a
        person says it - `BT`, `bt-201` and `BT (201)` find the same one, and a
        label that could be several fails every section, naming them (see
        below). `track_names` are the team's GPS tracks on the same map, as
        coverage_for takes them: a team number such as "101" finds the lines
        named for that team, and a line whose name only loosely reads as the
        team's - a drone flight, a probability ring - is used only when
        nothing is plainly named for it, with a coverage warning naming it;
        one reaching the assignment that is set aside beside a plainly named
        track is named in a warning too. A name with no number in it, such
        as a call sign, finds only lines where it stands as a word of its
        own, and a warning names the lines it alone found: the assignment's
        label, passed by mistake, finds every team's lines on it. `public`
        (default true) reads the map without credentials; false reads it
        through the server's CalTopo account, for a private map.

        WHAT IT STAGES, AND WHAT THAT COSTS. One ground, through
        `prepare_terrain`, at what prepare_terrain costs: the assignment plus
        200 m round it, at a fixed 1,024-pixel raster, with elevation,
        vegetation and roads downloaded from public services on new ground, and
        only one map read when the ground on disk is already the assignment as
        it now stands (the common case; a moved boundary is staged afresh). How
        long a FIRST staging takes was not measured for this tool: allow
        minutes, not seconds. For comparison, `prepare_area` took from under
        a minute to two minutes a ground in the other scenarios (measured on
        2026-10-22, on grounds kilometres across, larger than most
        assignments), and an
        OpenStreetMap that cannot be reached is retried before it is given up.
        Then terrain_metrics, coverage_for and identify_features each read the
        map once more, to check the boundary has not moved, and compute from
        the staged files. The computing alone, measured on 2026-10-22 on a
        cached 44 ha assignment with every map read stubbed out, took under 3
        seconds for all three sections together (the metrics, the coverage of
        one track and the default kinds); on ground already staged, add one map
        read for the staging and one for each section. If the assignment cannot
        be staged - a map that cannot be read, a label that names no assignment
        or several - every section asked for fails with that reason, and
        nothing is raised.

        EVERY SECTION has `status` - "ok", "failed", or "skipped" when left out
        of `include` - `error` (null when ok, else "<ExceptionType>:
        <message>") and `warnings` (the tool's own). A skipped section has
        those three and nothing else. One asked for also has `ground`
        ("assignment"); once it has answered, its `handle` and compact facts;
        and `detail_call` - `{tool, arguments}`, the exact call that returns
        the full answer, geometry included. A section whose call raised keeps
        that call's `detail_call`, to retry it or make it directly; one that
        never made its call - the assignment could not be staged, or the
        coverage of an area with no metrics to take its sweep width from -
        has none, and its error says why. No outline comes back: only gap
        points.

        `metrics` - `terrain_metrics`, what the ground is. `area_m2` and
        `dem_resolution_m`; `elevation` (min, max, relief); `slope` (mean, 90th
        percentile and max degrees, and the share of ground in each band);
        `aspect` (dominant direction and the share facing each way); `landform`
        (the share of ridge, valley, bench and slope, and how many cliff
        bands); `hydrology` (`drainage_count`, `upstream_truncated` - true when
        a drainage enters through the edge of the staged ground, so
        contributing areas are understated - and each drainage, ranked by
        contributing area, with its length inside the assignment and in full,
        contributing area, bank-top depth and sidewall slope each side, and how
        many transects measured it); `vegetation` (LANDFIRE lifeform shares,
        the dominant one and the commonest types; canopy fraction and density
        class are read off a vegetation panel, which ground prepare_terrain
        stages has not got, so they are null and a warning says density is
        unknown); `traverse` (walking rate before and after vegetation, the
        `effective_sweep_width_m` - from the lifeform shares, at moderate
        density when the density is unknown - and the hours one pass takes);
        `access` (OpenStreetMap routes on the staged ground counted by type and
        the names among them, and `ingress_count`, how many points a trail,
        track or road crosses the boundary at - null when the ground has no
        OpenStreetMap, never "no routes"); and `shapes`, how many shapes of
        each feature kind the metrics identified. `warnings` are the metrics'
        data-quality warnings: a missing input is a warning and a null, never a
        zero. The section also saves the metrics beside the ground, which
        coverage reads and `slope_stats` and `elevation_profile` need.

        `coverage` - `coverage_for` with `track_names`. `assignment_kind`
        ("area" or "line"), `teams` (the names given), `tracks` (each selected
        track's name, length inside the assignment and in all, the share
        inside, and when it entered and left, null without GPS timestamps),
        `effective_sweep_width_m` (an area's from the metrics; a line's
        corridor width), `covered_fraction` (the share of the area, or of the
        line's length, within half a sweep width of a track), `effort_coverage`
        (track length inside times sweep width over the area - the Koopman
        coverage, null for a line), `zone_area_m2` or `line_length_m`,
        `time_entered`, `time_exited` and `time_searching_hours` (earliest in,
        latest out; null without timestamps). `gaps`, largest first as
        coverage_for lists them: each `id`, `area_m2` (an area's; null for a
        line) or `length_m` (a line's), `unsearched_shapes` (an area's only:
        the terrain shapes the metrics identified with at least a tenth of
        their own extent inside the gap, each id, kind and that fraction; never
        "the gap contains it". They are of every kind the metrics count in
        `shapes` - ridges and routes too - not only the kinds `features` lists,
        and `slope_stats` takes any of their ids), and `point`, a WGS84 Point
        inside the gap (on it, for a line) to put on a map or measure to;
        `dominant_vegetation` and `mean_slope_deg` are null today. An area's
        gaps under 500 m2, and a line's under 25 m, are not reported.
        `gap_count`, `gap_area_m2` and `gap_length_m` total them - 0 when no
        gap was found, null for the measure the assignment lacks. `closed_gaps`
        counts the slivers between sweep lanes, or the stretches of a line,
        folded into covered ground as GPS noise, with their total - not gaps,
        not unsearched. A track mostly outside the assignment is warned of,
        never dropped. An AREA's sweep width comes from the metrics: when the
        metrics section fails, coverage of an area does not run, and its error
        names the metrics failure. With metrics left out of `include`, coverage
        reads the metrics an earlier call saved on this ground (a re-staging
        discards them); with none saved, coverage of an area does not run, and
        its error says to add "metrics" to `include`. A LINE's coverage runs
        either way, on the width saved before or, with none saved, on
        coverage_for's 10 m default corridor; a warning says which whenever the
        metrics section failed or no width was saved.

        `features` - `identify_features` for `kinds`. Per kind: `count` (null
        when an input the kind reads is missing - a clearing with no
        vegetation panel, a route or way in with no OpenStreetMap: unknown,
        never the 0 of a real "none here"), `ids` (the same ids the metrics'
        shapes carry, which gaps cite and `slope_stats` takes), `summary`
        (the kind's own single facts, such as whether drainages are cut by
        the edge of the staged ground or the tallest cliff pitch's drop; the
        drainages' measurements are in the metrics) and its `warnings`.
        `kinds` defaults to steep_band, drainage and cliff_band: the ground
        that slowed or stopped the team, and the channels a subject follows
        and a team crosses. Name any `list_feature_kinds` describes; left out
        unless asked for are clearing (read off a vegetation panel, which
        ground prepare_terrain stages has not got, so it comes back unknown
        with a warning), route and ingress (the metrics already count them),
        ridge (the metrics already count ridges in `shapes`, and a gap cites
        their ids; listing them here costs a second flow-routing pass over
        the ground) and cliff_pitch (the drop you could rig on). Empty shapes
        on good input are a real answer, a count of 0; a missing input is a
        warning and a null count. The section's `warnings` are the kinds',
        each once, prefixed with the kinds that gave it unless every kind
        did.

        OVERALL `status`: "ok" when every section asked for is ok, "failed"
        when every one failed, otherwise "partial". Skipped sections count
        neither way. A failed section never stops the others: features still
        answer when coverage could not find the tracks.

        `summary` - the facts to read first, one block per section asked for,
        null when that section did not answer. `metrics`: `area_m2`,
        `relief_m`, `slope_mean_deg`, `slope_p90_deg`, `dominant_lifeform`,
        `density_class`, `effective_sweep_width_m`,
        `sweep_time_hours_per_pass`, `drainages`, `cliff_bands` and
        `ingress_points` (null without OpenStreetMap). `coverage`:
        `assignment_kind`, `covered_fraction`, `effort_coverage`, `tracks`,
        `gaps`, `gap_area_m2`, `gap_length_m`, `largest_gap` (id, area or
        length, point; null with no gap) and `time_searching_hours`.
        `features`: the count of each kind, null for a kind whose input is
        missing - as `ingress_points` is null on ground with no
        OpenStreetMap - never 0.

        `staleness` - the boundary check every computation makes, said once:
        the latest section's block (null when no section answered). Every
        section is computed from the same staged ground, so what the latest
        check says holds for all of them. When `current` is false the
        assignment boundary moved after staging, and a warning says every
        section that answered describes the OLD polygon: call again, and
        prepare_terrain re-stages it. When `checked` is false the boundary
        could not be checked at all - a different fact from checked and
        current - and a warning says so; coverage cannot read the tracks then
        and fails. A move an earlier check in the call found is warned of too,
        unless the latest found the boundary as staged again, when the warning
        says it may be being edited.

        `staging` - one record, `assignment`: `status`, `error`, `handle`,
        `map_id`, `label`, `staged` (false when the ground on disk was already
        current), `reason`, the boundary `fingerprint`, `warnings` (vegetation
        or roads unavailable) and the `prepare_terrain` call as `detail_call`.
        `handles` maps it to its handle (null when it could not be staged), for
        any tool that takes one: `slope_stats` on a shape id,
        `elevation_profile` along a drainage, `site_facts`,
        `describe_boundary`.

        `exports` is always empty: a debrief draws nothing new for CalTopo -
        the assignment and the tracks are already on the map - and writes no
        file. `get_assignment` carries the assignment's outline, and
        coverage_for's own answer the gaps' - in Web Mercator (EPSG:3857)
        metres, the coverage grid's coordinates, not WGS84 longitude and
        latitude, so not GeoJSON a map takes as it stands; the `point` here is
        WGS84. There is no medical section either: a debrief is about ground
        already searched, and `plan_medevac` or `nearest_medical_facilities`
        answer where a patient would go.

        `overrides` - keyword arguments for the tool behind a section, by
        section, as the other scenarios take them. Today none of the three
        takes a keyword this tool does not set itself - the handle, and
        `track_names` and `kinds`, which are its own arguments - so every
        keyword is refused, naming why, as are a section it does not have and a
        value that is not a mapping. Blank `map_id` or `label`, an unknown or
        empty `include`, `track_names` that is not a list of names or is empty
        while coverage is asked for, and an unknown feature kind are refused
        the same way, before the map is read.

        `assignment` echoes the map id and label the call was made for.
        `warnings` are the call's own, beside each section's: the boundary
        warnings `staleness` describes. `note` says in a few sentences what
        NO VERDICT below says, to show beside any answer.

        NO VERDICT. This lays out what the three tools measured and judges
        neither the search nor the team. Coverage is geometry: ground within
        half a modelled sweep width of a GPS track - where the receiver was,
        not where eyes went - and the sweep width is a table by lifeform and
        density, not a detection experiment. `covered_fraction` is not a
        probability of detection; set it beside the searchers' own account of
        what they saw. Tracks are chosen by name, so a mislabelled line counts
        or a team's real track is missed: check `tracks` and the coverage
        warnings, which name any line chosen or set aside on a doubtful
        name. The terrain is read
        off a 1,024-pixel elevation raster and public vegetation and road
        layers; a feature smaller than the raster resolves is not there. Which
        gaps to search again, by whom and how, is the planner's call. Drill
        down with each section's `detail_call`, and on a shape id with
        `slope_stats` or `elevation_profile`.
        """
        return debrief_assignment(ctx, map_id=map_id, label=label, track_names=track_names,
                                  public=public, kinds=kinds, include=include,
                                  overrides=overrides)

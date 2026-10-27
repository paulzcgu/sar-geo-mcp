# AGENTS.md

How this repository works, for coding agents and developers. README.md is
for the people who use SAR Geo; this file is for changing it.

## Ground rules

- A wrong coordinate is worse than no coordinate; claims and computations
  are recorded side by side, never merged; no logic lives in an adapter.
- Read-only: nothing writes to CalTopo (see *What is not here, and why*).
- stdout is the MCP protocol: nothing may print there (see *Running from
  source*).
- Every change that ships gets a `# <version>:` block above `version = ...`
  in `pyproject.toml`, saying what was wrong and what changed. It becomes
  the release notes.
- Tests: `.venv/bin/python -m pytest -q -m "not live"`. Live tests hit the
  network and are opt-in.
- No real search data, anywhere: not in code, tests, fixtures, commit
  messages, PRs or release notes. That means no subject or team member
  names, incident numbers, real map ids, or real places last seen, finds
  or landing zones. Use made-up values (see *Keeping real search data
  out*).

## Two ways in, one implementation

The server is an adapter and holds no logic of its own. Every tool is a thin
call into a module that is independently tested, and any Python consumer can
import those modules directly and skip the server entirely.

```python
from sar_geo_mcp.coords import parse_location, format_mgrs
parse_location("10S EG 61234 41234", near=(37.40, -122.32))
```

Every tool is also a plain function of its own name that takes a
`ServerContext` first and returns exactly what an MCP caller gets - for
example `sar_geo_mcp.features.lz.lz_candidates(ctx, handle, ...)` - in
`sar_geo_mcp.components`, `.features` or `.scenarios`. That is the tool
layering, and the reason for it, below.

## Running from source

```bash
pip install -e ".[mcp]"
```

The `mcp` extra is what pulls `fastmcp`; without it the library still works
and only the server entry point is unavailable. Then point an MCP client at
it — for Claude Code:

```bash
claude mcp add sar-geo -- sar-geo-mcp
```

`--norms PATH` (or `ZA_ASSIGNMENT_NORMS`) replaces the shipped assignment
norms with another agency's file. A candidate set the norms decide - hasty
lines, flagged against their bands, and urban assignments sized to their
median - is made afresh under another file, under a new set id, rather than
read back from the cache as it was made under the old one. A set or
candidate id already held still fetches the set it named, as it was
measured then: ask for the set again to have it read against the new norms.

Transport is stdio: a subprocess on your own laptop. No port, no
certificate, no hosting bill, no new attack surface. That matches how this
project deploys everywhere else, and it is why remote transport is a
separate decision with a threat model attached rather than a flag.

**stdout is the protocol.** Anything a dependency prints there corrupts the
JSON-RPC stream — `caltopo_python`'s own log handler did exactly that, and
the server now repoints stdout handlers to stderr before it starts. If you
add a dependency, keep it off stdout.

## Keeping real search data out

This repository is public, and so are its CI logs.
`scripts/check_sensitive.py` refuses real search data and credentials in
these places:

| Where | How | What it reads |
|---|---|---|
| Commit | `.githooks/pre-commit`, `.githooks/commit-msg` | Staged lines, commit message |
| Push | `.githooks/pre-push` | Every commit the remote lacks; an old branch sends its whole history |
| PR, push to master | `.github/workflows/check.yml` | Diff, every commit message, PR title and body |
| Release | `scripts/release.py` | Commits since the last tag, the release notes |
| Claude Code | `scripts/claude_pr_guard.py` (`.claude/settings.json`) | Text a `gh pr/issue/release` command would publish |

Turn the git hooks on once per clone:

```bash
git config core.hooksPath .githooks
```

It runs two sets of rules:

- **Generic rules, in the repository.** Any coordinate more than 250 m
  from every point in `tests/fixture_points.py` is refused. So is any
  MGRS or UTM reference or CalTopo map id not listed there, and any data
  file: GPX, KML, GeoJSON, GeoTIFF, a staged `manifest.json`, `cache/`,
  `docs/`, `exports/`, credentials. To use a made-up value, add it to
  `tests/fixture_points.py`, or mark its line `fixture-ok`.
- **A private denylist, outside the repository.** It lists real names,
  incident numbers, map ids (`map ID`), patterns (`re ...`) and places
  (`point LAT LON RADIUS_M`). Keep it at `~/.config/sar_geo/denylist.txt`
  or set `SAR_SENSITIVE_DENYLIST`. For CI, put its text in the repository
  secret `SAR_DENYLIST`. Nothing excuses a denylist hit, and a finding
  names the line, never the text it matched.
- **Credentials.** The check reads the real values on this machine: every
  value and account name in `cts.ini`, the token and key variables in
  `.env`, and the `ZA_*` token and key variables in the environment. Any of
  them appearing anywhere is refused, and `fixture-ok` does not excuse it.
  Known token formats are refused too: GitHub, PyPI, Google, Mapbox, AWS,
  Slack, Anthropic and OpenAI keys, JWTs, private keys, and a password in
  a URL (unless the host is `.example`, `.test`, `.local` or localhost).
  So is a long, random-looking value given to a setting with a secret
  name. So are key files (`.pem`, `.key`, `id_rsa`, `.pypirc`,
  `credentials.json`). A test's fake token that looks real carries
  `fixture-ok`. GitHub secret scanning and push protection are on for the
  public repository as a second net.

`--history` scans every commit on every branch, and `--tree` scans the
working copy of every tracked file.

Only the PR workflow cannot be skipped. `git commit --no-verify` skips
the local hooks, so the workflow is the backstop. A check cannot clean
history: if real data was ever pushed, rewrite the history and rotate
anything it exposed.

## The Claude Desktop extension and releasing

`dist/sar-geo.mcpb` is a one-file install for the Claude desktop app
(macOS and Windows): open it and click Install. Claude Desktop asks for
the optional keys in a form and keeps them in the system keychain. Build
it with:

```bash
.venv/bin/python scripts/build_extension.py
```

The file holds only a launcher (`extension/src/launcher.py`). Each time
Claude Desktop starts it, the launcher asks PyPI for the newest
`sar-geo-mcp` and runs that with `uv`. So friends install once, and every
release reaches them on their next Claude restart. Offline, it runs the
copy already cached. The "Pin version" setting holds one version.
Files a user asks to save go in `~/Documents/sar_geo/<chat folder>`
(`SAR_GEO_EXPORTS_DIR`); downloaded terrain is cached in the system cache
folder (`~/Library/Caches/sar_geo` on macOS, `%LOCALAPPDATA%\sar_geo` on
Windows), not in Documents, which many people sync to the cloud.
Rebuild and resend the `.mcpb` only when something in `extension/`
changes.

**Releasing** is one command. Write the `# <version>:` comment block above
`version = ...` in `pyproject.toml` (it becomes the release notes), bump
`version`, commit, then:

```bash
.venv/bin/python scripts/release.py
```

It checks the tree is clean, the tag is new and the notes exist, runs the
tests, then pushes the branch and an annotated tag `v<version>`. The tag
starts `.github/workflows/release.yml`, which tests again, publishes to
PyPI, and then creates the GitHub Release with those notes and the wheel,
sdist and `sar-geo.mcpb` attached. `--dry-run` runs every check and
stops before tagging.

A tag is a release to everyone with the extension, so tag only what you
would hand to a team in the field. One-time setup: on pypi.org, add a
pending trusted publisher for project `sar-geo-mcp`, repository
`paulzcgu/sar-geo-mcp`, workflow `release.yml`, environment `pypi`.

Users can ask their Claude `check_for_update` ("am I on the latest
sar-geo?") and `key_status` ("why can't I see imagery?").

## Credentials and keys, in detail

Public maps need none: `public=True` uses the unauthenticated endpoint, and
that is how most of this code has been exercised. Authenticated reads want a
`cts.ini` next to where you run the server:

```ini
[account]
id = YOUR_ACCOUNT_ID
key = YOUR_KEY
```

Or give the credential directly as `ZA_CALTOPO_ID`, `ZA_CALTOPO_KEY` and
(for PDF printing only) `ZA_CALTOPO_ACCOUNT_ID`, which is how the Claude
Desktop extension passes it; when the ID and key are both set they are
used instead of the file.

A **read-only** team key is sufficient and is all v1 ever wants. `cts.ini`
is gitignored and must stay that way. A handle reads its map by the path
its staging named, or by the one a later `prepare_terrain` or
`prepare_area` call for it - a scenario's own staging included - has since
read the map by. So a call corrected to `public=False` on ground already
staged reads a private map through the account from then on, without
staging the ground again, while a later call whose own read failed by its
path - `public` left at its default `True` on a private map - says so in
its warnings and leaves the path as it was. `ZA_CALTOPO_HOST` (or
`--caltopo-host`) selects the host for team subdomains.

The raster tools, the canopy panel fetch and the imagery chips read
through a separate `wms_token` (`ZA_WMS_TOKEN`), from the CalTopo account
page's API Access block.

Two more keys are optional and add sources to `lz_candidates`' imagery
chips: `ZA_MAPBOX_TOKEN` for Mapbox Satellite, `ZA_GOOGLE_MAPS_KEY` for
Google Satellite. Neither is required - the CalTopo chips work without
them - and neither key nor the token ever reaches a log or a returned
warning: a failed fetch is reported by exception type and status only.
Google's terms of service restrict how its static-image tiles may be
used and displayed; a deployment that configures `ZA_GOOGLE_MAPS_KEY`
must satisfy itself of that before doing so. A Mapbox chip is cut with
`attribution=false` (its text credit would sit under the patch outline
and go unread) but keeps the Mapbox logo on the image; the text credit
travels in the envelope instead. Every chip carries an `attribution`
field - "© Mapbox © OpenStreetMap © Maxar" for Mapbox, "Imagery © Google"
for Google (which also draws its own logo and credit on the image),
"CalTopo" and "CalTopo, USDA NAIP" for the CalTopo layers - and any page
or report that shows a chip must show its `attribution` beside it. A
static image is held in memory only; the outlined chip under `lz/` is
the one file written.

Authenticated maps are read through one held session per map for the
life of the server process. The library behind it leaves a thread behind
for every session ever constructed, so the server never opens a second
one; a Python consumer that reads a private map repeatedly should use
`sar_geo_mcp.caltopo.reader.MapReader` for the same reason.

## Architecture: tool layers

Every MCP tool this server exposes belongs to exactly one of three layers.
Below them sit two layers of plain Python that are never MCP tools.

```
scenario    plan_medevac, plan_initial_response, ...          ┐
feature     lz_candidates, egress_routes, hasty_candidates    ├ MCP tools
component   parse_coordinate, get_candidate, cache_status     ┘
─────────────────────────────────────────────────────────────────
domain      terrain/, coverage/, planning/ algorithms,          ┐ internal
            staging, prefetch, handles, coords                   │ libraries
infra       net, net_errors, config, cache/, tiles, geo,         │
            CalTopo session and WMS transport                    ┘
```

## The layers

**Component** - a building block with one purpose: read a map, convert a
coordinate, stage ground, fetch one candidate, report the cache. Cheap,
predictable, composable. Lives in `src/sar_geo_mcp/components/`.

**Feature** - a multi-step answer to one planning question: where can a
helicopter land, what is the way out, which lines can a team run on day
one. A feature fuses several sources (elevation, vegetation, OSM, CalTopo
layers, agency data) and usually returns a candidate set. Lives in
`src/sar_geo_mcp/features/`.

**Scenario** - a whole planning situation that needs several features at
once: an initial response, a medevac, an urban search, an assignment
debrief. Lives in `src/sar_geo_mcp/scenarios/`.

**Domain** - the SAR logic itself: what a ridge is, how an LZ is graded,
how fast a walker moves through brush. Independently tested.

**Infra** - plumbing with no SAR knowledge: HTTP and retries, the disk
cache, tile arithmetic, CalTopo login and WMS transport.

## Rules

1. **A tool is a plain function first.** Every tool, in every layer, is a
   module-level function of the tool's own name that takes the
   `ServerContext` first, then the tool's arguments, and returns the whole
   payload, envelope fields (`staleness`, `warnings`, `map_provenance`)
   included. All the logic is there. The module's `register(mcp, ctx)`
   wraps each function as an MCP tool, and the wrapper only calls it: it
   carries the operator-facing description and the signature MCP builds
   the schema from, and nothing else. A scenario, a script or a test calls
   the function and gets what an MCP caller gets, without MCP.

   ```python
   # features/lz.py
   def lz_candidates(ctx: ServerContext, handle: str, center_latitude: float,
                     center_longitude: float, limit: int = 10, ...) -> dict[str, Any]:
       """The `lz_candidates` tool as a plain function: ... (one paragraph,
       pointing to the tool's description)"""
       resolved = ctx.staged(handle)
       payload = lz_candidates_impl(resolved.directory, ...)
       payload["staleness"] = resolved.status
       return payload

   def register(mcp, ctx) -> None:

       @mcp_tool(mcp, name="lz_candidates", tags={"feature"})
       def _lz_candidates_tool(handle: str, center_latitude: float,
                               center_longitude: float, limit: int = 10,
                               ...) -> dict[str, Any]:
           """Where a helicopter can put down ... (the full description: what
           each field means, what null means, what is not checked)"""
           return lz_candidates(ctx, handle=handle, center_latitude=center_latitude,
                                center_longitude=center_longitude, limit=limit, ...)
   ```

   The wrapper is named `_<tool>_tool` in the source, because a wrapper of
   the tool's own name would shadow the function inside `register`. It is
   registered through `context.mcp_tool`, not `mcp.tool` directly:
   `mcp_tool` passes `name=` and also sets the wrapper's `__name__` to the
   tool's name. FastMCP builds the argument validator from the wrapper, and
   pydantic names the function in every validation error, so without the
   rename a caller who passes a bad argument would read
   `call[_lz_candidates_tool]`, a tool that does not exist. The wrapper's
   parameters are the function's after `ctx` - names, order, annotations
   and defaults - and it passes each one, by keyword, to the parameter of
   the same name. `tests/test_tool_registration.py` holds every tool to
   that shape, the forwarding and the name included, and
   `tests/test_plain_functions.py` checks that a direct call answers as
   the MCP call does.

   A direct call is not validated against the tool's schema the way an MCP
   call is. The caller is Python and passes the declared types (a
   `BoundaryTarget`, not a dict), and nothing checks a `Literal` value on
   the way in: `returns="images"` reaches the function as it is. The domain
   functions behind the `Literal` parameters refuse an unknown value with a
   `ValueError` (egress `mode`, hasty `include`, unmarked `sources`, relief
   and slope `returns`), and a new one must too, or a typo in a scenario
   becomes a quietly reduced answer rather than an error. A direct payload
   may also share objects with what the server holds, which a serialised
   MCP answer never does: treat it as read-only. `assignment_norms`
   returns a copy, because its source is the norms every later call reads.
2. **Dependencies point down only.** scenario -> feature -> component ->
   domain -> infra. A scenario calls feature *functions* directly, never
   through MCP. Features do not call other features' tools; logic two
   features share moves down into domain. `tests/test_layering.py`
   enforces this.
3. **Every tool carries its layer as a tag** (`component`, `feature`,
   `scenario`), so a client can filter by layer. The layer is NOT part of
   the tool name: renames break clients, and the prefix costs tokens.
4. **Scenarios summarise.** A scenario returns one section per feature,
   each with its own status, so a failed section (say, a medical lookup
   with no network) does not sink the rest. It returns candidate ids, not
   geometry; the caller drills down with `get_candidate`. It takes
   defaults plus an `overrides` mapping rather than every feature's
   parameters.
5. **Naming.** Components: plain verb or noun (`parse_coordinate`,
   `get_candidate`). Features: what they produce (`lz_candidates`,
   `egress_routes`). Scenarios: `plan_<situation>` or the action
   (`debrief_assignment`).

## Shared state

`ServerContext` (`src/sar_geo_mcp/context.py`) carries what every layer
needs: the config, the one `MapReader`, the assignment norms, `state()` (a
map read) and `staged()` (a handle, revalidated). `server.py` builds it
and calls each layer's `register`, which hands it to every wrapper; a
scenario passes the one it was given straight on to the feature and
component functions it calls. The tool layers import `context` to name the
type of `ctx` and for `mcp_tool`, which registers every wrapper; domain and
infra never import it, and nothing imports `server`, `context` included
(`tests/test_layering.py`).

## Where each tool lives

| Layer | Module | Tools |
|---|---|---|
| component | `components/caltopo.py` | `list_assignments`, `get_assignment`, `list_map_features` |
| component | `components/cache.py` | `prefetch_map`, `cache_status`, `evict_cache` |
| component | `components/coords.py` | `parse_coordinate`, `convert_coordinate`, `distance_bearing` |
| component | `components/geometry.py` | `point_in_geometry`, `geometry_bounds`, `boundary_clearance`, `buffer_geometry` |
| component | `components/staging.py` | `prepare_terrain`, `prepare_area` |
| component | `components/ground.py` | `terrain_metrics`, `site_facts`, `list_feature_kinds`, `identify_features`, `slope_stats`, `elevation_profile` |
| component | `components/norms.py` | `assignment_norms` |
| component | `components/candidates.py` | `get_candidate`, `caltopo_export` |
| component | `components/files.py` | `save_export` |
| component | `components/setup.py` | `key_status`, `check_for_update` |
| feature | `features/relief.py` | `shaded_relief`, `slope_shading` |
| feature | `features/unmarked_paths.py` | `unmarked_paths` |
| feature | `features/boundary.py` | `describe_boundary` |
| feature | `features/hasty.py` | `hasty_candidates` |
| feature | `features/parking.py` | `parking_candidates` |
| feature | `features/lz.py` | `lz_candidates` |
| feature | `features/egress.py` | `egress_routes` |
| feature | `features/travel_rings.py` | `travel_time_rings` |
| feature | `features/medical.py` | `nearest_medical_facilities` |
| feature | `features/urban.py` | `populated_areas`, `urban_assignments` |
| feature | `features/coverage.py` | `coverage_for` |
| scenario | `scenarios/medevac.py` | `plan_medevac` |
| scenario | `scenarios/initial_response.py` | `plan_initial_response` |
| scenario | `scenarios/urban_search.py` | `plan_urban_search` |
| scenario | `scenarios/debrief.py` | `debrief_assignment` |

## Scenario conventions (`scenarios/_sections.py`)

What every scenario shares lives in `scenarios/_sections.py`, and a new
scenario reuses and extends it rather than forking it. One module per
scenario (`scenarios/medevac.py`), listed in `scenarios.MODULES`, tagged
`scenario`, written as rule 1 says: a plain function taking `ctx` first,
and a wrapper that only calls it.

- **Sections.** A scenario answers in sections, one per feature. Each is a
  dict with `status` (`ok`, `failed` or `skipped`), `error` (null when ok,
  else `"<ExceptionType>: <message>"`, with any credential the server holds
  blanked out), `warnings` (the feature's own) and its compact facts.
  `guard` runs a section and records any `Exception` - never
  `BaseException` - as its failure, so one section never raises out of the
  scenario or stops another. A section made of several calls fails when
  any of them failed but keeps the ones that answered: egress, one call per
  mode, each with its own status under `modes`; urban, the blocks and then
  their grouping, the blocks under `blocks` when the grouping raised. The
  overall `status` is `ok` when every section asked for
  is ok, `failed` when every one failed with nothing to show
  (`answered_in_part`), otherwise `partial`; skipped sections count neither
  way, and nor does an `export` section (`EXPORT_SECTION`), which finds
  nothing of its own and whose every problem is a warning - so a full disk
  never turns `ok` into `partial`, nor an export with nothing to draw
  `failed` into `partial`.
- **Refused before any work.** The scenario's own arguments - bad
  coordinates (`check_point`), a radius `prepare_area` would refuse
  (`check_radius`, which runs `build_area` and the pixel rule itself,
  because `prepare_area` measures its own circle and refuses 4,500 m by a
  hair; `radius_in_metres` also refuses a `radius_unit` it does not know,
  or one other than metres beside `radius_m`, which is metres by name), an
  unknown or empty `include`, or one naming only `export`
  (`requested`), bad `overrides` - names and values both, below
  (`check_overrides`, and
  `check_egress_overrides` for an egress keyword whose default differs by
  mode when both modes run), a limit that is not
  a whole number from 0 (`check_limit`), `medical_per_category` outside
  1-10 (`check_per_category`), ring times the rings tool would refuse
  (`check_ring_minutes`), a hasty `corridor_widths_m` override that is
  not one (`check_hasty_overrides`, which also refuses a misspelt class
  the model would silently ignore), a walking-model override the rings
  tool would refuse (`rings_ground`, through the feature's own
  `rings_model`), an export `label_prefix` `caltopo_export` would
  refuse or a blank export `resource_type` (`check_export_overrides`), a
  `target_ha` not above zero
  (`check_target_ha`), a block rule or marker colour (null included) the
  urban tools would refuse (`check_urban_overrides`, through
  `planning.populated.check_params`) and a search circle too wide to read
  lot lines over (`check_urban_radius`, through the feature's own
  `panel_pixel_m`), a blank map id or assignment label
  (`check_assignment`), track names that are not a list of names, or none
  while coverage is asked for (`check_track_names`), and a feature kind
  `identify_features` does not know (`check_feature_kinds`, in its own
  words) - raise `ValueError` before anything
  is staged or fetched. That is a caller's mistake, not a
  section failure. A search radius under 50 m (`SMALL_RADIUS_M`) is not
  refused but warned of at the top level, by every scenario taking a
  `radius_unit` (`small_radius_warnings`): it is most likely half a mile
  asked as `radius=0.5` with the unit left out, and the answer would read
  as a search of the area - an empty hasty section as "no lines here".
- **`include`** is a list typed as a `Literal` of the scenario's section
  names (None is all); a section left out is `skipped` and stages nothing.
- **`overrides`** is `{section: keyword arguments}`, passed to every call
  that section's feature function gets. An unknown section, a keyword the
  feature does not take (the refusal lists those it does, read once from
  the real function), and a keyword the scenario sets itself - the handle,
  the centre, and every keyword a helper always passes (`limit`, `mode`,
  `per_category`), held in `SECTION_TOOLS` so a scenario cannot forget one
  - are refused, each saying why; a scenario's `scenario_sets` only names
  its own argument to use instead, such as `lz_limit`. The `export`
  section's `handle`, `set_id` and `candidate_ids` are the scenario's (it
  draws the set its own section made); `include_area` is passed as a
  default the caller may override, and a scenario may set more from its
  own arguments (`settings`, spread after the overrides so they win as
  the helper's own keywords do, and named in its `scenario_sets` too so
  the caller hears a refusal: plan_urban_search drafts in its
  `resource_type`, with house markers exactly when `building_markers`
  asked for them). A section that calls
  more than one feature (`SECTION_PARTS`) takes its overrides by tool,
  each mapping checked against its own tool: the urban section's
  `{"populated_areas": {...}, "urban_assignments": {...}}`, because both
  take `max_distance_m` and a flat mapping could not say which call it
  is for. A flat mapping there is refused with the shape it needs, and a
  keyword the scenario sets is refused as the scenario's only for a tool
  that takes it - given to one that does not, it is refused as a stranger,
  with what that tool accepts. A tool that takes no keyword the scenario
  does not set (`terrain_metrics`, `coverage_for` and `identify_features`
  in `debrief_assignment`) refuses every one, saying there is nothing to
  override. Every value is then read as the feature tool's own schema
  reads it over MCP (`_read_values`, through pydantic adapters over the
  real functions' annotations, `FEATURE_TYPES`): the `overrides` schema
  is an untyped mapping, so nothing converts a value on the way in, and
  read as given "false" is truthy - it would draw the very search area a
  caller left out. A value the schema converts is passed on converted
  ("6" as 6.0, "false" as false, a whole number where a float goes kept
  as the caller wrote it), in the call and its `detail_call` alike; one it
  refuses - text for a number, a null slope, a hasty kind that is not
  one, a bare string where a list goes - is refused naming the keyword,
  as is one the feature's own range checks would refuse once staged
  (`FEATURE_CHECKS`: `hasty_params`, and the `check_params` of the
  landing-zone, egress and medical domain modules, run over the tool's
  defaults). One refusal is the scenario's own, not the schema's: a
  number that is not finite ("nan", "inf"), which the schema takes but
  no range check can be trusted to catch - NaN passes every one, and the
  section would answer `ok` with nothing found. The few keywords a
  scenario check already reads in the domain's words (`DOMAIN_CHECKED`:
  the corridor widths, the walking pace, the label prefix, a
  house-marker colour) are left to that check, their text read by the
  schema first so a pace of "0.5" is 0.5.
  Overrides given for a section left out of `include` are
  accepted, since leaving a section out is no mistake, but no call took
  them: `unused_overrides` says so in the top-level `warnings`. An
  override reaches its own section's calls and no other, so where two
  sections measure one thing it can part them: a `travel_rings` walking
  model (`RINGS_MODEL_KEYWORDS`) re-walks the rings, but
  `hasty_candidates` takes none and walks its
  `travel_minutes_from_center` on the default model. When hasty answers in
  a call that asked for the rings on another model, `hasty_pace_warnings`
  says so at the top level, naming the keywords, rather than leave a line
  6 minutes away to read as inside a 12-minute ring at half pace that
  stops short of it - or, with a cliff limit of the rings' own, a null
  minute to read as ground the subject could not reach.
- **Staging.** `stage_grounds` stages each distinct `Ground` (a radius
  round the scenario's point at a resolution) once through the
  `prepare_area` component and shares the handle among the sections that
  can use it. A ground that cannot be staged is a failed record, and
  `ground_handle` raises `StagingFailed` inside the sections standing on it
  alone. Every handle comes back in `handles`. Each feature needs its own
  ground: `lz_candidates` at `LZ_RESOLUTION_M` (2.0); `egress_routes` and
  `travel_time_rings` at `WALKING_RESOLUTION_M` (10.0) centred on the point
  (they refuse ground over 1,800 x 1,800 cells); hasty, parking, populated
  areas and urban assignments at the default. `rings_ground` sizes the
  rings' ground for the longest ring by the rings tool's own rule
  (`planning.travel_rings.radius_to_fit`, the radius its edge warning
  names, on the model's quickest ground, `fastest_factor`), capped at
  4,499 m (`LARGEST_RADIUS_M`), the largest `prepare_area` stages and the
  largest the edge warning ever names. A scenario on one CalTopo assignment
  rather than a point stages it with `stage_assignment`, through the
  `prepare_terrain` component, into a record of the same shape that names
  the assignment and its map instead of a radius; `ground_handle` fails
  its sections the same way. A section can also stand on another section:
  coverage of an area assignment takes its sweep width from the metrics
  `terrain_metrics` saves, so when the metrics section fails, coverage of
  an area raises `DependencyFailed` naming that failure rather than read a
  width this call did not measure; with the metrics left out of `include`
  it reads the width an earlier call saved on that ground (a re-staging
  discards it), and with none saved raises `DependencyFailed` saying to
  add "metrics" to `include`, rather than pass on coverage_for's own
  refusal, which names a local path and a command line. A line, which has
  a fallback width, still runs, with a warning saying which width.
- **One helper per feature** - `lz_section`, `egress_section`,
  `medical_section`, `parking_section`, `hasty_section`, `rings_section`,
  `urban_section` (populated_areas, then urban_assignments),
  `metrics_section`, `coverage_section`, `features_section` (on an
  assignment handle),
  `set_export_section` (`caltopo_export` of the set another section made),
  more as scenarios need them - each calling the feature function through
  its module, never MCP and never `planning` directly, so the envelope
  matches the tool's and a test can stub the function where it lives. The
  top-line facts read off an answered section are shared the same way:
  `lz_facts`, `egress_facts`, `medical_facts`, `parking_facts`,
  `hasty_facts`, `rings_facts`, `urban_facts`, `metrics_facts`,
  `coverage_facts`, `features_facts`.
- **The boundary check, once.** Every computation on an assignment handle
  revalidates the boundary and returns a `staleness` block. A helper keeps
  it off its section (`Outcome.staleness`), and the scenario returns the
  latest one at the top level. Every section stands on the same staged
  ground, so what a check says of that ground holds for all of them:
  `staleness_warnings` lets the latest check decide, as the top level
  does, and says a boundary that moved, or could not be checked, of every
  section that answered - never only of the one whose own check saw it.
- **Compact output.** Points (a touchdown, an exit, a hospital, a point
  inside a coverage gap), never a polygon, route, ring or outline. Every
  call a section makes carries
  `detail_call`, `{"tool", "arguments"}`, the exact feature call that
  returns the full answer - a helper makes the call through
  `call_feature`, so a call that raised leaves its `detail_call` on the
  failed section too; a section whose ground could not be staged made no
  call, and its ground's staging record carries the `prepare_area` call. A
  candidate set's section returns its `set_id` and candidate ids for
  `get_candidate`.
- **Export.** What a scenario found goes into one CalTopo-importable
  GeoJSON at `<staged dir>/exports/`, beside its primary ground
  (`write_export`); the paths come back in `exports`. Existing exporters
  and conventions are reused - the medical tool's markers, `point` and
  `firstaid` symbols, `caltopo_export` for a candidate set. A marker says
  what was not checked there, and what a feature's own point marker would
  have said moves onto the scenario's point (`Outcome.point_notes`), such
  as how the hospitals were ranked - by road time, by straight-line
  distance when the road router could not be reached or found no road, and
  nothing when no hospital is drawn - and where the hospital road times
  start. A point note, like every line in the file, says only what is true
  of what is drawn. An exit's line says its vehicle
  end (`route_line`): the road, an unnamed one by its kind, or the lot and
  the road it opens onto; the road's access, tagged or not; and its gates -
  crossed on the way, on the stretch of road the exit lies on (one
  OpenStreetMap way, within the staged ground) or none mapped there, not
  checked on the road in beyond that stretch or into a lot, or unknown. A
  line silent about a gate reads as open. On ground that cannot tell, a
  road exit's `gate_on_road` is null, but an exit in a lot with no road
  beside it carries nothing that says so - only the egress answer's own
  warnings do - so `egress_section` hands each mode's warnings to the line.
  An exit at the point itself has no line, and what it would say goes on
  the point (`Outcome.point_notes`). Nothing is sent to CalTopo, and
  an export failure is a warning, never a failed section. A candidate set
  goes through `caltopo_export` as draft assignments in a file of its own
  (`set_export_section`), and everything else found goes into a second
  file beside it (`write_markers`, which also says the export's top-level
  warnings). Drafts given a resource type the server's assignment corpus
  has no median for carry it as given, and the export section says so
  (`resource_type_warnings`): DOGS for DOG is a typo CalTopo would show.
  Where a scenario has an `export` section, every way it can draw
  nothing - the set's section left out or failed, `caltopo_export` raising
  (its `detail_call` kept to make again), a file that cannot be written,
  no ground to write beside - is a warning on that section, which stays
  `ok` with a null path; the top-level warnings then say either that
  nothing was found to draw or which file was not written although
  something found was for it - the drafts, the markers file or both, since
  one file in `exports` does not show the other was meant - never one for
  the other. A scenario's own file has a fixed name, so a call whose export
  writes none leaves an earlier call's where it was: the top-level warnings
  name it as an earlier call's (`earlier_export_warnings`), and it is not
  removed, since the earlier answer gave that path. Only that file is
  named: a call with `export` left out of `include` names nothing, and the
  drafts `caltopo_export` leaves under the same set id are not named, since
  it can raise after writing them and this call's own would then be called
  an earlier call's. `debrief_assignment` draws nothing new - the
  assignment and its tracks are already on the map - so its `exports` is
  always empty.
- **`summary`** holds the top-line facts a planner reads first - distances,
  minutes, counts, and a quickest exit's vehicle end (its road, access,
  gate on the road and lot, so a gated or private one is not read as
  open where the ground can tell; its nulls are the section's) - and
  never a recommendation or an order beyond the features' own. Every
  scenario builds it with `summarise` from its own `FACTS` mapping
  (section to facts helper): one block per section asked
  for, null when that section did not answer. A section named in
  `in_part` is read whenever it holds an answer in part as well
  (`answered_in_part`): `plan_medevac`'s egress, so a walk-out that came
  back while the carry-out raised is still summarised, with null for the
  mode that failed. The export has no block; its paths are in `exports`.
  A fact read off a fallback says so beside it, since the summary carries
  no warning: `medical_facts` gives the nearest trauma center's
  `ranked_by`, "straight_line" when it has no road time (the road router
  down, or no road found), where a null `road_minutes` alone would read as
  a road time not reported of a hospital still chosen by road.
- **Unknown is null, never zero.** A feature whose input was not staged
  answers an empty list and a warning. The section keeps the warning, but
  the summary keeps none beside its counts, where a 0 - or an axis missing
  from a count by axis - reads exactly as ground that was read and holds
  nothing. So each count the summary copies is null when its input was
  missing, in the section and the summary alike, decided by the helper
  that sees the feature's whole answer. Parking: `total_before_limit`,
  and the summary's `count`, when no lots could be read (no osm.json, or
  one older than parking: `_lots_unknown`). Hasty, on ground with no
  osm.json (`_unknown_axes`): each axis OpenStreetMap gives that was asked
  for; `line` when trail or road was asked for, `corridor` when waterway
  was the only corridor axis asked for; `total_before_limit` and the
  summary's `count` when no axis asked for could be read. A debrief kind
  whose detector declares the warning it gave for a missing input
  (`FeatureDetector.missing_input_warnings`, read by
  `terrain.features.input_missing`, so the scenario takes the detector's
  word rather than matching prose). A count that read some of its input -
  a debrief kind's, say - counts what it read, the
  nulls beside it saying what it could not; a count whose input was read
  stays a number, 0 included. A section field the summary does not copy
  can still read 0 for unknown: hasty's `existing_assignments` with no
  map read (PR #4 review B43).

## Known exceptions to the dependency rule

None. Since 0.27.1 infra imports nothing from domain:

- The cache's eviction (`cache/evict.py`) no longer works out which tiles
  staged ground rests on. `staging.tiles_in_use` does, and the
  `evict_cache` component hands the keys to `evict(..., protected=...)`.
- Finding an assignment by name (`find_assignment`) moved from
  `caltopo/session.py` to `caltopo/assignment.py`, beside the model it
  returns, so CalTopo transport (infra) and parsing (domain) are apart.

`tests/test_layering.py` forbids new ones: any import pointing up the
layers fails it, and so does importing an infra module in a fresh
interpreter if that loads a domain module (which also catches a package
`__init__` doing the importing). Its `KNOWN_EXCEPTIONS` set is kept, and
empty: a debt taken on deliberately goes there and in a table here saying
how it is to be paid off. Both checks honour it (the run-time one excuses
whatever a listed import loads in turn), and the test fails again once the
debt is fixed but still listed.

The test's `INFRA` set is the authoritative list of infra modules;
everything else outside the tool layers and the wiring (`server`,
`context`) is domain.

## Two axes, not one

Most of the tools answer **"sweep this ground looking for a person"**: hasty
candidates, corridors, coverage, sweep width, assignment norms.

`site_facts`, `boundary_clearance` and the `cliff_pitch` feature kind answer a
second question — **"can a team occupy and work this ground?"** That is the
rope-rescue question, and equally the helispot, landing zone, command post and
staging-area question. Three measurements it takes that the search tools do not:

- the **bench above** a face and the **landing below** it, separately, because a
  site with a good face and nowhere to stand is a different problem from the
  reverse;
- **vegetation on each footprint separately**, because a clean face above a
  scrub bench is not a usable site, and brush disqualifies ground however good
  the slope;
- **land tenure** — clearance to the boundary on eight bearings, and whether the
  straight line from a trailhead stays inside it.

Resolution is part of this axis too. `prepare_area(resolution_m=...)` exists
because at the default 1,024-pixel raster a 25 m rock face a couple of metres
wide is simply not there: sweeps of two preserves found no cliffs at 6-7 m per
pixel and real ones at 1 m.

None of these return a verdict. The thresholds that make a site feasible belong
to the agency running the operation.

## What is not here, and why

- **Writes to CalTopo.** The tier is designed; it needs its governance kit
  first — a staging folder so proposals are visually segregated from
  human-authored features, an ownership ledger so the hub can only edit what
  it created, dry-run, a map allowlist, and a separate write-capable
  credential. Shipping writes before that machinery would put an agent's
  mistakes onto a map live teams navigate by. `caltopo_export` is not a
  write: it saves a GeoJSON file beside the staged ground, and a person
  imports it into CalTopo, where every assignment arrives as a DRAFT.
- **Terrain for arbitrary geometry, except as a planner's area.**
  `prepare_area` stages a point-and-radius or polygon because that input
  cannot move, so it needs no revalidation; the map context it is compared
  against is still read fresh. An assignment-shaped polygon on no map is
  described against such an area with `describe_boundary`, never staged on
  its own.
- **Reading the debrief app's artefacts.** Those belong to the app that
  writes them. Teaching this package their filenames would point the
  dependency backwards, at the very moment the split was made to stop that.

## For planning agents

The planning tools measure; the agent judges. Hasty search around a place
last seen: `parse_coordinate(text, near)` on what the person said, never a
converted value; `prepare_area(center, radius_m, map_id)`;
`hasty_candidates(handle)` (trails and roads, the 50 nearest by default;
name `waterway` in `include` for creek corridors, and pass `max_distance_m`
and `limit` deliberately on a large area) and read the summaries, which are
ordered by distance and carry effort and overlap with existing assignments;
a named trail or road is one candidate drawn to its ends, so a trail that
loops or forks comes back whole, with `facts.branches` counting its pieces
and a warning that it is not a single walk end to end; those ends are the
edge of the staged ground (the area plus a 1,000 m planning buffer), not the
edge of the search circle, because a team is handed a trail rather than the
arc where the circle cut it - `facts.clipped_at_edge` is true, with a warning,
when the way carries on past the ground that was measured, while drainage,
ridge and waterway corridors are still clipped to the area itself;
`describe_boundary` or `get_candidate` for the shortlist only; then present
to a human with the description, priority and hazards the agent writes.

`parking_candidates(handle)` answers the other question a first operational
period asks - where does everyone park - from the same staged ground, at no
extra Overpass cost. It offers public parking lots and, where OpenStreetMap
has mapped no lot, the public hardstand around a school, church, park or
picnic site, flagged `parking_unverified`. **Nothing is hidden for being
awkward**: a private, permit-only, customers-only or fee-charging lot, and a
parking deck, all come back with a flag saying so, because a locked
commercial gate is a phone call a search manager can make and a lot they
never see is a lot they cannot make it about. Residential parking is the one
exception and never appears - a resident's parking space is somebody's home.
Each row carries a point guaranteed to lie inside its own polygon (the pole
of inaccessibility, not the centroid, which on an L-shaped lot lands in the
trees), so a marker put there is on asphalt. Note that OSM rarely NAMES a car
park - on real ground in a wooded preserve, 48 lots carried no name between them -
so each row also carries `nearest_way`: the named trail, track or road
nearest the LOT ITSELF (not its point), true ground metres, within 150 m -
or `None` when nothing named is that close. Nearest wins whatever the way's
class, so a trailhead lot beside a named trail is described by that trail
rather than a farther road; a fence, creek or lakeshore is never named. A row
without a `nearest_way` is still told apart by distance and bearing.

`lz_candidates(handle, center_latitude, center_longitude)` answers the
medevac question: where near this point can a helicopter put down. Stage
the area at `resolution_m=2.0` first - the default raster is 3.9 m per pixel,
the tool grades nothing coarser than 3 m, and it says so rather than
grading noise. On hub-staged ground the canopy panel comes through
CalTopo's WMS, as does the imagery chip; with no `wms_token` the call is
refused for want of the panel, rather than graded on slope alone. It grades
the way the CalTopo First Response safety course does (0 to 5 degrees, 7 if
desperate; canopy under 20 %; a 25 m circle at least, 45 by 60 m preferred),
each open pocket on its own ground, and reports the nearest staged power
line as a distance and its kind (`line` or `minor_line`), null for nothing
mapped that close, or `"unknown"` when OpenStreetMap could not be read, or
when the ground was staged before 0.16.0 labelled wires - re-stage with
`force=True` to label them. Null is never no wire.
`assumptions.powerline_kinds_checked` says which kinds were looked for:
`["line", "minor_line"]` on current ground, `["line"]` on ground staged at
query version 3 (before distribution lines were asked for, with a warning),
and `[]` when power lines are unknown. Four closures run before
any circle is fitted, not merely reported after: ground within 60 m of any
mapped power line (`powerline_clearance_m`, transmission and distribution
both) and within 30 m of a building on CalTopo's `structures` layer
(`structure_clearance_m`) is never offered as a touchdown; a structures
layer that cannot be had is a warning, `assumptions.structures` "not
checked", and grading still runs - the filter is lost, not the grade.
Trees are closed from true-colour NAIP through the shared `terrain.trees`
block - a cast shadow and the green crown beside it, so a wet meadow stays
open - because the CIR canopy proxy misses dark conifers; without the
imagery `assumptions.trees` is "not checked" and a warning says so.
Open water is closed too, since a lake is flat and treeless and would
otherwise grade as the best zone on the map: LANDFIRE open water from the
staged `evt.tif` and every OpenStreetMap lake, reservoir or coast shoreline,
a closed ring filled. `assumptions.water_sources_checked` names the sources
read; with neither, a warning says water was not checked. It
cuts a 200 m imagery chip per candidate for a human or a vision model to
check for the boulders and trees an elevation model cannot see - one per
source that could be had, in order: CalTopo imagery and NAIP through the
WMS token, then Mapbox and then Google static satellite images when the
server is configured with `ZA_MAPBOX_TOKEN` or `ZA_GOOGLE_MAPS_KEY`
(`chip_path` still names the first chip cut; `assumptions.chip.sources`
lists the sources asked, a refused one included - which chips exist is each
candidate's `chips` list) - and reports no aspect where
the ground has none, and no bearing to a clearing underfoot. No verdict: be
conservative, and the pilot decides.

Each landing zone also says which road it touches or sits beside, and which
lot it is in - **said, never used**: a road or a lot closes, flags and
regrades nothing. `road` is the nearest mapped road, track or other vehicle
way within 150 m of the touchdown (`powerline_notice_m`), with its name,
class (`highway`), surface and distance, and `on_road`. By the operator's
ruling, **"within the whole clearing"**: `on_road` is true when the way comes
within half `clear_diameter_m` of the touchdown, however wide the clearing -
a road 40 m off in a 195 m field is on that zone, and it needs traffic
control. Unbuilt roads, via ferratas and bus guideways are left out, as
egress leaves them out; a motorway is kept. `road` is null when none is that
close and `"unknown"` without OpenStreetMap; on ground staged before query
version 5 `highway` is null and `assumptions.road_class` is "unknown", and an
unbuilt road cannot be told from a built one there. `parking_lot` is
`{id, off}` when the touchdown lies inside a mapped lot's outline, `off`
naming the road it opens onto as `parking_candidates` does; null in no lot,
and `"unknown"` when the ground cannot tell (no OpenStreetMap, or staged
before parking was fetched).

`egress_routes(handle, center_latitude, center_longitude, mode)` answers the
question after the find: what is the way out on the ground. From a location -
where a subject was found, where a team is - it returns the quickest exit a
2WD vehicle can reach and the quickest a 4x4 can. Stage the area with
`prepare_area(..., resolution_m=10.0)` centred on the location first: walking
needs no finer ground, the search visits every cell, and ground staged too
fine (more than 1,800 x 1,800 cells, such as a 2 m landing-zone handle) is
refused with the re-stage named.

- **Both exits always come back**, 2WD first. One that cannot be reached
  says why: no such road mapped in the area, the litter grade limit, the
  location itself too steep to carry off, or ground cut off by a gap in the
  elevation model. A 2WD exit is a mapped road that is not dirt, not a
  motorway or pedestrian street, or a parking lot; a 4x4 exit is a track or
  a dirt road.
- **Time counts the climb.** "Quickest" is least walking time: Tobler's
  hiking speed on each step's signed grade, fastest on a gentle downhill, so
  a far exit along the contour can beat a near one over a ridge, and up and
  down cost differently. Mapped trails, tracks and roads are faster than
  open ground.
- **Carry-out or walk-out** (`mode`). `walk_out` is a team or an ambulatory
  subject, and walks the shared walking ground the rings and hasty minutes
  walk: factors 1.0 on a way and 0.6 off it times the LANDFIRE vegetation
  factor, 5 minutes to get into a creek, and rivers, shorelines, open water
  and ground over 45 degrees closed except at a mapped way across them (an
  exit's own cells are never closed). `carry_out` is a subject on a litter:
  0.25 and 0.08 (about 1.3 km/h on flat trail, 0.4 km/h off it), and no step
  steeper than 40 % either way, over the ways and motorways alone - it does
  not see vegetation or water. The route changes with the mode; the factors
  and the grade limit can be overridden per call.
- **Gates and access are said, never excluded.** A route through a gate, or
  an exit on a private road, is still the route: the planner can call for
  the combination or the ranger. Each exit names its road (with `access` and
  `gate_on_road`), its lot, and the gates it crosses.
- **A rank, not a promise.** It walks the elevation model, OpenStreetMap
  and (walk-out) LANDFIRE; it does not see fences, private land or trails
  OSM lacks, and carry-out does not see brush or water either. Read the minutes to compare two exits, not as an arrival time.
  The team leader decides.

**One movement model.** Every walking number lives in one module,
`sar_geo_mcp.terrain.movement`, and every tool that turns ground into time
reads it there: hasty candidates' `effort` and `travel_minutes_from_center`,
`travel_time_rings`, the debrief `terrain_metrics` traverse rate, and
`egress_routes` walk-out (which also walks the same ground; carry-out keeps
the litter's own factors). Tobler's hiking speed on the grade; on a
mapped way factor 1.0, off it 0.6 times a factor for the LANDFIRE vegetation
type (brush 0.4, timber 0.8, grass 0.9 ...); open water and ground over 45
degrees closed off a way; 5 minutes to get into a creek. Vegetation is read
by type everywhere; canopy density, where the debrief app has measured it,
narrows sweep width but never changes speed. A hasty candidate's minutes
from the centre come from the same walk the rings draw, so the two agree
cell for cell - at the default pace: rings drawn at a pace of their own
(`speed_factor` and the like) walk another, and `plan_initial_response`
warns when it shows the two side by side. Each answer states the model it
used under `movement`.

`travel_time_rings(handle, center_latitude, center_longitude, minutes)` is
the travel-time counterpart of a radius circle round a place last seen:
each ring is the ground a walker reaches within so many walking minutes
(15, 30 and 60 by default), so every point on its outline is - to the model -
the same time away. Stage with `prepare_area(..., resolution_m=10.0)` centred
on the location, as for egress. It is solved once on egress's walking graph,
outward from the location:

- **Terrain**: Tobler on each step's signed grade, so climbing out of a
  canyon costs more than walking down into it; ground off a way steeper than
  `cliff_slope_deg` (45 by default) is closed - on a 10 m model, a cliff.
- **Trails and vegetation**: on a mapped way the factor is 1.0; off one it
  is Tobler's 0.6 times a factor for the LANDFIRE lifeform there - brush 0.4,
  timber 0.8, grass 0.9 - overridable by name with `vegetation_factors`.
- **Water**: getting into a creek costs `creek_crossing_minutes` (5); a
  river, canal, lake or reservoir shore and open water are closed except
  within a cell of a mapped way across them - a bridge or ford. A creek in a
  culvert is no crossing.
- **The subject**: `speed_factor` scales every speed - 0.5 for someone who
  walks at half pace.

Each ring has its polygon (holes where ground inside it cannot be reached
in time), an `outline` to draw, its area and the radius of a circle of the
same area, and the farthest reachable ground in each of eight directions -
which, beside a circle, says where the ground stretches and where it is
pinched. For each direction it also traces the quickest route there and
splits it: minutes on trail, minutes off it, the trail it walked and the
point where it left - a subject who walks the trail and then strikes off
into rougher ground is exactly what draws the outline. A ring that runs into the edge of the staged ground says so and
names the radius that would hold it; the largest area reaches about 5.5 km,
roughly an hour on good trail. **The minutes are walking time, not time
since last seen**: a lost person stops, rests and backtracks, and turning
elapsed time into walking time - like any likelihood - is the planner's
call. Route records now keep `waterway` and `tunnel`, so a river can be told
from a creek; ground staged before 0.21.0 reads every waterway as a creek,
with a warning, until it is staged again with `force=True`.

On ground staged before query version 5, road class, access and gates are
unknown - null, never false - and a warning says so; no OpenStreetMap at all
is refused, since with no roads there is nothing to exit to.

`populated_areas(handle)` answers the urban-search question: where in this
area do people live, cut into the blocks a planner would draw by hand on
CalTopo's parcel layer. Stage the town with `prepare_area` (under about 8 km
across - lot lines are read at 1 m) and it returns one block per street:

- **A lot is read off the drawn lot lines**, built-up when CalTopo's
  structures layer puts at least 30 m2 of roof on it, and urban or suburban
  when it is no bigger than an acre (`max_parcel_area_m2`). Rural ground has
  big lots and returns no blocks, with a warning that it reads as rural - this
  tool is for town.
- **A block is a street's lots, both sides and the street between**, joined
  by a 12 m closing (`bridge_m`), with every other block's lots cut back out
  so no house is in two. A lot belongs to the named OpenStreetMap street it
  has the most edge along, so a corner lot can land on the street its address
  is not on. A street whose lots fall into separate runs gives one block per
  run, numbered, and every cross street ends a run: it is cut along each
  other named street's OpenStreetMap centreline (a lot is never cut, but
  goes whole to the piece holding most of it), so a block is a street both
  sides from cross street to cross street. A run the circle keeps is
  finished out to its cross streets - a corner lot or a row facing the
  cross street, filed under a block the circle does not reach, joins the
  run it touches; a run under `min_parcels` (3) houses joins the block whose
  lots it touches, and only one touching none is left out, counted in the
  summary. A vacant lot touching its street's houses is searched with them.
- **A big lot that is mostly roof is its own block** (`axis`
  "large_parcel"): an apartment complex, a school, a shopping centre - at
  least 10 % roof over an acre. A farmhouse on forty acres never qualifies.
  Condominium unit parcels are folded into the complex around them.
- **Summaries first**: blocks are saved as a candidate set (`kind` "block")
  and come back without geometry, nearest the centre first; `get_candidate`
  returns a block's polygon. It needs the WMS token and refuses without it.
- **House markers, only when asked for** (`building_markers=True`): each
  block then carries one CalTopo Marker per building, standing on the roof -
  CalTopo's house symbol (`hut`) in `marker_color`, white (`#FFFFFF`) by
  default - instead of any tracing of the buildings. They come back with the
  block from `get_candidate`; summaries carry only the count. A building is
  marked once, in the block holding most of its roof, and `facts.buildings`
  counts by the same rule, so the two agree.

- **One piece each, no shared ground** (0.20.0): blocks are made final
  nearest the centre first, each giving up any street another block already
  took, so no two outlines overlap. A run another block's lots cut in two
  comes back as two blocks, each with its own lots and a warning saying so.

A first cut, not a census: an undrawn house reads as a vacant lot and an
undrawn lot line as one lot. The planner adjusts the blocks and decides
which to search first.

`urban_assignments(handle, target_ha)` turns those blocks into assignments
of about one size - 10 ha by default, the corpus median for a ground team
(`resource_type`), or the `target_ha` you give. A type the corpus has no
median for - DOGS for DOG - is sized by the median of every assignment, and
`target_source` and a warning say so rather than name a median it lacks.
Each block is one street, both sides, back fence to back fence, and blocks
merge smallest first along the streets that join them - a cul-de-sac into
the road it opens onto - into the most compact shape, until a group reaches
60 % of the target and never past 140 %, so a team always has both sides of
its streets; blocks that only back onto each other merge only when a group
is left under 30 % of the target. Hand-drawn door-to-door assignments run 1
to 4 ha: pass `target_ha` (about 2.5) for teams that size. A group never
crosses an arterial (an OpenStreetMap motorway, trunk, primary or secondary
road), so the boulevards are the edges; small groups left over pair across
water, two at most - two isles a team drives between - drawn as one polygon
by a thin strip. Sizes follow the streets, so they spread round the target:
in a bayside suburb (0.33.0, before street-joined merging), half a mile
round a residential street at 11.7 ha, 65 blocks became
14 assignments of 3.9 to 16.2 ha, median 11.0, and the smallest is a pocket
boxed in by two boulevards. Pass `max_distance_m` (the radius) to keep the
blocks of the search circle: every block the circle reaches is kept whole,
out to the back fences beyond the nearest street - and out to the end of
its street: lot lines are read 200 m past the area, and again twice as far
while a kept block runs into the edge of what was read, up to the 1,000 m
planning buffer (`assumptions.margin_m`). Edges snap to the streets, as on
a hand-drawn CalTopo plan (`snap_outlines`): OpenStreetMap centrelines are
barriers no ground is given across, so two assignments either side of a
street meet on its centreline; an edge facing a street within 30 m is drawn
out to it, straight ahead of the edge (no finger runs along the street past
the run), as is a notch opening onto one; a pocket or slit most of whose
surroundings the assignments already cover goes to the nearest one in
sight, and so does the rest of a city block they mostly cover; centrelines
are run on 15 m past their ends, closing the gap OpenStreetMap leaves at a
T junction; a boulevard's median goes with the ground beside it; and ground between two assignments up to 90 m apart
(`snap_gap_m`) with no street through it - a fence-line strip - is split
down its middle. No gap, no overlap, and every edge simplified as one
coverage, so a shared edge is one line. An outer back fence facing no other
assignment stays where it is. `area_ha` includes the street ground taken.

`caltopo_export(handle, set_id)` writes any candidate set - urban
assignments, populated blocks, hasty lines - as a GeoJSON file CalTopo
imports as DRAFT assignments, labelled `U01 Main Street +2` and onward,
with the search area as a Shape and its centre as a Marker, and the house
markers when the set carries them. CalTopo holds one ring per polygon and
one polygon per assignment, so a multi-part candidate becomes `U05a` and
`U05b` and a hole is cut open. The file is all it does: a person imports it,
checks every assignment and issues it.

`prepare_area` takes the radius in the unit the planner said it in -
`radius=0.5, radius_unit="mi"` - and half a mile and 804.672 m are one
handle; a unit other than metres beside `radius_m`, which is metres by
name, is refused rather than dropped. Its result's `area` is the circle to draw.
`convert_coordinate` gives degrees-minutes-seconds too
(`37°30'00.0"N 122°15'00.0"W`), and every form it gives parses back.

**For parking, ground staged by an older release used to hold no parking at all.** Query
version 3 is what fetches it, so a directory staged before 0.11.0 had a
version-2 `osm.json` with none in it. `prepare_area` now notices this itself:
reusing such a directory re-runs the OSM half of staging - one Overpass fetch
per tile, never the DEM or vegetation, which no query change touches - and
the returned `warnings` say so, so a planner never has to ask for
`force=True` just to get parking read. Only the OSM half was ever behind;
`force=True` remains what re-stages everything else, and is still what a
planner reaches for when the DEM or vegetation itself needs refreshing.

**0.17.0 re-fetches OpenStreetMap for every staged area.** Query version 4
asks for distribution power lines (`power=minor_line`), so every
`osm.json` written before it - version 3 and older - is topped up the next
time `prepare_area` reuses its directory: one Overpass fetch per tile, as
above, with a warning saying what the old file lacked (parking below
version 3, only distribution lines at version 3). The top-up rewrites the
directory's manifest, so every candidate set saved on that ground before it
reports stale from then on and should be generated again.

**0.17.1 stops an old outage repeating.** Ground staged while Overpass was
down recorded "osm unavailable: ..." and `prepare_area` returned it on every
reuse, even after a top-up had fetched the file. Once a top-up has written
an `osm.json` that warning is no longer returned - a file that is merely
behind the current query is reported by the top-up's own warnings - and a
successful top-up moves it to the manifest's `osm_retired_warnings`; every
other warning still is returned. In an assignment directory, a file no
top-up wrote, beside an outage, is a leftover from an older hub (the
boundary may have moved) and the outage stands; an area directory has one
bbox for life, so its file is always its own ground and the outage goes. An existing file
always belongs to the ground staged: a re-stage that cannot reach Overpass
keeps the earlier file only for the same box, and says so in that
staging's response ("the OSM refetch failed (...); the osm.json staged
earlier for this same ground is kept (query version N)"), and removes one
fetched for another box, such as a boundary that moved. An `osm.json` that
cannot be parsed is fetched again rather than failing the call, and removed
when it cannot be.

No warning names a local path, a URL or a credential. A failed OSM fetch
names the exception type, and an HTTP error's status, never its message.
The one sentence kept whole is `OverpassUnavailable`'s, which the hub writes
itself and which names only the last host tried. Overpass endpoints are
reported by host and port alone, with any `user:password@`, path, query or
fragment from `ZA_OVERPASS_URLS` removed; that includes the
`overpass_endpoints` list in every routes result's assumptions, which now
holds hosts rather than URLs.

**0.18.0 re-fetches OpenStreetMap for every staged area once more.** Query
version 5 keeps road class, access and tracktype on every route and fetches
gates, for `egress_routes` and the landing-zone road fact. Every `osm.json`
written before it is topped up the next time `prepare_area` reuses its
directory, one Overpass fetch per tile, with a warning naming every group
the old file lacked - parking below version 3, distribution lines below 4,
road class, access and gates below 5. As before, the top-up makes every
candidate set saved on that ground report stale.

`nearest_medical_facilities(latitude, longitude)` answers "where does a
patient go from here, and how long does it take?" It needs no staged ground.
For each of three categories - any acute-care hospital, an adult trauma
center, a pediatric trauma center - it returns the closest three by ROAD time
(`per_category`), and a Markdown `table` ready to show: trauma level, helipad
Y / N / unknown, road distance and time, air time, phone and address.

- **Hospitals** come from the HIFLD Hospitals list (open hospitals only;
  psychiatric, rehabilitation and long-term care left out). The same campus
  listed twice is shown once. A children's hospital is never offered as an
  adult trauma center, even where its record claims one (HIFLD lists Lucile
  Packard Children's as adult Level I; that designation is Stanford Health
  Care's). A category with nothing within `search_radius_km` (100 by
  default) is searched again out to 400 km.
- **Road** distance and time come from OSRM over OpenStreetMap roads:
  free-flow car time, from where the point joins the road network.
  `road_start` says where that is, and a warning says so when it is more
  than 200 m from the point - the walk or carry to it is not counted.
- **Helipad** is HIFLD's flag, cross-checked against OpenStreetMap helipads
  within 300 m; either saying yes is a Y.
- **Air time** is straight-line distance at 240 km/h plus 5 minutes for
  lift-off and landing, and only where there is a helipad. It is the
  transport leg alone - not dispatch, the flight in, loading, weather or
  night.
- **US only.** The hospital list has no other country's hospitals, so the
  point is checked against US Census state boundaries: `point.in_us` is
  false, and the first warning says so, for a point in Canada, Mexico or
  offshore - a closer hospital across the border would not be here.
- **`markers`** import into CalTopo as Markers: a red first-aid cross per
  hospital, its description holding the table row (categories and ranks,
  trauma level, helipad, road and air time, phone, address), and a blue dot
  for the point. Nothing is written to CalTopo.

It is not a destination decision: designations are as HIFLD last validated
them (each row's `validated`), and emergency-department status and diversion
are unknown. A failed road or helipad lookup is a warning; a failed hospital
lookup is an error.

`assignment_norms(resource_type)` gives the defaults and their provenance.
What the agent must not delegate: the radius and region, the resource and
team hours, the wording, the priority, any likelihood reasoning - the server
has no opinion on any of these and returns none. Summaries carry ids, not
geometry; fetch geometry only for what will be shown or issued. Area handles
are never stale; anything compared against the map carries `map_provenance`.

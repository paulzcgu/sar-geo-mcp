# sar-geo-mcp

**Search and rescue planning tools for Claude, built on CalTopo maps.**

SAR Geo is a ready-made set of tools, features and scenarios that a
language model can call directly. Without it, a model asked to plan a
search has to reinvent the work every time - writing its own coordinate
parsing, slope arithmetic or route search from scratch, and getting a
slightly different answer each time. With it, the model calls one tested
tool and gets the same measured answer for the same question. That makes
results stable, and makes them adjustable: change a tool's parameters,
not the model's improvised code. It is also a plain Python package, so
any Python program can import the same functions as a library, with no
MCP server and no language model involved.

SAR Geo gives Claude forty-seven tools for planning a search: read a
CalTopo map, stage the terrain around a place last seen, and ask for hasty
search lines, parking, helicopter landing zones, the way out, walking-time
rings, the nearest hospitals and trauma centers, and urban assignments of
one size. Four scenarios run several of those in one call: a medevac, the
initial response, an urban search and an assignment debrief. Results come
back as facts with GeoJSON files CalTopo imports as draft assignments.

It is **read-only**: it never writes to a CalTopo map. The tools measure;
the people running the search decide.

## Install

### Claude Desktop (no coding needed)

1. Download `sar-geo.mcpb` from the
   [latest release](https://github.com/paulzcgu/sar-geo-mcp/releases/latest)
   (direct link:
   [sar-geo.mcpb](https://github.com/paulzcgu/sar-geo-mcp/releases/latest/download/sar-geo.mcpb)).
2. Open it, and click **Install** in Claude Desktop (macOS or Windows).
   Claude warns that the extension is not verified by Anthropic; that is
   expected for anything not in Anthropic's directory.
3. Fill in any keys you have (all optional; see *Keys* below) and save.

The first start downloads the server from PyPI, which takes a minute or
so; later starts reuse that copy, and offline it runs the copy it already
has. It updates itself: each time Claude Desktop starts it, it runs the
newest release, so you install the `.mcpb` once. Ask Claude "check for a
sar-geo update" to see your version, or set **Pin version** to stay on
one, such as during an operation.

### Any other MCP client (Claude Code, and others)

The package is on [PyPI](https://pypi.org/project/sar-geo-mcp/) and needs
Python 3.12 or newer. The `mcp` extra adds the server; without it you get
the library alone. With [uv](https://docs.astral.sh/uv/), nothing needs
installing first:

```bash
claude mcp add sar-geo -- uvx --python 3.12 --from "sar-geo-mcp[mcp]" sar-geo-mcp
```

Or install it with pip and register the command it puts on your `PATH`:

```bash
pip install "sar-geo-mcp[mcp]"
```

```bash
claude mcp add sar-geo -- sar-geo-mcp
```

Other clients take the same command (`sar-geo-mcp`) as a stdio server.

### From source

```bash
git clone https://github.com/paulzcgu/sar-geo-mcp.git
```

```bash
cd sar-geo-mcp && pip install -e ".[mcp]"
```

[AGENTS.md](AGENTS.md) covers the architecture, tests and releases.

## Use it from Python

Every tool is also a plain Python function, and the modules behind them
can be imported on their own. A script, a notebook or another program gets
exactly what an MCP caller gets:

```python
from sar_geo_mcp.coords import parse_location

parse_location("10S EG 61234 41234")          # (37.4158..., -122.3080...)

# A dropped western minus is put back only against a reference point:
parse_location("37.3, 122.32", near=(37.3045, -122.32))   # (37.3, -122.32)
```

A tool function takes a `ServerContext` first, then the tool's own
arguments:

```python
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.context import ServerContext
from sar_geo_mcp.components.coords import distance_bearing

ctx = ServerContext.build(CaltopoConfig.from_env())
distance_bearing(ctx, 37.3, -122.32, 37.3045, -122.32)
# about 499 metres, bearing 0 (due north)
```

Install with `pip install sar-geo-mcp` (no `[mcp]` needed). Features live
in `sar_geo_mcp.features`, scenarios in `sar_geo_mcp.scenarios`;
[AGENTS.md](AGENTS.md) lists which module holds each tool.

## Keys

Every key is optional. Without any, public CalTopo maps, terrain,
OpenStreetMap, parking, hasty lines, egress and hospitals all work. Ask
Claude "what's my sar-geo key status?" to see which are set and what each
unlocks.

| Key | Turns on | Where to get it |
| --- | --- | --- |
| CalTopo WMS token | Terrain pictures, slope shading, the canopy panel, landing-zone imagery, urban blocks | CalTopo → your name → Your Account → API Access: the six characters between `/api/` and `/wms` (paid plan or team) |
| CalTopo credential ID + key | Private and team maps | CalTopo → Your Account → Credentials (a read-only team credential is enough) |
| Mapbox token | Mapbox satellite pictures of landing zones | account.mapbox.com |
| Google Maps key | Google satellite pictures of landing zones | Google Cloud console, Maps Static API (its terms limit how images may be used) |

From source, set them as `ZA_WMS_TOKEN`, `ZA_CALTOPO_ID` and
`ZA_CALTOPO_KEY` (or a `cts.ini`), `ZA_MAPBOX_TOKEN` and
`ZA_GOOGLE_MAPS_KEY`. The CalTopo account ID field in the extension is
only for printing PDFs through CalTopo; no tool needs it today.

## Where files go

Ask Claude to save a file (an export, an imagery chip) and it goes in
`Documents/sar_geo/<chat folder>`, one folder per chat, and opens in Finder
or Explorer. Downloaded terrain is cached out of the way, in
`~/Library/Caches/sar_geo` on a Mac and `%LOCALAPPDATA%\sar_geo\cache` on
Windows (`~/.cache/sar_geo` elsewhere), and is downloaded again if
deleted. It can grow to a few GB.

Both folders can be changed: in the extension's settings (**Exports
folder**, **Cache folder**), or from source with `SAR_GEO_EXPORTS_DIR` and
`ZA_CACHE_DIR`.

## The tools

Each tool carries its layer as an MCP tag - `component`, `feature` or
`scenario` - so a client can list one layer; [AGENTS.md](AGENTS.md) says
which tool is in which. The four scenarios, last in the table, call
several features at once and answer in compact sections, each with its
own status and the exact feature call that returns the full answer.

| Tool | What it answers |
| --- | --- |
| `list_assignments` | What assignments are on this map, and where |
| `get_assignment` | One assignment in full — geometry, description, centre |
| `list_map_features` | Everything on the map, filterable by class |
| `parse_coordinate` | Read a coordinate a person wrote or said |
| `convert_coordinate` | The same point in DD, DDM, DMS, UTM, and MGRS/USNG |
| `distance_bearing` | Geodesic ground distance and initial bearing |
| `point_in_geometry` | Is this clue inside the assignment, and how far from the edge |
| `geometry_bounds` | Bounding box and centre of any GeoJSON geometry |
| `boundary_clearance` | How much of a boundary's inside surrounds this, and whether the way in stays inside it |
| `buffer_geometry` | A line widened into the ribbon CalTopo's Buffer draws, so much each side; `single_ring` cuts a loop's hole open so CalTopo keeps it |
| `prefetch_map` | Download a map's ground now, before losing connectivity |
| `cache_status` | What the tile cache holds, by tier |
| `evict_cache` | Trim the cache to a ceiling, oldest maps first |
| `save_export` | Copy a file a tool wrote (an export GeoJSON, a chip) to `~/Documents/sar_geo/<chat folder>` and show it in Finder or Explorer |
| `key_status` | Which optional keys are set (never their values), what each unlocks and where to get it |
| `check_for_update` | The version running against the newest on PyPI, and how to update |
| `list_data_sources` | Every external data source the server reads - what it is, its coverage and attribution, whether its key is set (never the key), and which tool pulls it |
| `fetch_raster` | One raster source's own image of an area, saved as a file with a short summary: elevation, vegetation type, any CalTopo map layer, NAIP imagery or Mapbox satellite. Raw data, without the planning tools' checks |
| `query_features` | One feature source's own features over an area as GeoJSON: OpenStreetMap ways and nodes filtered by tag, the land agencies' road and trail records, or hospitals. Raw data, without the planning tools' checks |
| `prepare_terrain` | Stage one assignment's ground; returns the handle everything below names |
| `terrain_metrics` | Everything measured for a staged assignment — the debrief bundle |
| `identify_features` | Just the feature kinds you name — steep bands, drainages, clearings, cliff bands, `cliff_pitch` (ranked by the drop you would rig on), routes, ingress |
| `list_feature_kinds` | Which feature kinds `identify_features` can look for, with their assumptions |
| `slope_stats` | Slope statistics inside one named shape — "quantify that sidewall" |
| `slope_shading` | CalTopo's slope bands beside this package's own, and the patches where the DEM reads shallower |
| `shaded_relief` | Terrain to look at, and benches OpenStreetMap has not got - clean and OSM-overlaid, plus measured lineaments |
| `unmarked_paths` | Roads and trails OpenStreetMap has not got, from Forest Service, BLM and USGS inventories - closed Forest Service roads included - each checked against NAIP imagery for a corridor still bare on the ground, ranked by how many independent inventories saw it, junction gaps bridged along flat ground, and written to a GeoJSON file CalTopo imports as coloured lines. DEM benches are an experimental source, read only when asked for |
| `elevation_profile` | Elevation along a drainage centreline — how steep is the channel |
| `coverage_for` | Coverage and gaps for a staged assignment, from named GPS tracks |
| `prepare_area` | Stage the ground around a place last seen (point + radius, in m, km, ft, yd or mi) or a polygon; returns an `aoi-` handle that never goes stale, and the area's outline to draw |
| `assignment_norms` | What real assignments look like, from a corpus of past searches, with provenance; per-resource target sizes |
| `site_facts` | Can a team occupy and work this ground - drop, span, the bench above, the landing below, and cover on all three |
| `describe_boundary` | What an assignment's or drawn shape's edges follow, how it was drawn, what it holds, against corpus norms |
| `hasty_candidates` | Day-1 hasty candidates around a staged area: trails and roads as lines by default, mapped waterways as corridors when `include` names them (no drainages or ridges), measured and ordered by distance |
| `parking_candidates` | Where a search can park near a staged area: public lots and public hardstand, nearest first, each with a point guaranteed to sit inside it and, since OSM rarely names a car park, the nearest named trail or road within 150 m |
| `lz_candidates` | Where a helicopter can put down near a staged area: open, flat patches graded preferred, minimal or marginal per the CalTopo safety course, with a touchdown point, the nearest power line or "unknown", the road it touches or sits beside, the lot it is in, and an imagery chip to look at |
| `egress_routes` | The way out on the ground from a location: the quickest exit a 2WD vehicle can reach and the quickest a 4x4 can, on foot or by litter, each with its route, minutes, distance, climb and descent, and the road, lot and gates it ends on |
| `travel_time_rings` | Rings of equal walking time round a place last seen - the travel-time counterpart of a radius circle - stretched along trails, pinched by cliffs and unbridged rivers, slowed by creek crossings and brush |
| `nearest_medical_facilities` | The closest medical center, trauma center and pediatric trauma center to a point, ranked by road time, as a table: trauma level, helipad Y/N, road distance and time, air time where there is a helipad, phone and address - plus markers to draw. Stateless |
| `populated_areas` | Urban and suburban search blocks: the built-up lots along each street, both sides and the street between them, outlined as one block per street from CalTopo's parcel and structures layers; apartment complexes, schools and shopping centres as blocks of their own |
| `urban_assignments` | Those blocks merged into assignments of about one size (the corpus median for the resource, or a target you give), never across an arterial |
| `caltopo_export` | Any candidate set as a GeoJSON file CalTopo imports as DRAFT assignments, with the search area and house markers - a file, never a write to CalTopo |
| `get_candidate` | One candidate in full, geometry included, by id |
| `plan_medevac` | A patient at a point, in one call: the landing zones nearest them, the 2WD and 4x4 exits by litter and on foot, and the nearest hospitals and trauma centers - compact sections, each with its own status and the exact call for the full answer, plus one GeoJSON file CalTopo imports. Facts, no verdict |
| `plan_initial_response` | The first operational period from a place last seen, in one call: the hasty lines a team can run (trails and roads by default; waterways through `overrides`), the nearest parking, walking-time rings, the landing zones nearest the place and the nearest hospitals - compact sections, each with its own status and the exact call for the full answer, plus the hasty lines as DRAFT CalTopo assignments and a second GeoJSON file of everything else. Facts, no verdict |
| `plan_urban_search` | An urban or suburban search round a place last seen, in one call: the populated blocks grouped into assignments of about one size, the nearest parking and the nearest hospitals - compact sections, each with its own status and the exact call for the full answer, plus the assignments as DRAFT CalTopo assignments and a second GeoJSON file of parking and hospitals. Needs the CalTopo WMS token for the blocks. Facts, no verdict |
| `debrief_assignment` | A team back from a CalTopo assignment, in one call: what the ground was (slope, landform, drainages, vegetation, sweep width, ways in), how much of it the team's GPS tracks covered and where the gaps are - each gap's size, the terrain it leaves unsearched and a point inside it - and the terrain features worth asking about - compact sections, each with its own status and the exact call for the full answer, and the boundary check said once. Facts, no verdict |

`parse_coordinate` handles decimal degrees, degrees-decimal-minutes, DMS,
UTM, and MGRS/USNG, and it **refuses to guess**: a transposed pair or a
dropped western minus is resolved only against a reference point you supply,
and a grid reference that decodes far from that point comes back unparsed
rather than plausible. A wrong coordinate is worse than no coordinate.

## Documentation

Each tool describes itself to Claude, so asking Claude how to use one is
the quickest guide. [AGENTS.md](AGENTS.md) has the architecture and how
to change, test and release the code.

## Data

Elevation (USGS 3DEP), vegetation type (LANDFIRE), roads and trails
(OpenStreetMap, Forest Service, BLM, USGS), hospitals (HIFLD), road routing
(OSRM), state boundaries (Census) and NAIP imagery are public. CalTopo's
map layers need the WMS token above; Mapbox and Google satellite images
need their own keys. Ask Claude to "list the sar-geo data sources" for
each one's coverage and attribution. Coverage is the United States.

## Origin

Extracted from a private SAR debrief-automation project, which is now its
first consumer.

## License

MIT - see [LICENSE](LICENSE). It comes with no warranty: SAR Geo measures
ground and reports facts, and every decision in a search stays with the
people running it.

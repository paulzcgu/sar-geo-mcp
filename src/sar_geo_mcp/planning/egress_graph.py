"""Egress's walking graph: the least walking time out to a road, on foot or by
litter.

WHAT THIS IS
------------
A DIRECTED least-cost surface over the staged elevation model. A step from
cell a to its neighbour b costs its ground length over a speed: Tobler's
hiking speed on the step's SIGNED grade (fastest on a gentle downhill,
slower climbing, slower descending steeply), times a ground factor for what
b is - on a mapped trail, track or road, or off one. Carry-out adds a grade
limit: a litter does not take a step steeper than it either way.

The litter's limit applies to the GROUND'S SLOPE, not just the step's
direction: a cell off any mapped way whose steepest gradient is over the
limit is closed to a litter, so a face too steep to carry up cannot be
climbed by zigzagging on diagonals that each come in under it. A cell ON a
mapped way is exempt from the ground's slope - a benched trail across a
steep sidehill reads steep on the 10 m DEM while its tread is flat, and a
litter uses trails exactly there. A step ALONG a mapped way (both ends on
it) is exempt from the per-step limit too: a pitch of trail steeper than
the litter's limit is one the team belays the litter down or hauls it up,
never one it leaves the trail to walk round over open ground (ruled
2026-10-24, after a test carry-out left the trail for 150 m round
such a pitch). Egress says the metres of such pitches on each exit. Off a
way the per-step limit stands, so open ground too steep for a litter is
never climbed by zigzagging.

Two kinds of blocked cell. A DEM void (no elevation) has no step in or out.
A CLOSED cell (a motorway; for a litter, ground over its grade limit) has no
step INTO it but does have steps out, so a subject found on a freeway
shoulder can still walk off it. A diagonal step needs at least one of the
two cells at its corners to be one it could step into, so it never squeezes
between two blocked cells - a 45-degree line burned with all_touched is
joined only at its corners. A diagonal step ALONG a mapped way (both ends on
it) needs only one corner of hard ground - measured, not a motorway - so a
litter can follow a 45-degree trail across a face whose corner cells are
closed to it for their slope alone.

WHAT THIS IS NOT
----------------
A surveyed route. It walks the elevation model and whatever ground its
caller marks - ways and motorways always; vegetation, creeks and closed
water when the caller passes them (`planning.walking` does) - and never
fences, private land or trails OSM lacks. Read the minutes as a rank.
"""

from __future__ import annotations

import math

import numpy as np

from ..terrain.movement import OFF_WAY_FACTOR, ON_WAY_FACTOR, signed_tobler_kmh
from ..terrain.slope import slope_gradient

#: Mode defaults (spec 3.1). Walk-out walks as everyone else walks - its
#: factors are the shared movement model's (`terrain.movement`). Carry-out's
#: are its own, from litter-carry rules of thumb: ~1.3 km/h on flat trail,
#: ~0.4 km/h off it. Tunable per call.
MODES: dict[str, dict[str, float | None]] = {
    "walk_out": {"on_way_factor": ON_WAY_FACTOR, "off_way_factor": OFF_WAY_FACTOR,
                 "max_grade": None},
    "carry_out": {"on_way_factor": 0.25, "off_way_factor": 0.08, "max_grade": 0.40},
}
#: Once a route has joined a mapped way, a step off every way counts this
#: many times its minutes over (1 / factor) in choosing the route: 10x, so
#: a team leaves a trail only where the trail's way round is ten times
#: longer than the ground cut across. The reported minutes are walking
#: minutes, never counted over. The same in both modes; tunable per call.
LEAVE_WAY_FACTOR = 0.1

#: All eight neighbours; the graph is directed, so each direction is its own edge.
_STEPS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


def build_graph(elevation: np.ndarray, *, cell_m: float, on_way: np.ndarray,
                impassable: np.ndarray, on_way_factor: float, off_way_factor: float,
                max_grade: float | None, off_way_scale: np.ndarray | None = None,
                crossing: np.ndarray | None = None, crossing_minutes: float = 0.0):
    """The directed step graph as a CSR matrix of minutes, cells indexed
    row-major.

    - A cell with no elevation (NaN) has no edge in or out.
    - A cell marked `impassable` (a motorway) is CLOSED: no edge into it,
      but edges out of it, so an origin on one can leave.
    - With `max_grade`, a cell off `on_way` whose ground slope is over it is
      closed the same way (a cell on a way is exempt from its slope), and
      any single step steeper than it either way has no edge - unless both
      its ends are on a way: a steep pitch of trail is belayed, not left.
    - A diagonal step needs one of its two corner cells open to entry; a
      diagonal step from a way cell to a way cell needs only one corner of
      hard ground (finite, not `impassable`), whatever its slope.
    - A step too steep for Tobler to give a speed (it underflows to zero)
      has no edge, rather than an infinite one.
    - `off_way_scale`, a per-cell factor (vegetation, for the travel-time
      rings), multiplies `off_way_factor` on cells off a way; a way's cells
      keep `on_way_factor`, since a trail cuts through the brush beside it.
    - `crossing` marks cells that cost `crossing_minutes` more to get INTO
      from a cell outside them (a creek): stepping in from the bank, or a
      diagonal step between two crossing cells at its corners, which is how
      a 45-degree line burned all_touched is joined. Walking along inside
      the mask, or stepping out of it, costs nothing extra.

    Egress passes neither, and its graph is unchanged.
    """
    from scipy.sparse import coo_matrix

    elevation = np.asarray(elevation, dtype="float64")
    height, width = elevation.shape
    size = height * width
    finite = np.isfinite(elevation)
    on_way = np.asarray(on_way, dtype=bool)
    hard = finite & ~np.asarray(impassable, dtype=bool)
    crossing = (np.asarray(crossing, dtype=bool) if crossing is not None and crossing_minutes
                else None)
    enterable = hard.copy()
    if max_grade is not None:
        # A NaN slope (beside a void) compares False: unknown ground is not
        # closed on its slope, and the per-step limit still applies to it.
        # A way's cells are exempt from the slope (the DEM does not see a
        # bench), never from the per-step limit below; `&=` keeps a void or
        # a motorway closed whatever it is.
        with np.errstate(invalid="ignore"):
            enterable &= ~(slope_gradient(elevation, cell_m) > max_grade) | on_way
    off_factor = off_way_factor
    if off_way_scale is not None:
        off_factor = off_way_factor * np.asarray(off_way_scale, dtype="float64")
    factor = np.where(on_way, on_way_factor, off_factor)
    index = np.arange(size, dtype=np.int32 if size < 2**31 else np.int64)
    index = index.reshape(height, width)
    starts, ends, minutes = [], [], []
    for drow, dcol in _STEPS:
        row0, row1 = max(0, -drow), height - max(0, drow)
        col0, col1 = max(0, -dcol), width - max(0, dcol)
        if row1 <= row0 or col1 <= col0:
            continue
        here = (slice(row0, row1), slice(col0, col1))
        there = (slice(row0 + drow, row1 + drow), slice(col0 + dcol, col1 + dcol))
        ok = finite[here] & enterable[there]
        if drow and dcol:
            corner_a = (slice(row0 + drow, row1 + drow), slice(col0, col1))
            corner_b = (slice(row0, row1), slice(col0 + dcol, col1 + dcol))
            open_corner = enterable[corner_a] | enterable[corner_b]
            open_corner |= on_way[here] & on_way[there] & (hard[corner_a] | hard[corner_b])
            ok &= open_corner
        if not ok.any():
            continue
        step_m = cell_m * (math.sqrt(2.0) if drow and dcol else 1.0)
        grade = (elevation[there][ok] - elevation[here][ok]) / step_m
        with np.errstate(divide="ignore"):
            step_minutes = step_m * 0.06 / (signed_tobler_kmh(grade) * factor[there][ok])
        if crossing is not None:
            enters = crossing[there] & ~crossing[here]
            if drow and dcol:
                enters |= (crossing[corner_a] & crossing[corner_b]
                           & ~crossing[here] & ~crossing[there])
            step_minutes = step_minutes + np.where(enters[ok], crossing_minutes, 0.0)
        keep = np.isfinite(step_minutes)                # (m / 1000) / kmh * 60
        if max_grade is not None:
            # A step along a mapped way (both ends on it) is exempt: a team
            # belays a litter down a steep pitch of trail rather than leave
            # the trail for open ground. Such a pitch is said on the exit.
            along = (on_way[here] & on_way[there])[ok]
            keep &= (np.abs(grade) <= max_grade) | along
        starts.append(index[here][ok][keep])
        ends.append(index[there][ok][keep])
        minutes.append(step_minutes[keep])
    if not starts:
        return coo_matrix((size, size)).tocsr()
    # Concatenate one list at a time and drop it, so the per-direction pieces
    # and the joined arrays are never all alive at once (the 1,800-cell
    # maximum area has ~26 million edges).
    rows = np.concatenate(starts)
    del starts
    cols = np.concatenate(ends)
    del ends
    data = np.concatenate(minutes)
    del minutes
    graph = coo_matrix((data, (rows, cols)), shape=(size, size))
    del rows, cols, data
    return graph.tocsr()


def solve_from(graph, shape: tuple[int, int], origin_cell: tuple[int, int], *,
               void: np.ndarray):
    """Minutes to every cell from `origin_cell` (row, col), and the
    predecessor of each cell on its quickest path (-9999 where none).

    `void` is the ground's no-elevation mask (np.isnan(elevation)). It is
    required: from the graph alone an origin on a void cannot be told from
    one walled in, and a void origin has no measured ground to start from,
    so every cell - itself included - is infinite, as in
    travel.travel_surface. An origin off the grid is refused, not wrapped.
    """
    from scipy.sparse.csgraph import dijkstra

    height, width = shape
    void = np.asarray(void, dtype=bool)
    if void.shape != (height, width):
        raise ValueError(f"void mask is {void.shape}, not the grid's {(height, width)}")
    row, col = int(origin_cell[0]), int(origin_cell[1])
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(f"origin cell {(row, col)} is outside the {height} x {width} grid")
    if void[row, col]:
        return (np.full((height, width), np.inf),
                np.full(height * width, -9999, dtype=np.int32))
    minutes, predecessors = dijkstra(graph, directed=True, indices=row * width + col,
                                     return_predecessors=True)
    return np.asarray(minutes, dtype="float64").reshape(height, width), predecessors


class WaySolve:
    """A solve that stays on the ways once it has joined one
    (`solve_staying_on_ways`). `cost` is the minutes the search ranked by,
    each step off a way after the first way counted `1 / leave_way_factor`
    times over; `trace` gives a cell's path from the origin and
    `walk_minutes` that path's plain walking minutes, uncounted over."""

    def __init__(self, *, cost, before, before_predecessors, after_predecessors, joined,
                 graph, penalised, leave_way_factor, origin_index):
        self.cost = cost
        self._before = before.ravel()
        self._before_predecessors = before_predecessors
        self._after_predecessors = after_predecessors
        self._joined = joined
        self._graph = graph
        self._penalised = penalised
        self._factor = leave_way_factor
        self._origin = origin_index

    def _trace_before(self, cell: int) -> list[int]:
        path = [int(cell)]
        while path[-1] != self._origin:
            previous = int(self._before_predecessors[path[-1]])
            if previous < 0 or len(path) > len(self._before_predecessors):
                raise RuntimeError("the predecessors do not lead back to the location")
            path.append(previous)
        path.reverse()
        return path

    def trace(self, cell: int) -> list[int]:
        """The flat cell indices from the origin to `cell`, in walking order."""
        cell = int(cell)
        if not self._joined[cell]:
            return self._trace_before(cell)
        seed = len(self._joined)                    # the added source node
        after = [cell]
        while True:
            previous = int(self._after_predecessors[after[-1]])
            if previous == seed:
                break
            if previous < 0 or len(after) > seed:
                raise RuntimeError("the predecessors do not lead back to a way")
            after.append(previous)
        after.reverse()
        return self._trace_before(after[0]) + after[1:]

    def walk_minutes(self, path: list[int]) -> float:
        """The plain walking minutes along a path `trace` returned."""
        if not self._joined[path[-1]]:
            return float(self._before[path[-1]])
        # The path's first way cell is where the search joined the ways:
        # the minutes to it are the first stage's, and every step after is
        # priced again without the leave-way penalty.
        start = next(i for i, cell in enumerate(path) if self._joined_seed(cell))
        minutes = float(self._before[path[start]])
        for a, b in zip(path[start:-1], path[start + 1:]):
            step = float(self._graph[a, b])
            minutes += step * self._factor if self._penalised[b] else step
        return minutes

    def _joined_seed(self, cell: int) -> bool:
        return int(self._after_predecessors[cell]) == len(self._joined)


def solve_staying_on_ways(graph, shape: tuple[int, int], origin_cell: tuple[int, int], *,
                          void: np.ndarray, on_way: np.ndarray, leave_way_factor: float,
                          keep_open: np.ndarray | None = None) -> WaySolve:
    """The least-time paths from `origin_cell`, in two stages, so a route
    that has reached a mapped way does not leave it for a shortcut.

    Before its first way cell a path walks `graph` as it is: getting from
    the location to a trail is what off-way walking is for. From its first
    way cell on, a step INTO a cell off every way costs `1 / leave_way_factor`
    times its minutes - a medevac does not leave a trail to save a minute or
    two, only to save many times what the ground off it costs. A cell in
    `keep_open` (an exit: a lot beside the road) is never counted over.

    The first stage solves `graph` with every way cell made a sink - no
    step out of it - so its minutes are those of paths that have not yet
    touched a way. The second solves the counted-over graph from a source
    node joined to every way cell the first reached, by an edge of the
    first stage's minutes to it. A cell's cost is the lower of the two."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    height, width = shape
    size = height * width
    way = np.asarray(on_way, dtype=bool).ravel()
    before, before_predecessors = solve_from(_rows_closed(graph, way), shape, origin_cell,
                                             void=void)
    flat = before.ravel()
    origin_index = int(origin_cell[0]) * width + int(origin_cell[1])
    penalised = ~way
    if keep_open is not None:
        penalised &= ~np.asarray(keep_open, dtype=bool).ravel()
    seeds = np.flatnonzero(way & np.isfinite(flat)).astype(graph.indices.dtype)
    data = np.where(penalised[graph.indices], graph.data / leave_way_factor, graph.data)
    counted = csr_matrix(
        (np.concatenate([data, flat[seeds]]),
         np.concatenate([graph.indices, seeds]),
         np.append(graph.indptr, graph.indptr[-1] + seeds.size).astype(graph.indptr.dtype)),
        shape=(size + 1, size + 1))
    del data
    if seeds.size:
        after, after_predecessors = dijkstra(counted, directed=True, indices=size,
                                             return_predecessors=True)
        after = np.asarray(after[:size], dtype="float64")
    else:
        after = np.full(size, np.inf)
        after_predecessors = np.full(size + 1, -9999, dtype=np.int32)
    joined = after < flat
    cost = np.where(joined, after, flat).reshape(height, width)
    # Prices again from the graph the second stage walked: its edges into a
    # penalised cell are divided by the factor, so they are multiplied back.
    return WaySolve(cost=cost, before=before, before_predecessors=before_predecessors,
                    after_predecessors=after_predecessors, joined=joined, graph=counted,
                    penalised=penalised, leave_way_factor=leave_way_factor,
                    origin_index=origin_index)


def _rows_closed(graph, closed: np.ndarray):
    """`graph` with every edge out of a `closed` cell removed."""
    from scipy.sparse import csr_matrix

    counts = np.diff(graph.indptr)
    keep = np.repeat(~closed, counts)
    indptr = np.concatenate([[0], np.cumsum(np.where(closed, 0, counts))])
    return csr_matrix((graph.data[keep], graph.indices[keep],
                       indptr.astype(graph.indptr.dtype)), shape=graph.shape)

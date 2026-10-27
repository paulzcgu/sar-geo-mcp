"""CalTopo interaction hub and geo-computing accessories for SAR.

Extracted from a SAR debrief automation project,
which is now this package's first consumer. The operating principles came
with the code and are load-bearing, not stylistic:

- A wrong coordinate is worse than no coordinate. Every parser here fails
  to `None` rather than to a plausible guess - "no location recorded" sends
  a planner to ask someone, while a transposed pair sends a team to the
  wrong drainage.
- Claims and computations are recorded side by side, never merged.
- Distances are true ground metres. Web Mercator inflates by 1/cos(lat),
  and that bug has been reintroduced twice; see `geo.py`.
"""

import logging as _logging

# httpx logs every request's full URL at INFO. A CalTopo WMS URL carries the
# account token in its path, and the Mapbox and Google static-image URLs carry
# their keys in the query string, so an INFO line is a published credential
# (seen in a dev server's log on 2026-09-23). Quieted here, on import, because
# every process that talks to those services imports this package first.
for _name in ("httpx", "httpcore"):
    _logging.getLogger(_name).setLevel(_logging.WARNING)
